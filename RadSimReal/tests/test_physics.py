"""Independent numerical and geometry checks, including Appendix B equivalence."""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.signal import fftconvolve

from radsimreal.config import RadarConfig, SceneConfig
from radsimreal.radar import (
    analytic_psf, truncate_psf, measure_psf, estimate_noise, sparse_convolve,
    dense_convolve, cartesian_to_polar, polar_to_cartesian, points_to_indices,
    simulate, matched_filter_reference, load_psf, save_psf,
)
from radsimreal.simulation import (
    ReflectionPoints, Scene, Vehicle, sample_scene, generate_reflections,
    save_reflections, load_reflections,
)


class PhysicsTests(unittest.TestCase):
    def test_coordinate_round_trip(self):
        rng = np.random.default_rng(7)
        x, y = rng.uniform(-50, 50, 300), rng.uniform(1, 100, 300)
        r, theta = cartesian_to_polar(x, y)
        xx, yy = polar_to_cartesian(r, theta)
        assert_allclose(xx, x, atol=2e-14)
        assert_allclose(yy, y, atol=2e-14)
        assert_allclose(cartesian_to_polar(1, 1)[1], np.pi / 4)

    def test_sparse_dense_equivalence_including_edges_and_duplicate_points(self):
        rng = np.random.default_rng(1)
        shape = (15, 9, 17)
        psf = rng.normal(size=(5, 3, 7))
        indices = np.asarray([[0, 0, 0], [14, 8, 16], [5, 4, 8], [5, 4, 8]])
        coefficients = rng.normal(size=4)
        sparse = sparse_convolve(indices, coefficients, psf, shape)
        dense = dense_convolve(indices, coefficients, psf, shape)
        assert_allclose(sparse, dense, atol=4e-15)

    def test_fast_sparse_matches_loop_reference_with_out_of_domain_points(self):
        from radsimreal.radar.formation import _sparse_convolve_loop
        rng = np.random.default_rng(3)
        shape = (20, 8, 30)
        for psf_shape in [(5, 3, 7), (3, 1, 9), (1, 1, 1), (15, 1, 31)]:
            psf = rng.normal(size=psf_shape) + 1j * rng.normal(size=psf_shape)
            indices = rng.uniform([-6, -3, -20], [26, 11, 50], size=(400, 3))
            coefficients = rng.normal(size=400) + 1j * rng.normal(size=400)
            for interpolation in ("nearest", "trilinear"):
                fast = sparse_convolve(indices, coefficients, psf, shape, interpolation)
                loop = _sparse_convolve_loop(indices, coefficients, psf, shape, interpolation)
                assert_allclose(fast, loop, rtol=0, atol=1e-11 * np.abs(loop).max())

    def test_complex_convolution_preserves_cancellation(self):
        psf = analytic_psf((3, 3, 5), (1.2, 1.2, 2), "complex")
        a = sparse_convolve([[4, 3, 4], [4, 3, 4]], [1 + 1j, -1 - 1j], psf, (9, 7, 9))
        assert_allclose(a, 0, atol=1e-15)

    def test_offgrid_trilinear_dense_sparse_agreement(self):
        indices = [[2.1, 3.3, 4.7], [6.9, 4.2, 8.3]]
        psf = analytic_psf((3, 3, 5), (1.3, 1.5, 2.1))
        sparse = sparse_convolve(indices, [2, 3], psf, (12, 9, 14), "trilinear")
        dense = dense_convolve(indices, [2, 3], psf, (12, 9, 14), "trilinear")
        assert_allclose(sparse, dense, atol=2e-15)
        impulses = sparse_convolve(indices, [2, 3], np.ones((1, 1, 1)), (12, 9, 14), "trilinear")
        assert_allclose(impulses.sum(), 5)

    def test_generic_matched_filter_equals_psf_convolution(self):
        rng = np.random.default_rng(9)
        grid = np.zeros((7, 5, 9), dtype=complex)
        grid[1, 2, 3], grid[5, 4, 7] = 1 + 2j, -0.5 + 0.1j
        waveform = rng.normal(size=(4, 3, 5)) + 1j * rng.normal(size=(4, 3, 5))
        received, matched, psf = matched_filter_reference(grid, waveform)
        self.assertEqual(received.shape, (10, 7, 13))
        expected = fftconvolve(grid, psf, mode="full")
        assert_allclose(matched, expected, atol=1e-13)

    def test_psf_truncation_energy_and_origin(self):
        psf = analytic_psf((17, 11, 41), (1.6, 1.4, 4.0))
        cropped, metadata = truncate_psf(psf, 0.99)
        self.assertTrue(all(n % 2 == 1 for n in cropped.shape))
        self.assertGreaterEqual(metadata["retained_energy_fraction"], 0.99 - 1e-14)
        self.assertLess(np.prod(cropped.shape), np.prod(psf.shape))
        assert_allclose(cropped[tuple(np.asarray(cropped.shape) // 2)], 1.0)
        self.assertLessEqual(np.sum(abs(cropped) ** 2), np.sum(abs(psf) ** 2))

    def test_psf_crop_energy_is_invariant_to_extreme_finite_scale(self):
        psf = analytic_psf((7, 5, 15), (1.5, 1.5, 3.2))
        ordinary, baseline = truncate_psf(psf, 0.99)
        for scale in (1e200, 1e-200):
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                scaled, metadata = truncate_psf(psf * scale, 0.99)
            self.assertEqual(scaled.shape, ordinary.shape)
            assert_allclose(scaled / scale, ordinary, atol=1e-15)
            self.assertAlmostEqual(metadata["retained_energy_fraction"], baseline["retained_energy_fraction"])

    def test_measured_psf_and_noise(self):
        rng = np.random.default_rng(5)
        target = analytic_psf((7, 5, 15), (1.5, 1.2, 3))
        frames = target[None] + rng.normal(0, 0.005, (100, *target.shape))
        measured, metadata = measure_psf(frames, (3, 2, 7), energy_fraction=1.0)
        self.assertEqual(metadata["frames_averaged"], 100)
        assert_allclose(measured, target, atol=0.002)
        noise = rng.normal(0, 0.2, 100000)
        moments = estimate_noise(noise)
        self.assertAlmostEqual(moments["variance"], 0.04, delta=0.001)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "psf.npz"
            save_psf(path, measured, metadata)
            assert_allclose(load_psf(path), measured)

    def test_known_grid_coordinate_and_radial_velocity(self):
        cfg = RadarConfig()
        index = (20, 9, 50)
        theta = cfg.azimuths_rad[index[2]]
        x, y = polar_to_cartesian(cfg.ranges[index[0]], theta)
        velocity = cfg.velocities[index[1]] * np.asarray([[np.sin(theta), np.cos(theta), 0]])
        found, valid = points_to_indices([[x, y, 0]], velocity, cfg)
        self.assertTrue(valid[0])
        assert_allclose(found[0], index, atol=1e-13)

    def test_calibration_ignores_unrelated_off_axis_reflector(self):
        frames = np.zeros((10, 31, 17, 41))
        frames[:, 15, 8, 20] = 1
        frames[:, 2, 2, 3] = 10  # Not part of target-centered axis slices.
        measured, metadata = measure_psf(frames, (15, 8, 20), noise_variance=0.01)
        self.assertEqual(measured.shape, (1, 1, 1))
        assert_allclose(measured, 1)

    def test_psf_sensor_grid_metadata_validation(self):
        cfg = RadarConfig()
        grid = {key: getattr(cfg, key) for key in
                ("range_step", "velocity_step", "angular_step", "azimuth_coordinate", "representation")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "psf.npz"
            save_psf(path, np.ones((1, 1, 1)), {"grid": grid})
            kernel, metadata = load_psf(path, return_metadata=True, config=cfg, strict_metadata=True)
            self.assertEqual(metadata["grid"], grid)
            assert_allclose(kernel, 1)
            with self.assertRaises(ValueError):
                load_psf(path, config=replace(cfg, range_step=1.0), strict_metadata=True)
            save_psf(path, np.ones((1, 1, 1)))
            with self.assertRaises(ValueError):
                load_psf(path, config=cfg, strict_metadata=True)

    def test_empty_scene_noise_and_doppler_max(self):
        cfg = RadarConfig(noise_std=0.1)
        refs = ReflectionPoints(np.empty((0, 3)), np.empty((0, 3)), np.empty(0))
        result = simulate(refs, cfg, rng=np.random.default_rng(2))
        self.assertEqual(result.tensor.shape, cfg.shape)
        self.assertEqual(result.image.shape, (cfg.range_bins, cfg.azimuth_bins))
        assert_allclose(result.image, np.maximum(result.tensor, 0).max(axis=1), atol=2e-8)
        self.assertAlmostEqual(float(result.tensor.var()), 0.01, delta=0.0002)
        complex_result = simulate(refs, replace(cfg, representation="complex"), rng=np.random.default_rng(2))
        self.assertAlmostEqual(float(np.mean(np.abs(complex_result.tensor) ** 2)), 0.01, delta=0.0002)
        assert_allclose(complex_result.image, (np.abs(complex_result.tensor) ** 2).max(axis=1), atol=1e-8)

    def test_scene_determinism_and_external_round_trip(self):
        radar, scene_cfg = RadarConfig(), SceneConfig()
        scene_a = sample_scene(radar, scene_cfg, np.random.default_rng(2024))
        scene_b = sample_scene(radar, scene_cfg, np.random.default_rng(2024))
        self.assertEqual(scene_a.to_dict(), scene_b.to_dict())
        self.assertEqual(Scene.from_dict(scene_a.to_dict()).to_dict(), scene_a.to_dict())
        refs = generate_reflections(scene_a, radar, scene_cfg, np.random.default_rng(6))
        self.assertGreater(len(refs), 0)
        self.assertTrue(np.isfinite(refs.coefficients).all())
        self.assertTrue((refs.coefficients >= 0).all())
        result = simulate(refs, radar, rng=np.random.default_rng(7))
        self.assertTrue(np.isfinite(result.image).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reflection.npz"
            save_reflections(path, refs)
            restored = load_reflections(path)
            assert_allclose(restored.xyz, refs.xyz)
            assert_allclose(restored.coefficients, refs.coefficients)

    def test_occlusion_removes_far_vehicle_samples(self):
        cfg = RadarConfig()
        scene_cfg = SceneConfig(points_per_vehicle=100, clutter_points=0)
        scene = Scene([Vehicle(0, 10, 4, 3, 0, 0, 0, object_id=0),
                       Vehicle(0, 20, 4, 2, 0, 0, 0, object_id=1)])
        refs = generate_reflections(scene, cfg, scene_cfg, np.random.default_rng(4))
        self.assertTrue(np.any(refs.object_ids == 0))
        self.assertFalse(np.any(refs.object_ids == 1))

    def test_simulation_rejects_float32_image_overflow(self):
        cfg = RadarConfig(noise_std=0)
        huge_intensity = ReflectionPoints([[0, 10, 0]], [[0, 0, 0]], [1e40])
        with self.assertRaisesRegex(FloatingPointError, "float32"):
            simulate(huge_intensity, cfg, psf=np.ones((1, 1, 1)), rng=np.random.default_rng(4))
        huge_field = ReflectionPoints([[0, 10, 0]], [[0, 0, 0]], [1e30 + 0j])
        with self.assertRaisesRegex(FloatingPointError, "float32"):
            simulate(huge_field, replace(cfg, representation="complex"),
                     psf=np.ones((1, 1, 1)), rng=np.random.default_rng(4))

    def test_intensity_rejects_negative_coefficients_but_allows_signed_psf(self):
        cfg = RadarConfig(noise_std=0)
        negative = ReflectionPoints([[0, 10, 0]], [[0, 0, 0]], [-1])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            simulate(negative, cfg, rng=np.random.default_rng(4))
        positive = ReflectionPoints([[0, 10, 0]], [[0, 0, 0]], [1])
        signed_psf = np.asarray([[[-0.05, 1, -0.05]]])
        result = simulate(positive, cfg, psf=signed_psf, rng=np.random.default_rng(4))
        self.assertTrue(np.isfinite(result.image).all())
        self.assertTrue((result.image >= 0).all())

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            sample_scene(RadarConfig(azimuth_min_deg=10, azimuth_max_deg=60), SceneConfig(), np.random.default_rng(3))
        with self.assertRaises(ValueError):
            analytic_psf((4, 3, 3), (1, 1, 1))
        with self.assertRaises(ValueError):
            truncate_psf(np.zeros((3, 3, 3)))
        with self.assertRaises(ValueError):
            ReflectionPoints([[np.nan, 1, 2]], [[0, 0, 0]], [1])
        with self.assertRaises(ValueError):
            estimate_noise([1])


if __name__ == "__main__":
    unittest.main()
