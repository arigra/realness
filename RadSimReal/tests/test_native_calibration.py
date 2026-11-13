"""Realistic native-file calibration fixtures; no real recording validation."""
import json
from pathlib import Path
import numpy as np
import pytest

from radsimreal.config import load_config
from radsimreal.native import NativeReader, reference_profile
from radsimreal.native.calibration import calibrate_native, noise_diagnostics
from radsimreal.radar import load_psf
from radsimreal.radar.noise import sample_noise


def native_stationary_fixture(root, *, phases=None, frames=8):
    """Write noisy coherent target captures in native R,A,D files."""
    profile = reference_profile("raddet", (24, 16, 12), "complex")
    profile.range_step = 1.0
    covariance = np.array([[.04, .015], [.015, .09]])
    mean = np.array([.2, .1])
    rng = np.random.default_rng(93)
    r, d, a = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), np.arange(-3, 4), indexing="ij")
    psf = np.exp(-(r * r + d * d + (a / 1.4) ** 2))
    center = np.array([12, 6, 8])
    phases = np.zeros(frames) if phases is None else np.asarray(phases)
    for i in range(frames):
        quadratures = rng.multivariate_normal(mean, covariance, size=profile.shape)
        canonical = quadratures[..., 0] + 1j * quadratures[..., 1]
        canonical[10:15, 4:9, 5:12] += 10 * np.exp(1j * phases[i]) * psf
        folder = root / "train/RAD/part1"
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / f"{i:06d}.npy", canonical.transpose(0, 2, 1)[::-1].astype(np.complex64))
    reader = NativeReader(root, "raddet", profile)
    ids = [record.id for record in reader.records]
    selection = {"targets": [{"record_id": record.id, "center": center.tolist(), "half_widths": [2, 2, 3]}
                               for record in reader.records],
                 "noise_regions": [{"record_id": record.id, "bounds": [[0, 4], [0, 12], [0, 16]]}
                                     for record in reader.records],
                 "target_is_isolated": True, "sensor_target_stationary": True,
                 "domain_verified": True, "energy_fraction": .99}
    return reader, selection, set(ids), covariance, mean, psf


def test_native_complex_calibration_covariance_profile_and_noise_bank_roundtrip(tmp_path):
    reader, selection, allowed, covariance, mean, _ = native_stationary_fixture(tmp_path / "native")
    output = tmp_path / "calibration"
    config_path = calibrate_native(reader, selection, output, allowed_ids=allowed, max_noise_samples=10000)
    config = load_config(config_path)
    assert config.radar.representation == "complex"
    assert config.radar.noise_model == "empirical"
    assert config.radar.doppler_wrap is True
    assert Path(config.radar.noise_bank_path).is_file()
    kernel, meta = load_psf(config.radar.psf_path, return_metadata=True,
                            config=config.radar, strict_metadata=True)
    assert kernel.ndim == 3
    assert all(n % 2 for n in kernel.shape)
    assert kernel[tuple(np.array(kernel.shape) // 2)] == pytest.approx(1 + 0j)
    assert meta["retained_energy_fraction"] >= .99
    assert meta["native_profile"] == json.loads(json.dumps(reader.profile.to_dict()))
    assert meta["frames_averaged"] == 8
    assert meta["selection_status"] == "user_confirmed_target_assumptions"
    assert meta["complete_sensor_calibration"] is False
    np.testing.assert_allclose(meta["noise"]["quadrature_covariance"], covariance, rtol=.15, atol=.004)
    np.testing.assert_allclose([meta["noise"]["mean_real"], meta["noise"]["mean_imag"]], mean, atol=.025)
    assert config.radar.noise_std ** 2 == pytest.approx(np.trace(meta["noise"]["quadrature_covariance"]))
    bank = np.load(output / "noise_bank.npz")["samples"]
    sampled = sample_noise(config.radar, (10, 4, 3), np.random.default_rng(17))
    assert np.all(np.isin(sampled, bank))
    assert (output / "calibration_diagnostics.png").is_file()


def test_calibration_rejects_frame_outside_declared_training_partition(tmp_path):
    reader, selection, allowed, *_ = native_stationary_fixture(tmp_path / "native", frames=2)
    wrong = dict(selection)
    wrong["targets"] = [dict(selection["targets"][0], record_id=reader.records[1].id)]
    with pytest.raises(ValueError, match="not in the declared"):
        calibrate_native(reader, wrong, tmp_path / "bad", allowed_ids={reader.records[0].id})


def test_calibration_requires_manifest_ids_and_rejects_ambiguous_integer_id(tmp_path):
    reader, selection, allowed, *_ = native_stationary_fixture(tmp_path / "native", frames=1)
    with pytest.raises(ValueError, match="allowed_ids"):
        calibrate_native(reader, selection, tmp_path / "no_partition")
    selection["noise_regions"][0]["record_id"] = 0
    with pytest.raises(ValueError, match="full string record_id"):
        calibrate_native(reader, selection, tmp_path / "ambiguous", allowed_ids=allowed)


def test_phase_alignment_is_explicit_and_recorded(tmp_path):
    reader, selection, allowed, *_ = native_stationary_fixture(tmp_path / "native", phases=[0, np.pi, 0, np.pi], frames=4)
    output = tmp_path / "aligned"
    path = calibrate_native(reader, selection, output, allowed_ids=allowed, phase_align=True)
    cfg = load_config(path)
    kernel, meta = load_psf(cfg.radar.psf_path, return_metadata=True)
    assert meta["phase_aligned_to_center"]
    assert meta["unadjusted_center_phase_coherence"] < .1
    assert kernel[tuple(np.array(kernel.shape) // 2)] == pytest.approx(1 + 0j)


def test_unconfirmed_target_assumptions_stay_provisional(tmp_path):
    reader, selection, allowed, *_ = native_stationary_fixture(tmp_path / "native", frames=2)
    selection.pop("domain_verified")
    calibrate_native(reader, selection, tmp_path / "provisional", allowed_ids=allowed)
    meta = json.loads((tmp_path / "provisional/calibration.json").read_text())
    assert meta["selection_status"] == "provisional_target_assumptions_unconfirmed"
    assert meta["physical_response"] == "coherent_field"
    assert meta["spatial_noise_correlation_modeled"] is False


def test_gaussian_noise_uses_measured_non_circular_quadrature_covariance(tmp_path):
    reader, selection, allowed, covariance, mean, _ = native_stationary_fixture(tmp_path / "native")
    path = calibrate_native(reader, selection, tmp_path / "gaussian", allowed_ids=allowed, noise_model="gaussian")
    cfg = load_config(path)
    samples = sample_noise(cfg.radar, (50000,), np.random.default_rng(8))
    measured = np.cov(np.stack([samples.real, samples.imag]), ddof=0)
    np.testing.assert_allclose(measured, cfg.radar.noise_covariance, rtol=.04, atol=.002)
    assert samples.real.mean() == pytest.approx(cfg.radar.noise_mean_real, abs=.01)
    assert samples.imag.mean() == pytest.approx(cfg.radar.noise_mean_imag, abs=.01)


def test_calibrated_periodic_doppler_wrap_restores_psf_edge_tail(tmp_path):
    from radsimreal.radar import simulate
    from radsimreal.simulation import ReflectionPoints
    reader, selection, allowed, *_ = native_stationary_fixture(tmp_path / "native", frames=2)
    path = calibrate_native(reader, selection, tmp_path / "calibration", allowed_ids=allowed)
    cfg = load_config(path).radar
    cfg.noise_model = "none"
    kernel = load_psf(cfg.psf_path)
    point = ReflectionPoints(np.array([[0., 12., 0.]]),
                             np.array([[0., cfg.velocity_min, 0.]]), np.array([1. + 0j]))
    periodic = simulate(point, cfg, psf=kernel, rng=np.random.default_rng(4))
    cfg.doppler_wrap = False
    finite = simulate(point, cfg, psf=kernel, rng=np.random.default_rng(4))
    assert np.max(np.abs(periodic.tensor[:, -1, :])) > .01
    assert np.max(np.abs(finite.tensor[:, -1, :])) == 0
    assert periodic.metadata["boundary"] == "zero range/angle; periodic Doppler"


def test_noise_diagnostics_retains_correlation_warning():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(30, 5, 7))
    correlated = np.repeat(base, 2, axis=0)
    info = noise_diagnostics(correlated.reshape(-1), [correlated])
    assert info["adjacent_cell_real_correlation_R_D_A"]["0"] > .4
    assert "violate" in info["caution"]
