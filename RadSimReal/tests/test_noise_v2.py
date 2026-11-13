import numpy as np
import pytest
from radsimreal.config import RadarConfig
from radsimreal.radar.noise import sample_noise
from radsimreal.radar import simulate
from radsimreal.simulation import ReflectionPoints


def test_measured_quadrature_covariance_and_mean():
    config = RadarConfig(representation="complex", noise_covariance=((2, .7), (.7, 1)),
                         noise_mean_real=3, noise_mean_imag=-2)
    z = sample_noise(config, (200000,), np.random.default_rng(2))
    assert z.real.mean() == pytest.approx(3, abs=.015)
    assert z.imag.mean() == pytest.approx(-2, abs=.015)
    np.testing.assert_allclose(np.cov([z.real, z.imag], ddof=0), [[2, .7], [.7, 1]], atol=.02)


def test_empirical_noise_preserves_bank_support(tmp_path):
    p = tmp_path / "noise.npz"
    bank = np.array([1+2j, 3-4j, -5+6j])
    np.savez(p, samples=bank)
    config = RadarConfig(representation="complex", noise_model="empirical", noise_bank_path=str(p))
    z = sample_noise(config, (10000,), np.random.default_rng(3))
    assert set(z) == set(bank)
    assert abs(np.mean(z == bank[0]) - 1/3) < .02
    config.representation = "intensity"
    with pytest.raises(ValueError, match="representation"):
        sample_noise(config, (1,), np.random.default_rng(1))


def test_doppler_wrap_alias_and_kernel_support():
    cfg = RadarConfig(range_bins=8, doppler_bins=8, azimuth_bins=9, range_step=1,
                      velocity_min=-4, velocity_step=1, noise_model="none", doppler_wrap=True, psf_energy=1)
    # Radial speed +4 aliases to the native -4 bin, kernel spills to period end.
    points = ReflectionPoints(np.array([[0, 3, 0]]), np.array([[0, 4, 0]]), np.array([1.]))
    result = simulate(points, cfg, psf=np.ones((1, 3, 1)))
    np.testing.assert_allclose(result.tensor[3, :, 4], [1, 1, 0, 0, 0, 0, 0, 1])
    assert result.metadata["in_domain_reflection_count"] == 1


def test_invalid_covariance_rejected():
    with pytest.raises(ValueError, match="PSD"):
        RadarConfig(noise_covariance=((1, 2), (2, 1)))


def test_coloured_gaussian_keeps_variance_and_imposes_correlation():
    from radsimreal.radar.noise import coloured_gaussian
    shape = (64, 8, 128)
    # azimuth spectrum confined to 8 of 128 frequencies -> strongly correlated neighbours
    s_az = np.zeros(128); s_az[:4] = 1; s_az[-4:] = 1; s_az /= s_az.mean()
    spectra = (np.ones(64), np.ones(8), s_az)
    x = coloured_gaussian(shape, 3.0, spectra, np.random.default_rng(0))
    assert abs(np.mean(np.abs(x) ** 2) / 9.0 - 1) < 0.05
    lag1 = np.mean(x[:, :, :-1] * np.conj(x[:, :, 1:])) / np.mean(np.abs(x) ** 2)
    assert abs(lag1) > 0.9
    lag1_range = np.mean(x[:-1] * np.conj(x[1:])) / np.mean(np.abs(x) ** 2)
    assert abs(lag1_range) < 0.05
