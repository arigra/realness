"""Explicit tensor-domain noise laws and measured quadrature covariance.

Empirical IID sampling preserves a measured marginal, not spatial correlation.
No law is inferred from variance alone. All parameters use linear tensor units.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np


@lru_cache(maxsize=4)
def _load_bank(path, mtime_ns):
    with np.load(path, allow_pickle=False) as data:
        bank = np.array(data["samples"]).reshape(-1)
    if len(bank) < 2 or not np.isfinite(bank).all():
        raise ValueError("Empirical noise bank requires at least two finite samples")
    bank.setflags(write=False)
    return bank


@lru_cache(maxsize=4)
def _load_spectrum(path, mtime_ns):
    with np.load(path, allow_pickle=False) as data:
        axes = tuple(np.asarray(data[k], dtype=float) for k in ("range", "doppler", "azimuth"))
    for a in axes:
        if a.ndim != 1 or not np.isfinite(a).all() or np.any(a < 0) or not np.isclose(a.mean(), 1.0):
            raise ValueError("Noise spectrum axes must be finite, nonnegative and mean-one")
    return axes


def coloured_gaussian(shape, std, spectra, rng, mean=0j):
    """Circular complex Gaussian noise with a separable measured power spectrum per axis.

    White unit-variance noise is filtered in the 3D Fourier domain by sqrt(S_r x S_d x S_a);
    mean-one spectra keep the per-bin variance std**2 while giving the measured
    neighbour correlation (autocorrelation = inverse FFT of the spectrum, circular).
    """
    if tuple(len(a) for a in spectra) != tuple(shape):
        raise ValueError("Noise spectrum lengths must equal the tensor shape")
    white = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
    gain = np.sqrt(spectra[0][:, None, None] * spectra[1][None, :, None] * spectra[2][None, None, :])
    return np.fft.ifftn(np.fft.fftn(white) * gain) * std + mean


def sample_noise(config, shape, rng):
    complex_mode = config.representation == "complex"
    model = getattr(config, "noise_model", "gaussian")
    if model == "none":
        return np.zeros(shape, dtype=np.complex128 if complex_mode else np.float64)
    if model == "empirical":
        path = getattr(config, "noise_bank_path", None)
        if not path:
            raise ValueError("Empirical noise requires noise_bank_path")
        path = Path(path).resolve()
        bank = _load_bank(str(path), path.stat().st_mtime_ns)
        if np.iscomplexobj(bank) != complex_mode:
            raise ValueError("Noise bank representation does not match real/complex sensor response")
        return bank[rng.integers(0, len(bank), size=shape)]
    if model == "coloured_gaussian":
        if not complex_mode:
            raise ValueError("Coloured Gaussian noise is defined for complex field tensors")
        path = getattr(config, "noise_spectrum_path", None)
        if not path:
            raise ValueError("Coloured Gaussian noise requires noise_spectrum_path")
        path = Path(path).resolve()
        spectra = _load_spectrum(str(path), path.stat().st_mtime_ns)
        mean = complex(getattr(config, "noise_mean_real", 0.0), getattr(config, "noise_mean_imag", 0.0))
        return coloured_gaussian(shape, config.noise_std, spectra, rng, mean)
    if model != "gaussian":
        raise ValueError(f"Unknown noise law {model!r}")
    mean_real = getattr(config, "noise_mean_real", 0.0)
    if not complex_mode:
        return rng.normal(mean_real, config.noise_std, size=shape)
    mean_imag = getattr(config, "noise_mean_imag", 0.0)
    covariance = getattr(config, "noise_covariance", None)
    if covariance is None:
        noise = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) * config.noise_std / np.sqrt(2)
        return noise + complex(mean_real, mean_imag)
    values = rng.multivariate_normal([mean_real, mean_imag], np.asarray(covariance), size=shape,
                                     check_valid="raise")
    return values[..., 0] + 1j * values[..., 1]
