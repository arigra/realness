"""RadSimReal Section 3 / Appendix B equations (4)-(5).

The sparse shifted-PSF sum is exactly the discrete convolution for integer
reflector indices and a centered PSF. Trilinear deposition of off-grid points is
an explicit discretization approximation. Boundaries are zero-extended, not
circular: the implementation does not synthesize velocity ambiguity.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.signal import fftconvolve

from .coordinates import points_to_indices
from .psf import analytic_psf, truncate_psf, _validate_kernel
from .noise import sample_noise


def matched_filter_reference(reflector_grid, waveform):
    """Generic linear reference for Appendix B Eqs. (1)-(3), full boundaries.

    This models a separable-delay linear system, not a sensor-specific FMCW
    waveform. Correlation uses the standard conjugate/reverse convention,
    clarifying the real-valued shorthand in the manuscript. Returns received
    signal, matched-filter output, and waveform autocorrelation PSF. Their
    convolution identity can be checked without proprietary hardware details.
    """
    reflector_grid = np.asarray(reflector_grid)
    waveform = np.asarray(waveform)
    if reflector_grid.ndim != 3 or waveform.ndim != 3:
        raise ValueError("reference arrays must be three-dimensional")
    if not np.isfinite(reflector_grid).all() or not np.isfinite(waveform).all():
        raise ValueError("reference arrays must be finite")
    matched_kernel = np.conjugate(waveform[::-1, ::-1, ::-1])
    received = fftconvolve(reflector_grid, waveform, mode="full")
    output = fftconvolve(received, matched_kernel, mode="full")
    psf = fftconvolve(waveform, matched_kernel, mode="full")
    return received, output, psf


def _deposits(indices, coefficients, interpolation):
    indices = np.asarray(indices, dtype=float)
    coefficients = np.asarray(coefficients)
    if indices.ndim != 2 or indices.shape[1] != 3 or coefficients.shape != (len(indices),):
        raise ValueError("indices must be (N,3), coefficients (N,)")
    if not np.isfinite(indices).all() or not np.isfinite(coefficients).all():
        raise ValueError("reflectors must be finite")
    if interpolation == "nearest":
        yield np.floor(indices + 0.5).astype(int), coefficients
    elif interpolation == "trilinear":
        base = np.floor(indices).astype(int)
        fraction = indices - base
        for offsets in product((0, 1), repeat=3):
            offsets = np.asarray(offsets)
            weight = np.prod(np.where(offsets[None, :] == 1, fraction, 1 - fraction), axis=1)
            keep = weight > 0
            yield base[keep] + offsets, coefficients[keep] * weight[keep]
    else:
        raise ValueError("interpolation must be nearest or trilinear")


def sparse_convolve(indices, coefficients, psf, shape, interpolation="nearest"):
    """Equation (4): sum shifted/scaled PSFs, with cropped finite boundaries.

    Same sum as _sparse_convolve_loop, computed as one scatter-add of all deposits
    onto a grid padded by the PSF support (so out-of-domain reflectors whose PSF
    reaches the domain still contribute), a full linear convolution, and a crop.
    """
    psf = _validate_kernel(psf)
    shape = tuple(int(n) for n in shape)
    if len(shape) != 3 or any(n < 1 for n in shape):
        raise ValueError("output shape must contain three positive dimensions")
    dtype = np.result_type(psf.dtype, np.asarray(coefficients).dtype, np.float64)
    length = np.asarray(psf.shape)
    radius = length // 2
    pad_lo = length - 1 - radius  # positions p >= radius - L + 1 can reach index 0
    padded_shape = np.asarray(shape) + length - 1
    grid = np.zeros(tuple(padded_shape), dtype=dtype)
    for positions, amplitudes in _deposits(indices, coefficients, interpolation):
        q = np.asarray(positions) + pad_lo
        inside = ((q >= 0) & (q < padded_shape)).all(axis=1)
        np.add.at(grid, tuple(q[inside].T), np.asarray(amplitudes)[inside])
    if not grid.any():
        return np.zeros(shape, dtype=dtype)
    crop = tuple(slice(int(l - 1), int(l - 1 + n)) for l, n in zip(length, shape))
    full = fftconvolve(grid, psf, mode="full")[crop]
    # Exact zeros outside the PSF support, as in the shifted-PSF sum: integer-valued
    # occupancy (*) tap-indicator convolution; FFT round-off is far below 0.5.
    support = fftconvolve((grid != 0).astype(np.float64), (psf != 0).astype(np.float64), mode="full")[crop] > 0.5
    full[~support] = 0
    return np.ascontiguousarray(full).astype(dtype, copy=False)


def _sparse_convolve_loop(indices, coefficients, psf, shape, interpolation="nearest"):
    """Reference implementation of Equation (4): one shifted PSF per deposit (slow)."""
    psf = _validate_kernel(psf)
    shape = tuple(int(n) for n in shape)
    if len(shape) != 3 or any(n < 1 for n in shape):
        raise ValueError("output shape must contain three positive dimensions")
    dtype = np.result_type(psf.dtype, np.asarray(coefficients).dtype, np.float64)
    output = np.zeros(shape, dtype=dtype)
    radius = np.asarray(psf.shape) // 2
    for positions, amplitudes in _deposits(indices, coefficients, interpolation):
        for position, amplitude in zip(positions, amplitudes):
            lower = position - radius
            upper = lower + np.asarray(psf.shape)
            output_lower = np.maximum(lower, 0)
            output_upper = np.minimum(upper, shape)
            if np.any(output_lower >= output_upper):
                continue
            target = tuple(slice(int(a), int(b)) for a, b in zip(output_lower, output_upper))
            source = tuple(slice(int(a), int(b)) for a, b in zip(output_lower - lower, output_upper - lower))
            output[target] += amplitude * psf[source]
    return output


def dense_convolve(indices, coefficients, psf, shape, interpolation="nearest"):
    """Equation (5): independent dense FFT reference for in-domain deposits.

    Used to verify the sparse implementation; it is not a waveform simulator.
    Only in-domain reflector centers are accepted to avoid hiding edge effects.
    """
    psf = _validate_kernel(psf)
    grid = np.zeros(shape, dtype=np.result_type(psf, np.asarray(coefficients), np.float64))
    for positions, amplitudes in _deposits(indices, coefficients, interpolation):
        valid = ((positions >= 0) & (positions < np.asarray(shape))).all(axis=1)
        np.add.at(grid, tuple(positions[valid].T), amplitudes[valid])
    return fftconvolve(grid, psf, mode="same")


@dataclass
class SimulationResult:
    tensor: np.ndarray
    image: np.ndarray
    psf: np.ndarray
    metadata: dict


def simulate(reflections, config, psf=None, rng=None):
    """Section 3: 3D convolution, additive noise, then max over Doppler.

    Default intensity mode sums nonnegative reflection intensities and uses
    additive zero-mean Gaussian tensor noise, clamping only for image display.
    This is an explicit phenomenological representation: a power tensor does
    not generally obey the coherent linear-superposition proof. Field mode is
    the physical complex-field extension: complex PSF/coefficients, circular
    Gaussian noise, then magnitude squared before Doppler max. The manuscript
    does not specify the field/power convention or the noise distribution.
    Intensity coefficients must be nonnegative; a supplied real measured PSF
    may contain signed residual/noise samples. Float32 output overflow is
    rejected explicitly rather than saved as an infinite training image.
    """
    rng = np.random.default_rng() if rng is None else rng
    field_mode = config.representation in ("field", "complex")
    if config.representation not in ("intensity", "field", "complex"):
        raise ValueError("unknown signal representation")
    if not np.isfinite(config.noise_std) or config.noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    if psf is None:
        psf = analytic_psf(config.psf_shape, config.psf_width_bins, config.representation)
        source = "uncalibrated analytic separable sinc demo PSF"
    else:
        source = "user-supplied PSF; user must validate coordinate and response convention"
    psf, crop = truncate_psf(psf, config.psf_energy)
    if not field_mode and (np.iscomplexobj(psf) or np.iscomplexobj(reflections.coefficients)):
        raise ValueError("complex PSF/coefficients require field representation")
    if not field_mode and np.any(np.asarray(reflections.coefficients) < 0):
        raise ValueError("intensity representation requires nonnegative reflection coefficients")
    indices, valid = points_to_indices(reflections.xyz, reflections.velocity, config)
    centers = indices[valid]
    coefficients = reflections.coefficients[valid]
    tensor = sparse_convolve(centers, coefficients, psf, config.shape, "trilinear")
    if getattr(config, "doppler_wrap", False):
        if psf.shape[1] > config.doppler_bins:
            raise ValueError("Wrapped Doppler requires kernel width <= Doppler period")
        for shift in (-config.doppler_bins, config.doppler_bins):
            wrapped = centers.copy()
            wrapped[:, 1] += shift
            tensor += sparse_convolve(wrapped, coefficients, psf, config.shape, "trilinear")
    if field_mode:
        tensor = tensor.astype(np.complex128)
    tensor += sample_noise(config, config.shape, rng)
    intensity = np.abs(tensor) ** 2 if field_mode else np.maximum(tensor, 0)
    image = intensity.max(axis=1)
    if not np.isfinite(tensor).all() or not np.isfinite(image).all():
        raise FloatingPointError("simulation produced a nonfinite result")
    if np.any(image > np.finfo(np.float32).max):
        raise FloatingPointError("radar image exceeds float32 storage range; recalibrate coefficient/PSF scale")
    image = image.astype(np.float32)
    if not np.isfinite(image).all():
        raise FloatingPointError("radar image became nonfinite during float32 conversion")
    return SimulationResult(tensor=tensor, image=image, psf=psf, metadata={
        "method": "RadSimReal shifted PSF convolution, Appendix B Eq. (4)-(5)",
        "axis_order": ["range", "doppler_velocity", "azimuth_coordinate"],
        "representation": config.representation, "psf_source": source,
        "psf_crop": crop, "reflection_count": int(len(reflections.coefficients)),
        "in_domain_reflection_count": int(valid.sum()), "interpolation": "trilinear (off-grid approximation)",
        "boundary": "zero range/angle; periodic Doppler" if getattr(config, "doppler_wrap", False) else "zero extension",
        "doppler_wrap": bool(getattr(config, "doppler_wrap", False)),
        "noise": getattr(config, "noise_model", "gaussian"),
        "noise_correlation": "IID; measured quadrature covariance supported, spatial covariance not modeled",
        "noise_std": float(config.noise_std), "doppler_reduction": "maximum intensity",
        "angular_coordinate": config.azimuth_coordinate,
        "calibrated": False,
    })
