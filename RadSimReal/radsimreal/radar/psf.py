"""Measured PSF calibration (Appendix A) and explicitly uncalibrated demo PSF.

All kernels have odd dimensions and their origin at shape//2. Energy below
means squared L2 norm, an explicit convention because the paper does not define
its discrete energy calculation or crop-search algorithm. Kernels are not
silently renormalized after truncation.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def _validate_kernel(kernel):
    kernel = np.asarray(kernel)
    if kernel.ndim != 3 or any(n % 2 != 1 for n in kernel.shape):
        raise ValueError("PSF must be a 3D odd-sized centered kernel")
    if not np.isfinite(kernel).all() or not np.any(np.abs(kernel) > 0):
        raise ValueError("PSF must be finite and have nonzero energy")
    return kernel


def analytic_psf(shape=(9, 5, 21), width_bins=(1.0, 1.0, 3.0), representation="intensity"):
    """Uncalibrated separable sinc PSF, NOT a measured TI sensor response.

    Width is the distance from the origin to the first null in each coordinate.
    Intensity mode uses sinc**2; field mode retains signed sinc amplitudes.
    FFT windowing, coupling and sensor-dependent sidelobes require measurement.
    """
    if len(shape) != 3 or len(width_bins) != 3:
        raise ValueError("shape and widths require three axes")
    if any(int(n) != n or n < 1 or n % 2 != 1 for n in shape):
        raise ValueError("PSF dimensions must be positive odd integers")
    if not np.isfinite(width_bins).all() or np.any(np.asarray(width_bins) <= 0):
        raise ValueError("PSF widths must be finite and positive")
    axes = [np.sinc((np.arange(n) - n // 2) / w) for n, w in zip(shape, width_bins)]
    kernel = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    if representation == "intensity":
        return kernel ** 2
    if representation in ("field", "complex"):
        return kernel.astype(np.complex128)
    raise ValueError("representation must be intensity or field")


def truncate_psf(psf, energy_fraction=0.99):
    """Centered symmetric rectangular crop retaining at least requested energy.

    Greedily expands the crop by the axis offering the largest added energy per
    added cell. The threshold is guaranteed; global minimum volume is not claimed.
    """
    psf = _validate_kernel(psf)
    if not 0 < energy_fraction <= 1:
        raise ValueError("energy_fraction must be in (0, 1]")
    # Scaling cancels from every energy ratio and avoids overflow/underflow for
    # a finite kernel supplied in very large or very small calibration units.
    magnitude_scale = max(float(np.max(np.abs(np.real(psf)))), float(np.max(np.abs(np.imag(psf)))))
    energy = np.abs(psf / magnitude_scale) ** 2
    total = float(energy.sum())
    center = np.asarray(psf.shape) // 2
    radius = np.zeros(3, dtype=int)

    def slices(r):
        return tuple(slice(int(c - v), int(c + v + 1)) for c, v in zip(center, r))

    retained = float(energy[slices(radius)].sum())
    while retained / total + 1e-14 < energy_fraction:
        candidates = []
        current_volume = np.prod(2 * radius + 1)
        for axis in range(3):
            if radius[axis] < center[axis]:
                proposal = radius.copy()
                proposal[axis] += 1
                mass = float(energy[slices(proposal)].sum())
                extra_cells = np.prod(2 * proposal + 1) - current_volume
                candidates.append(((mass - retained) / extra_cells, mass, axis))
        if not candidates:
            break
        _, retained, axis = max(candidates)
        radius[axis] += 1
    kernel = psf[slices(radius)].copy()
    return kernel, {
        "original_shape": list(psf.shape), "shape": list(kernel.shape),
        "retained_energy_fraction": float(np.sum(energy[slices(radius)]) / total),
        "requested_energy_fraction": float(energy_fraction),
        "energy_definition": "sum(abs(kernel)**2)",
        "origin": (np.asarray(kernel.shape) // 2).tolist(),
        "normalization": "unchanged",
    }


def estimate_noise(samples):
    """Appendix A noise moments in a supplied reflection-free linear ROI.

    Complex variance is E|n-E[n]|², the TOTAL two-quadrature variance. The
    population variance (ddof=0) is used. Gaussianity is not inferred by this
    measurement; the simulator's Gaussian law is a separate modeling choice.
    """
    values = np.asarray(samples).ravel()
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("need at least two finite reflection-free samples")
    mean = values.mean()
    variance = float(np.mean(np.abs(values - mean) ** 2))
    return {"mean_real": float(np.real(mean)), "mean_imag": float(np.imag(mean)),
            "variance": variance, "std": float(np.sqrt(variance)),
            "count": int(values.size), "variance_ddof": 0}


def measure_psf(tensors, center, half_widths=None, energy_fraction=0.99,
                noise_variance=None, normalize_peak=False):
    """Appendix A stationary-target averaging and centered 3D PSF extraction.

    Input shape is (frames,range,doppler,azimuth), linear response (not dB).
    Complex frames must be phase-stable/aligned; no phase alignment is invented.
    User half_widths explicitly choose support. Otherwise noise variance selects
    support from three axis slices through the target where squared response
    exceeds 0.1*variance; when absent, the largest centered symmetric available
    support is used. Axis slices prevent an unrelated off-axis reflector from
    expanding the crop, but still require an isolated target and enough frames
    to reduce noise. The final L2 energy fraction is relative to this selected
    support, not an assertion that all unmeasured sensor sidelobes are included.
    """
    tensors = np.asarray(tensors)
    if tensors.ndim != 4 or tensors.shape[0] < 1 or not np.isfinite(tensors).all():
        raise ValueError("calibration tensors must be finite (frames,R,D,A)")
    if len(center) != 3 or any(int(v) != v for v in center):
        raise ValueError("center must contain three integer bin indices")
    center = np.asarray(center, dtype=int)
    bounds = np.asarray(tensors.shape[1:])
    if np.any(center < 0) or np.any(center >= bounds):
        raise ValueError("target center outside calibration tensor")
    average = tensors.mean(axis=0)
    max_radius = np.minimum(center, bounds - center - 1)
    if half_widths is None:
        radius = max_radius.copy()
        if noise_variance is not None:
            if not np.isfinite(noise_variance) or noise_variance < 0:
                raise ValueError("noise_variance must be finite and nonnegative")
            for axis in range(3):
                line = list(center)
                line[axis] = slice(None)
                profile = np.abs(average[tuple(line)]) ** 2
                offsets = np.flatnonzero(profile > 0.1 * noise_variance) - center[axis]
                radius[axis] = min(int(np.max(np.abs(offsets))) if len(offsets) else 0,
                                   int(max_radius[axis]))
    else:
        radius = np.asarray(half_widths)
        if radius.shape != (3,) or np.any(radius < 0) or np.any(radius != radius.astype(int)):
            raise ValueError("half_widths must contain three nonnegative integers")
        if np.any(radius > max_radius):
            raise ValueError("requested symmetric PSF crop exceeds tensor bounds")
        radius = radius.astype(int)
    region = tuple(slice(int(c - w), int(c + w + 1)) for c, w in zip(center, radius))
    kernel = average[region].copy()
    peak_scale = float(np.abs(kernel[tuple(radius)]))
    if normalize_peak:
        if peak_scale <= 0:
            raise ValueError("cannot normalize a zero center response")
        kernel /= peak_scale
    kernel, metadata = truncate_psf(kernel, energy_fraction)
    metadata.update({"source": "stationary-target calibration (Appendix A)",
                     "frames_averaged": int(tensors.shape[0]), "target_bin": center.tolist(),
                     "noise_variance": noise_variance, "peak_normalized": bool(normalize_peak),
                     "pre_normalization_center_magnitude": peak_scale,
                     "threshold_convention": "target-centered axis slices: abs(response)**2 > 0.1*noise_variance",
                     "energy_reference": "selected candidate PSF support before final truncation"})
    return kernel, metadata


def save_psf(path, kernel, metadata=None):
    kernel = _validate_kernel(kernel)
    np.savez_compressed(path, psf=kernel, metadata=json.dumps(metadata or {}))


def load_psf(path, return_metadata=False, config=None, strict_metadata=False):
    """Load a kernel and optionally verify sensor-coordinate metadata.

    Store metadata['grid'] with range_step, velocity_step, angular_step,
    azimuth_coordinate and representation when calibrating. Passing config
    checks available values; strict_metadata additionally requires all five.
    Legacy NPY or metadata-free NPZ is allowed only without strict checking.
    Grid matching does not establish spatial PSF invariance or RF calibration.
    """
    path = Path(path)
    metadata = {}
    if path.suffix.lower() == ".npy":
        kernel = _validate_kernel(np.load(path, allow_pickle=False))
    else:
        with np.load(path, allow_pickle=False) as archive:
            kernel = _validate_kernel(archive["psf"])
            if "metadata" in archive:
                metadata = json.loads(str(archive["metadata"]))
                if not isinstance(metadata, dict):
                    raise ValueError("PSF metadata must be a JSON mapping")
    if strict_metadata and config is None:
        raise ValueError("strict sensor-grid validation requires a radar config")
    if config is not None:
        grid = metadata.get("grid", {})
        if not isinstance(grid, dict):
            raise ValueError("PSF grid metadata must be a mapping")
        for key in ("range_step", "velocity_step", "angular_step", "azimuth_coordinate", "representation"):
            if key not in grid:
                if strict_metadata:
                    raise ValueError(f"PSF missing required sensor-grid metadata: {key}")
                continue
            expected, actual = getattr(config, key), grid[key]
            if isinstance(expected, str):
                matches = actual == expected
            else:
                try:
                    matches = bool(np.isclose(float(actual), float(expected), rtol=1e-7, atol=1e-12))
                except (TypeError, ValueError):
                    matches = False
            if not matches:
                raise ValueError(f"PSF sensor-grid mismatch for {key}: file={actual}, config={expected}")
    return (kernel, metadata) if return_metadata else kernel
