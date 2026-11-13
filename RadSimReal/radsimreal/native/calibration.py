"""Streaming stationary-target/empty-ROI calibration against native tensors.

Selection coordinates are CANONICAL (range,Doppler,azimuth) bin indices. Target
isolation and stationarity are scientific input assumptions, not properties a
bright voxel proves. A selection without confirmed observations stays provisional.
"""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from scipy import stats
from ..config import SimulationConfig, SceneConfig
from ..radar.psf import estimate_noise, save_psf, truncate_psf


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def record_for_id(reader, identifier):
    if isinstance(identifier, int):
        raise ValueError("Use the full string record_id from the manifest, not an integer index or frame number")
    matches = [r for r in reader.records if str(r.id) == str(identifier)]
    if len(matches) != 1:
        raise ValueError(f"Calibration record id must identify one frame: {identifier!r}")
    return matches[0]


def _roi(cube, bounds):
    if len(bounds) != 3:
        raise ValueError("ROI bounds require three [start,stop] pairs in canonical R,D,A order")
    pairs = np.asarray(bounds)
    if pairs.shape != (3, 2) or not np.isfinite(pairs).all() or np.any(pairs != pairs.astype(int)):
        raise ValueError("ROI bounds must be finite integers")
    if np.any(pairs[:, 0] < 0) or np.any(pairs[:, 1] > np.asarray(cube.shape)) or np.any(pairs[:, 1] <= pairs[:, 0]):
        raise ValueError("ROI bounds are outside the tensor or empty")
    return cube[tuple(slice(int(a), int(b)) for a, b in pairs)]


def _check_training_record(record, allowed_ids=None):
    if allowed_ids is not None and str(record.id) not in allowed_ids:
        raise ValueError(f"Calibration frame {record.id} is not in the declared training/calibration partition")
    if getattr(record, "split", None) == "test" and allowed_ids is None:
        raise ValueError("Calibration requires a verified training/calibration manifest; source frame is marked test")


def noise_diagnostics(samples, roi_arrays=()):
    values = np.asarray(samples).reshape(-1)
    info = estimate_noise(values)
    channels = [values.real, values.imag] if np.iscomplexobj(values) else [values]
    names = ["real", "imag"] if np.iscomplexobj(values) else ["real"]
    info["quadratures"] = {}
    for name, data in zip(names, channels):
        std = float(data.std())
        if std > 0 and len(data) >= 8:
            z = (data - data.mean()) / std
            sampled = z[::max(1, len(z) // 5000)][:5000]
            ks = stats.kstest(sampled, "norm")
            info["quadratures"][name] = {"skewness": float(stats.skew(data)),
                "excess_kurtosis": float(stats.kurtosis(data)), "gaussian_KS_statistic": float(ks.statistic),
                "gaussian_KS_pvalue_diagnostic_only": float(ks.pvalue)}
        else:
            info["quadratures"][name] = {"constant_or_insufficient_samples": True}
    if np.iscomplexobj(values):
        info["quadrature_covariance"] = np.cov(np.stack([values.real, values.imag]), ddof=0).tolist()
    correlations = {str(a): [] for a in range(3)}
    for roi in roi_arrays:
        roi = np.asarray(roi)
        for axis in range(3):
            if roi.shape[axis] < 2:
                continue
            a = np.take(roi, range(roi.shape[axis] - 1), axis=axis).ravel()
            b = np.take(roi, range(1, roi.shape[axis]), axis=axis).ravel()
            a, b = a - a.mean(), b - b.mean()
            denom = np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
            if denom > 0:
                correlations[str(axis)].append(float(np.real(np.vdot(a, b)) / denom))
    info["adjacent_cell_real_correlation_R_D_A"] = {k: float(np.mean(v)) if v else None for k, v in correlations.items()}
    info["caution"] = "KS p-values assume independent samples; correlated radar cells violate that assumption. Diagnostics do not prove a noise law."
    return info


def calibrate_native(reader, selection, output_dir, *, allowed_ids=None, noise_model="empirical",
                     phase_align=False, normalize_peak=True, max_noise_samples=200000):
    """Estimate a centered PSF and noise law, loading one source frame at a time.

    selection targets: [{record_id, center:[r,d,a], half_widths:[r,d,a]}].
    selection noise_regions: [{record_id, bounds:[[r0,r1],[d0,d1],[a0,a1]]}].
    All target crops have equal odd shape. Repeated frames must depict the same
    stationary isolated reflector. A per-target integer center permits explicitly
    recorded alignment, never automatic sub-bin phase/motion recovery.
    """
    if isinstance(selection, (str, Path)):
        selection = json.loads(Path(selection).read_text(encoding="utf-8"))
    if not isinstance(selection, dict) or not selection.get("targets") or not selection.get("noise_regions"):
        raise ValueError("Selection needs targets and noise_regions lists")
    if noise_model not in ("empirical", "gaussian"):
        raise ValueError("Noise model must be empirical or gaussian")
    if max_noise_samples < 8:
        raise ValueError("Need max_noise_samples >= 8")
    if allowed_ids is None:
        raise ValueError("Native calibration requires allowed_ids from a declared training/calibration manifest")
    allowed_ids = {str(x) for x in allowed_ids}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    noise_parts, roi_arrays = [], []
    rng = np.random.default_rng(2024)
    provenance = {"targets": [], "noise_regions": []}
    for region in selection["noise_regions"]:
        record = record_for_id(reader, region["record_id"])
        _check_training_record(record, allowed_ids)
        cube = reader.load_tensor(record)
        roi = _roi(cube, region["bounds"])
        if not np.isfinite(roi).all():
            raise ValueError("Noise ROI contains nonfinite values")
        # Equal bounded contribution per declared ROI, with deterministic sampling.
        limit = max(8, max_noise_samples // len(selection["noise_regions"]))
        flat = roi.reshape(-1)
        if len(flat) > limit:
            flat = flat[rng.choice(len(flat), limit, replace=False)]
        noise_parts.append(flat.copy())
        if roi.size <= 100000:
            roi_arrays.append(roi.copy())
        provenance["noise_regions"].append({"record_id": str(record.id), "bounds": region["bounds"],
                                           "samples_used": len(flat)})
    samples = np.concatenate(noise_parts)
    if len(samples) > max_noise_samples:
        samples = samples[rng.choice(len(samples), max_noise_samples, replace=False)]
    noise = noise_diagnostics(samples, roi_arrays)
    baseline = complex(noise["mean_real"], noise["mean_imag"]) if np.iscomplexobj(samples) else noise["mean_real"]
    total, peaks, frame_count = None, [], 0
    for target in selection["targets"]:
        record = record_for_id(reader, target["record_id"])
        _check_training_record(record, allowed_ids)
        cube = reader.load_tensor(record)
        center = np.asarray(target["center"])
        half = np.asarray(target.get("half_widths", [4, 2, 10]))
        if center.shape != (3,) or half.shape != (3,) or np.any(center != center.astype(int)) or np.any(half != half.astype(int)) or np.any(half < 0):
            raise ValueError("Target center and half_widths require three integer coordinates")
        center, half = center.astype(int), half.astype(int)
        patch = _roi(cube, np.column_stack([center - half, center + half + 1])).copy() - baseline
        if total is not None and total.shape != patch.shape:
            raise ValueError("All target patches must have the same dimensions")
        peak = patch[tuple(half)]
        if not np.isfinite(patch).all() or abs(peak) <= 0:
            raise ValueError("Target patch is nonfinite or its center response is zero")
        peaks.append(complex(peak))
        if phase_align:
            if not np.iscomplexobj(patch):
                raise ValueError("Phase alignment requires complex field tensors")
            patch *= np.exp(-1j * np.angle(peak))
        total = np.asarray(patch, dtype=np.complex128 if np.iscomplexobj(patch) else np.float64) if total is None else total + patch
        frame_count += 1
        provenance["targets"].append({"record_id": str(record.id), "center": center.tolist(), "half_widths": half.tolist()})
    averaged = total / frame_count
    midpoint = tuple(np.asarray(averaged.shape) // 2)
    center_gain = averaged[midpoint]
    if abs(center_gain) <= np.finfo(float).tiny:
        raise ValueError("Target averaging cancelled its center; inspect phase stability or phase-align explicitly")
    if normalize_peak:
        averaged /= center_gain
    requested_energy = float(selection.get("energy_fraction", 0.99))
    kernel, crop = truncate_psf(averaged, requested_energy)
    verified = all(selection.get(k) is True for k in ("target_is_isolated", "sensor_target_stationary", "domain_verified"))
    profile = reader.profile
    radar = profile.radar_config()
    overrides = selection.get("sensor_overrides", {})
    supported = {"sensor_height", "frequency_hz", "polarization", "phase_mode", "scattering_gain", "reference_range"}
    if not isinstance(overrides, dict) or set(overrides) - supported:
        raise ValueError(f"sensor_overrides supports only {sorted(supported)}; grid changes require a new measured profile")
    if overrides:
        from ..config import RadarConfig
        radar = RadarConfig(**{**asdict(radar), **overrides})
    if profile.representation == "complex":
        radar.representation = "complex"
    else:
        radar.representation = "intensity"
    grid = {k: getattr(radar, k) for k in ("range_step", "velocity_step", "angular_step", "azimuth_coordinate", "representation")}
    metadata = {**crop, "source": "native stationary-target measurement",
                "dataset": profile.name, "axis_order": ["range", "doppler", "azimuth"], "grid": grid,
                "frames_averaged": frame_count, "phase_aligned_to_center": phase_align,
                "unadjusted_center_phase_coherence": float(abs(np.mean(peaks)) / np.mean(np.abs(peaks))),
                "normalized_by_complex_center": normalize_peak,
                "center_gain_before_normalization": [float(np.real(center_gain)), float(np.imag(center_gain))],
                "baseline_subtracted": [float(np.real(baseline)), float(np.imag(baseline))],
                "selection_status": "user_confirmed_target_assumptions" if verified else "provisional_target_assumptions_unconfirmed",
                "complete_sensor_calibration": False, "noise": noise, "provenance": provenance,
                "noise_law": noise_model, "spatial_noise_correlation_modeled": False,
                "physical_response": "coherent_field" if radar.representation == "complex" else "incoherent_power_surrogate_no_phase",
                "energy_reference": "selected measured target support, not unobserved full sensor response",
                "native_profile": profile.to_dict()}
    metadata["sensor_geometry_and_scattering_overrides"] = overrides
    save_psf(out / "psf.npz", kernel, metadata)
    np.savez_compressed(out / "noise_bank.npz", samples=samples)
    radar.psf_path = "psf.npz"
    radar.noise_model = noise_model
    radar.noise_bank_path = "noise_bank.npz" if noise_model == "empirical" else None
    radar.noise_std = noise["std"]
    radar.noise_mean_real, radar.noise_mean_imag = noise["mean_real"], noise["mean_imag"]
    radar.noise_covariance = noise.get("quadrature_covariance")
    radar.doppler_wrap = True
    extent = float(radar.ranges[-1])
    scene = SceneConfig(min_range=max(3.6, extent * 0.15), max_range=max(8, extent - 4),
                        lateral_limit=max(2, min(22, extent * 0.45)))
    config = SimulationConfig(radar=radar, scene=scene)
    json_write(out / "sensor_config.json", config.to_dict())
    json_write(out / "calibration.json", metadata)
    json_write(out / "selection.json", selection)
    _plot_calibration(kernel, samples, radar, out / "calibration_diagnostics.png")
    return out / "sensor_config.json"


def _plot_calibration(kernel, noise_samples, radar, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    center = np.asarray(kernel.shape) // 2
    for axis, name in enumerate(("Range bins", "Doppler bins", "Angular-coordinate bins")):
        sel = list(center)
        sel[axis] = slice(None)
        values = np.abs(kernel[tuple(sel)])
        axes[0, axis].plot(np.arange(len(values)) - center[axis], 20 * np.log10(np.maximum(values / values.max(), 1e-12)))
        axes[0, axis].set(xlabel=name, ylabel="Response magnitude (relative dB)", ylim=(-80, 3))
        axes[0, axis].grid(alpha=.2)
    sample = np.asarray(noise_samples)
    axes[1, 0].hist(sample.real, bins=60, density=True)
    axes[1, 0].set_title("Measured noise ROI: real component")
    if np.iscomplexobj(sample):
        axes[1, 1].hist(sample.imag, bins=60, density=True)
        axes[1, 1].set_title("Measured noise ROI: imaginary component")
    else:
        axes[1, 1].hist(np.maximum(sample, 0), bins=60, density=True)
        axes[1, 1].set_title("Measured scalar tensor-domain samples")
    axes[1, 2].imshow(np.abs(kernel).max(axis=1), origin="lower", aspect="auto")
    axes[1, 2].set(title="Measured PSF: Doppler-max magnitude", xlabel="Azimuth offset", ylabel="Range offset")
    fig.suptitle("Native-data calibration diagnostics; target assumptions require verification")
    fig.savefig(path, dpi=150)
    plt.close(fig)
