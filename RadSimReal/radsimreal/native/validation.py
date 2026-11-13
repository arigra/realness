"""Unpaired supplied-native/synthetic diagnostics; no accuracy claim by histogram."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy import stats
from ..datasets.bev_dataset import BEVDataset
from .calibration import json_write


def _collect(dataset, max_frames, seed):
    if len(dataset) == 0 or max_frames < 1:
        raise ValueError("Validation needs nonempty input splits and max_frames >= 1")
    rng = np.random.default_rng(seed)
    selected = np.linspace(0, len(dataset) - 1, min(max_frames, len(dataset))).astype(int)
    samples, range_profiles, dimensions, ids = [], [], [], []
    for i in selected:
        raw = dataset.raw(int(i))
        image = raw["image_ra"]
        values = image.ravel()
        sampled = values[rng.choice(len(values), min(8192, len(values)), replace=False)]
        samples.append(10 * np.log10(np.maximum(sampled, 1e-12)))
        range_profiles.append(10 * np.log10(np.maximum(image.mean(axis=1), 1e-12)))
        boxes = raw["boxes_physical"]
        dimensions.extend((boxes[:, 2:] - boxes[:, :2]).tolist())
        ids.append(raw["id"])
    sample = np.concatenate(samples)
    summary = {"frames_used": len(selected), "ids": ids, "label_domain": dataset.label_domain,
               "RA_Doppler_max_dB_quantiles": {str(q): float(np.quantile(sample, q)) for q in (0, .01, .1, .5, .9, .99, .999, 1)},
               "car_count": len(dimensions), "box_dimensions_metres": dimensions,
               "mean_range_profile_dB": np.mean(range_profiles, axis=0).tolist(),
               "sampling": "deterministic evenly spaced frames; up to8192 RA cells/frame"}
    return sample, summary


def validate_distribution(native_root, synthetic_root, output_dir, *, split="test", max_frames=32):
    native = BEVDataset(native_root, split, include_radar=False)
    synthetic = BEVDataset(synthetic_root, split, normalization=native.normalization, include_radar=False, grid=native.grid)
    for field in ("ranges", "velocities", "angular_grid"):
        a, b = getattr(native.config.radar, field), getattr(synthetic.config.radar, field)
        if a.shape != b.shape or not np.allclose(a, b):
            raise ValueError(f"Native and synthetic sensor {field} differ; diagnostics require matching declared sampling")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    real_values, real_summary = _collect(native, max_frames, 2024)
    synthetic_values, synthetic_summary = _collect(synthetic, max_frames, 2025)
    ks = stats.ks_2samp(real_values, synthetic_values)
    report = {"source_access": "supplied local files", "dataset_origin_independently_certified": False,
              "comparisons_are_paired_scenes": False, "split": split,
              "datasets": {"native": real_summary, "synthetic": synthetic_summary},
              "dB_Wasserstein_distance": float(stats.wasserstein_distance(real_values, synthetic_values)),
              "dB_two_sample_KS_statistic": float(ks.statistic),
              "KS_pvalue_diagnostic_only_correlated_cells": float(ks.pvalue),
              "same_label_policy": native.label_domain == synthetic.label_domain,
              "sensor_grid_matches": True, "sensor_realism_established": False,
              "published_result_reproduced": False,
              "interpretation": "Unpaired intensity/annotation diagnostics. Matching these summaries does not establish correct scattering, PSF fidelity, generalization or paper AP."}
    json_write(out / "distribution_report.json", report)
    _plot_comparison(native, synthetic, real_values, synthetic_values, report, out / "comparison.png")
    return report


def _plot_comparison(native, synthetic, real_values, synthetic_values, report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    both = np.r_[real_values, synthetic_values]
    lo, hi = np.quantile(both, [.01, .999])
    if hi <= lo:
        hi = lo + 1
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, dataset, title in zip(axes[0], (native, synthetic), ("Supplied native frame", "Independent synthetic scene")):
        raw = dataset.raw(0)
        with torch.no_grad():
            bev = dataset.project(torch.as_tensor(raw["image_ra"], dtype=torch.float64)).numpy()
        grid = dataset.grid
        mesh = ax.imshow(10 * np.log10(np.maximum(bev, 1e-12)), origin="lower", cmap="magma",
            extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max], aspect="equal", vmin=lo, vmax=hi)
        for x1, y1, x2, y2 in raw["boxes_physical"]:
            ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "c-", lw=1)
        ax.set(title=title, xlabel="Lateral x (m)", ylabel="Forward y (m)")
    fig.colorbar(mesh, ax=list(axes[0]), label="Power response (shared relative dB scale)")
    bins = np.linspace(float(min(both.min(), lo)), float(max(both.max(), hi)), 80)
    axes[1, 0].hist(real_values, bins=bins, density=True, histtype="step", label="Native")
    axes[1, 0].hist(synthetic_values, bins=bins, density=True, histtype="step", label="Synthetic")
    axes[1, 0].set(xlabel="RA Doppler-max response (dB)", ylabel="Sample density", title="Unpaired cell-intensity distributions")
    axes[1, 0].legend()
    for name, dataset in (("native", native), ("synthetic", synthetic)):
        axes[1, 1].plot(dataset.config.radar.ranges, report["datasets"][name]["mean_range_profile_dB"], label=name)
    axes[1, 1].set(xlabel="Range (m)", ylabel="Mean angular power (dB)", title="Average range profiles")
    axes[1, 1].legend()
    fig.suptitle("Unpaired supplied-native/synthetic diagnostics · does not establish paper reproduction")
    fig.savefig(output, dpi=145)
    plt.close(fig)
