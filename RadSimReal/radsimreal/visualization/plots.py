"""Headless, axis-aware physical and detector visualizations."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _edges(centers):
    centers = np.asarray(centers)
    mid = (centers[:-1] + centers[1:]) / 2
    return np.r_[centers[0] - (mid[0] - centers[0]), mid, centers[-1] + (centers[-1] - mid[-1])]


def _db(image):
    return 10 * np.log10(np.maximum(image, 1e-12))


def _boxes(ax, boxes, radar, color, scores=None, label=None):
    if boxes is None:
        return
    boxes = np.asarray(boxes).reshape(-1, 4)
    if not len(boxes):
        return
    # Normalized pixel edges -> fractional bin centers -> physical coordinates.
    ai = np.arange(radar.azimuth_bins)
    ri = np.arange(radar.range_bins)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x = np.interp(np.array([x1, x2]) * radar.azimuth_bins - 0.5, ai, np.rad2deg(radar.azimuths_rad))
        y = np.interp(np.array([y1, y2]) * radar.range_bins - 0.5, ri, radar.ranges)
        ax.plot([x[0], x[1], x[1], x[0], x[0]], [y[0], y[0], y[1], y[1], y[0]],
                color=color, lw=1.6, label=label if i == 0 else None)
        if scores is not None:
            ax.text(x[0], y[1], f"{scores[i]:.2f}", color=color, fontsize=7, va="bottom")


def _radar_axis(ax, image, radar, boxes=None, predictions=None):
    mesh = ax.pcolormesh(_edges(np.rad2deg(radar.azimuths_rad)), _edges(radar.ranges),
                         _db(image), shading="flat", cmap="magma", vmin=-30, vmax=30)
    _boxes(ax, boxes, radar, "cyan", label="Ground truth")
    if predictions is not None:
        _boxes(ax, predictions["boxes"], radar, "lime", predictions.get("scores"), "Predictions")
    ax.set(xlabel="Azimuth (degrees)", ylabel="Range (m)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=7)
    return mesh


def plot_radar(image, radar, output_path, boxes=None, predictions=None, title="Synthetic radar image"):
    fig, ax = plt.subplots(figsize=(8.5, 5.6), constrained_layout=True)
    mesh = _radar_axis(ax, image, radar, boxes, predictions)
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="Intensity (dB, relative units)")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_demo(scene, reflections, result, radar, output_path, boxes=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax = axes[0, 0]
    for obj in scene.objects:
        corners = np.asarray(obj.corners())[:, :2]
        closed = np.concatenate([corners, corners[:1]])
        ax.plot(closed[:, 0], closed[:, 1], "c-", lw=1.5)
        ax.text(obj.x, obj.y, f"Car {obj.object_id}", fontsize=7)
    xyz = reflections.xyz
    coeff = np.abs(reflections.coefficients)
    if len(xyz):
        points = ax.scatter(xyz[:, 0], xyz[:, 1], c=_db(np.maximum(coeff, 1e-12)), s=9, cmap="viridis", zorder=3)
        fig.colorbar(points, ax=ax, label="Reflection coefficient (dB magnitude)")
    ax.scatter([0], [0], marker="^", s=70, c="red", label="Radar")
    ax.set(xlabel="Lateral x (m)", ylabel="Forward y (m)", title="Approximate surface scenes and reflection points")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    psf = np.abs(result.psf)
    if radar.representation == "complex":
        psf = psf ** 2
    projection = psf.max(axis=1)
    ax.imshow(_db(projection), origin="lower", aspect="auto", cmap="magma", vmin=-60, vmax=0,
              extent=[-projection.shape[1] // 2 + 0.5, projection.shape[1] // 2 + 0.5,
                      -projection.shape[0] // 2 + 0.5, projection.shape[0] // 2 + 0.5])
    ax.set(xlabel="Azimuth bin offset", ylabel="Range bin offset", title="Truncated PSF (Doppler maximum)")
    ax = axes[1, 0]
    mesh = _radar_axis(ax, result.image, radar, boxes)
    ax.set_title("Synthetic range–azimuth radar image")
    fig.colorbar(mesh, ax=ax, label="Intensity (dB, relative units)")
    ax = axes[1, 1]
    cube = np.abs(result.tensor) ** 2 if radar.representation == "complex" else np.maximum(result.tensor, 0)
    mesh = ax.pcolormesh(_edges(radar.velocities), _edges(radar.ranges), _db(cube.max(axis=2)),
                         shading="flat", cmap="magma", vmin=-30, vmax=30)
    ax.set(xlabel="Radial velocity (m/s; positive receding)", ylabel="Range (m)", title="Range–Doppler (maximum across azimuth)")
    fig.colorbar(mesh, ax=ax, label="Intensity (dB, relative units)")
    fig.suptitle("RadSimReal method reconstruction · uncalibrated illustrative scene", fontsize=14)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_comparison(synthetic, real, radar, output_path):
    if synthetic.shape != real.shape or synthetic.shape != (radar.range_bins, radar.azimuth_bins):
        raise ValueError("Both images must match the same calibrated grid")
    if not np.isfinite(real).all() or np.any(real < 0) or np.iscomplexobj(real):
        raise ValueError("Real image must be finite nonnegative linear intensity/power")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, image, name in zip(axes, [synthetic, real], ["Synthetic", "User-supplied real"]):
        mesh = _radar_axis(ax, image, radar)
        ax.set_title(name)
    fig.colorbar(mesh, ax=axes, label="Intensity (dB, shared relative units)")
    fig.suptitle("Qualitative comparison only; scene pairing and gain calibration are user supplied")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return Path(output_path)

