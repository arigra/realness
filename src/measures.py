"""Realness measurements: how a pool of maps compares with real maps, without
training anything (docs/notes/2026-11-01-realness-score.md).

All on the maps the detector sees (each source normalised by its own stats).
Three groups:
- texture, per frame: spread of the map at four scales (differences of box
  means, 1/2/4/8 cells) and the correlation of the finest residual between
  neighbouring cells in range and in Doppler;
- contrast, per labelled object: its peak (max within +/-2 cells) over the
  frame's background floor (median) and over its strong clutter (99th
  percentile), background = the frame outside +/-3 cells of every label;
- label agreement, per labelled object: distance from the label to the
  brightest cell within +/-3 cells, and prominence = peak within +/-1 cell over
  the median of its ring (+/-6 cells outside +/-3).
Doppler is circular; range is not.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

from src.data import normalised

GROUPS = {"texture": ("spread1", "spread2", "spread4", "spread8", "corr_range", "corr_doppler"),
          "contrast": ("over_floor", "over_clutter"),
          "label": ("displacement", "prominence")}
MEASURES = [m for ms in GROUPS.values() for m in ms]


def _box(z, k):
    """Mean over a (2k+1) x (2k+1) window; z (R, D)."""
    if k == 0:
        return z
    p = F.pad(z[None, None], (k, k, 0, 0), mode="circular")
    p = F.pad(p, (0, 0, k, k), mode="replicate")
    return F.avg_pool2d(p, 2 * k + 1, stride=1)[0, 0]


def _window(z, r, d, k):
    rows = slice(max(r - k, 0), r + k + 1)
    cols = [(d + j) % z.shape[1] for j in range(-k, k + 1)]
    return z[rows][:, cols], rows, cols


def texture(z):
    out = {}
    for s0, s1, name in ((0, 1, "spread1"), (1, 2, "spread2"), (2, 4, "spread4"), (4, 8, "spread8")):
        out[name] = float((_box(z, s0) - _box(z, s1)).std())
    res = z - _box(z, 1)
    var = float((res ** 2).mean())
    out["corr_range"] = float((res[:-1] * res[1:]).mean()) / var
    out["corr_doppler"] = float((res * torch.roll(res, 1, 1)).mean()) / var
    return out


def objects(z, labels):
    """labels: [(range, Doppler)] bins of the frame's labelled objects."""
    if not labels:
        return []
    n_r, n_d = z.shape
    near = torch.zeros_like(z, dtype=torch.bool)
    cells = [(min(max(int(round(r)), 0), n_r - 1), int(round(d)) % n_d) for r, d in labels]
    for r, d in cells:
        _, rows, cols = _window(z, r, d, 3)
        near[rows, cols] = True
    bg = z[~near]
    floor = float(bg.median())
    clutter = float(torch.kthvalue(bg, max(int(0.99 * bg.numel()), 1)).values)
    out = []
    for (r, d), (rf, df) in zip(cells, labels):
        peak = float(_window(z, r, d, 2)[0].max())
        w3, rows, cols = _window(z, r, d, 3)
        i = int(w3.argmax())
        pr, pd = rows.start + i // w3.shape[1], cols[i % w3.shape[1]]
        dd = (pd - df + n_d / 2) % n_d - n_d / 2
        w6, rows6, cols6 = _window(z, r, d, 6)
        inner = torch.zeros_like(w6, dtype=torch.bool)
        inner[max(r - 3, 0) - rows6.start: r + 4 - rows6.start, 3:10] = True
        out.append({"over_floor": peak - floor, "over_clutter": peak - clutter,
                    "displacement": float(np.hypot(pr - rf, dd)),
                    "prominence": float(_window(z, r, d, 1)[0].max()) - float(w6[~inner].median())})
    return out


def measure(pool):
    """{measure: np.array} over every frame (texture) and labelled object (rest) of a pool."""
    vals = {m: [] for m in MEASURES}
    for item in pool:
        for t in range(item["x"].shape[0]):
            z = normalised(item, t)
            for k, v in texture(z).items():
                vals[k].append(v)
            labels = item["traj"][item["present"][:, t], t].tolist()
            for o in objects(z, labels):
                for k, v in o.items():
                    vals[k].append(v)
    return {k: np.asarray(v) for k, v in vals.items()}


def distances(sim, real):
    """Per measure: Wasserstein-1 between the pools, in units of the real spread."""
    return {m: wasserstein_distance(sim[m], real[m]) / (real[m].std() + 1e-12) for m in MEASURES}


def group_distances(d):
    return {g: float(np.mean([d[m] for m in ms])) for g, ms in GROUPS.items()}


# ---- profiles used to tune the simulators (src/tuned.py), in dB on the raw maps

def range_cut(pool, kr=12, kd=12):
    """Mean normalised level along range through each labelled object (+/-kr rows),
    on the label's Doppler column +/-0; objects closer than kr to an edge are skipped."""
    cuts = []
    for item in pool:
        for t in range(item["x"].shape[0]):
            z = normalised(item, t)
            for r, d in item["traj"][item["present"][:, t], t].tolist():
                ri, di = int(round(r)), int(round(d)) % z.shape[1]
                if kr <= ri < z.shape[0] - kr:
                    cuts.append(z[ri - kr: ri + kr + 1, di])
    return torch.stack(cuts).mean(0)


def clutter_rows(pool, zero_doppler=None, smooth=9):
    """Per range row, how far the static scene stands over the row's median (dB, median
    over frames): the zero-Doppler column when the radar is static (zero_doppler given),
    else the row's 99th percentile. Smoothed over `smooth` rows."""
    x = torch.stack([it["x"].float() for it in pool]).flatten(0, 1)
    med = x.median(dim=2).values
    top = x[:, :, zero_doppler] if zero_doppler is not None else torch.quantile(x, 0.99, dim=2)
    prof = (top - med).median(dim=0).values
    k = smooth // 2
    return F.avg_pool1d(F.pad(prof[None, None], (k, k), mode="replicate"), smooth, 1)[0, 0]
