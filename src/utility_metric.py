"""Detection adequacy vector: raw probes of a pool of range-Doppler sequences
(docs/notes/2026-11-06-metric-experiments.md, Phase 1).

Everything is measured on the maps the detector sees (each source normalised by its own
statistics, src/data.normalised), so "z" units below are standard deviations of the source's
training maps. Doppler is circular everywhere; range is not. Real and synthetic pools go
through exactly the same code.

raw(pool) returns three tables (dict of equal-length numpy arrays), each row tagged with its
sequence index so references can be resampled by sequence:
- objects: one row per labelled object per frame: coverage (range m, radial speed m/s,
  objects in the frame, class when known), label agreement (offsets to the strongest cell and to
  the power centroid, contrast, best peak prominence within 2/4/8 cells) and target response
  (shape of the response around the object's strongest cell);
- peaks: every local maximum (3x3, Doppler wrapping) with prominence over its range row's
  median >= PEAK_FLOOR, with its distance to the nearest label;
- frames: one row per frame (labels in it).
Thresholds that depend on the real reference (what counts as a target-like peak) are applied
later, in summarise(), so the raw tables never change with the reference.
"""
import dataclasses

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import wasserstein_distance

from src.data import normalised
from src.measures import objects as contrast_and_displacement

PEAK_FLOOR = 1.5            # z; real CARRADA labelled objects: 10th percentile of prominence 2.9
K_OFFSET = 4                # +/- cells searched for the strongest cell / power centroid of a label
RADII = (2, 4, 8)           # "a target-like peak within r cells of the label"
K_R, K_D = 12, 8            # target patch half-sizes (cells), as the classification crops
DROPS = (1, 2, 4)           # z below the peak: width / area thresholds
UNLABELLED_RADIUS = 4       # a peak farther than this from every label is unlabelled
QUANTILES = (10, 25, 50)    # target-like = prominence >= this percentile of the reference's labels
PRIMARY_Q = 25


@dataclasses.dataclass(frozen=True)
class Radar:
    range_res: float        # m per range bin
    vel_res: float          # m/s per Doppler bin
    zero: int               # zero-Doppler bin
    static: bool            # static radar: the zero-Doppler ridge is the static scene


RADARS = {"carrada": Radar(0.2, 0.42, 32, True), "raddet": Radar(0.1953125, 0.4197, 32, True)}


def config():
    """Everything that defines the raw features (hashed into every output)."""
    return {"version": 2, "peak_floor": PEAK_FLOOR, "k_offset": K_OFFSET, "radii": RADII, "k_r": K_R,
            "k_d": K_D, "drops": DROPS, "unlabelled_radius": UNLABELLED_RADIUS, "quantiles": QUANTILES,
            "primary_q": PRIMARY_Q, "radars": {k: dataclasses.asdict(v) for k, v in RADARS.items()}}


def wrap(dd, n):
    """Signed circular difference in (-n/2, n/2]."""
    return (dd + n / 2) % n - n / 2


# ------------------------------------------------------------------ peaks

def local_maxima(z):
    """Boolean (R, D): cells not below any 8-neighbour; Doppler wraps, range edges see -inf."""
    p = torch.cat([z[:, -1:], z, z[:, :1]], dim=1)
    p = F.pad(p[None, None], (0, 0, 1, 1), value=float("-inf"))
    return z >= F.max_pool2d(p, 3, stride=1)[0, 0]


def row_prominence(z):
    """z over its range row's median: the level of a cell above the background at its range."""
    return z - z.median(dim=1, keepdim=True).values


# ------------------------------------------------------------------ per object

def _window(a, r, d, kr, kd):
    """a[rows, cols] around cell (r, d), rows clipped, cols wrapped; with the row/col offsets."""
    rows = np.arange(max(r - kr, 0), min(r + kr + 1, a.shape[0]))
    cols = (d + np.arange(-kd, kd + 1)) % a.shape[1]
    return a[np.ix_(rows, cols)], rows - r, np.arange(-kd, kd + 1)


def _run(line, c, level):
    """Contiguous cells around index c with line >= level: (cells before c, cells after c)."""
    lo = c
    while lo - 1 >= 0 and line[lo - 1] >= level:
        lo -= 1
    hi = c
    while hi + 1 < len(line) and line[hi + 1] >= level:
        hi += 1
    return c - lo, hi - c


def _spectrum(patch, axis):
    """Normalised spectral entropy and low/mid/high energy shares along one axis (DC dropped)."""
    p = patch - patch.mean(axis=axis, keepdims=True)
    s = (np.abs(np.fft.rfft(p, axis=axis)) ** 2).mean(axis=1 - axis)[1:]
    if s.sum() <= 0:
        return np.nan, np.nan, np.nan, np.nan
    q = s / s.sum()
    ent = float(-(q * np.log(q + 1e-12)).sum() / np.log(len(q)))
    b = np.array_split(q, 3)
    return ent, float(b[0].sum()), float(b[1].sum()), float(b[2].sum())


def _acorr_len(patch, axis):
    """First lag where the mean normalised autocorrelation along an axis falls below 0.5."""
    p = patch - patch.mean(axis=axis, keepdims=True)
    n = p.shape[axis]
    var = (p ** 2).mean()
    if var <= 0:
        return np.nan
    for lag in range(1, n // 2):
        a = np.take(p, range(n - lag), axis=axis)
        b = np.take(p, range(lag, n), axis=axis)
        if (a * b).mean() / var < 0.5:
            return float(lag)
    return float(n // 2)


def target_response(z, pr, pd):
    """Shape of the response around the object's strongest cell (pr, pd): z units below the peak.
    Range rows nearer than K_R to an edge give NaN (the patch would be cut)."""
    out = {}
    names = ([f"width_r{t}" for t in DROPS] + [f"width_d{t}" for t in DROPS] + [f"area{t}" for t in DROPS]
             + ["near_ext2", "far_ext2", "tail_asym2", "tail_level", "n_peaks2",
                "entropy_r", "low_r", "mid_r", "high_r", "entropy_d", "low_d", "mid_d", "high_d",
                "acorr_r", "acorr_d"])
    if pr < K_R or pr >= z.shape[0] - K_R:
        return {n: np.nan for n in names}
    patch, _, _ = _window(z, pr, pd, K_R, K_D)
    c_r, c_d = K_R, K_D
    peak = patch[c_r, c_d]
    for t in DROPS:
        a, b = _run(patch[:, c_d], c_r, peak - t)
        out[f"width_r{t}"] = float(a + b + 1)
        a, b = _run(patch[c_r], c_d, peak - t)
        out[f"width_d{t}"] = float(a + b + 1)
        lab, _ = ndimage.label(patch >= peak - t)
        out[f"area{t}"] = float((lab == lab[c_r, c_d]).sum())
    near, far = _run(patch[:, c_d], c_r, peak - 2)          # range index grows away from the radar
    out["near_ext2"], out["far_ext2"], out["tail_asym2"] = float(near), float(far), float(far - near)
    out["tail_level"] = float(patch[c_r + 1:, c_d].mean() - patch[:c_r, c_d].mean())
    lm = local_maxima(torch.from_numpy(patch)).numpy() & (patch >= peak - 2)
    out["n_peaks2"] = float(lm.sum())
    for axis, s in ((0, "r"), (1, "d")):
        out[f"entropy_{s}"], out[f"low_{s}"], out[f"mid_{s}"], out[f"high_{s}"] = _spectrum(patch, axis)
        out[f"acorr_{s}"] = _acorr_len(patch, axis)
    return out


def label_agreement(z, x_db, prom, peaks_rd, rf, df):
    """One label at fractional (rf, df). peaks_rd: (n, 3) local maxima of the frame (r, d, prominence)."""
    n_r, n_d = z.shape
    ri, di = min(max(int(round(rf)), 0), n_r - 1), int(round(df)) % n_d
    w, o_r, o_d = _window(z, ri, di, K_OFFSET, K_OFFSET)
    i = np.unravel_index(int(np.argmax(w)), w.shape)
    out = {"peak_dr": float(ri + o_r[i[0]] - rf), "peak_dd": float(wrap(di + o_d[i[1]] - df, n_d))}
    out["peak_dist"] = float(np.hypot(out["peak_dr"], out["peak_dd"]))
    pr, pd = ri + o_r[i[0]], (di + o_d[i[1]]) % n_d
    # power centroid of the return, in a window centred on its strongest cell (centring on the
    # label would pull the centroid towards the label)
    lin, c_r, c_d = _window(x_db, pr, pd, K_OFFSET, K_OFFSET)
    lin = 10 ** (lin / 10)
    wgt = np.clip(lin - np.median(lin), 0, None)
    if wgt.sum() > 0:
        out["cen_dr"] = float((wgt.sum(1) @ (pr + c_r)) / wgt.sum() - rf)
        out["cen_dd"] = float(wrap((wgt.sum(0) @ (pd + c_d)) / wgt.sum() - df, n_d))
    else:
        out["cen_dr"] = out["cen_dd"] = np.nan
    pw, _, _ = _window(prom, ri, di, 2, 2)
    out["label_prom"] = float(pw.max())
    if len(peaks_rd):
        dist = np.hypot(peaks_rd[:, 0] - rf, wrap(peaks_rd[:, 1] - df, n_d))
        for r in RADII:
            near = dist <= r
            out[f"best_prom_r{r}"] = float(peaks_rd[near, 2].max()) if near.any() else -np.inf
    else:
        for r in RADII:
            out[f"best_prom_r{r}"] = -np.inf
    return out, int(pr), int(pd)


# ------------------------------------------------------------------ a pool

def raw(pool, radar, classes=None):
    """The three raw tables of a pool (list of cache items). classes: (n_seq, M) class per slot or None."""
    objs, peaks, frames = [], [], []
    for s, item in enumerate(pool):
        for t in range(item["x"].shape[0]):
            zt = normalised(item, t)
            z, x_db = zt.numpy(), item["x"][t].float().numpy()
            prom_t = row_prominence(zt)
            prom = prom_t.numpy()
            lm = local_maxima(zt).numpy() & (prom >= PEAK_FLOOR)
            pr_, pd_ = np.nonzero(lm)
            pk = np.column_stack([pr_, pd_, prom[pr_, pd_]]).astype(np.float64)
            slots = torch.nonzero(item["present"][:, t]).flatten().tolist()
            labels = item["traj"][slots, t].tolist()
            n_d = z.shape[1]
            if labels and len(pk):
                lab = np.asarray(labels)
                dist = np.hypot(pk[:, None, 0] - lab[None, :, 0], wrap(pk[:, None, 1] - lab[None, :, 1], n_d))
                nearest = dist.min(1)
            else:
                nearest = np.full(len(pk), np.inf)
            on_ridge = np.abs(wrap(pk[:, 1] - radar.zero, n_d)) <= 1
            for (r, d, p), nd, rid in zip(pk, nearest, on_ridge):
                peaks.append({"seq": s, "frame": t, "r": r, "d": d, "prom": p, "nearest_label": nd,
                              "on_ridge": float(rid)})
            frames.append({"seq": s, "frame": t, "n_labels": len(labels)})
            contrast = contrast_and_displacement(zt, labels)
            for k, (slot, (rf, df)) in enumerate(zip(slots, labels)):
                la, pr, pd = label_agreement(z, x_db, prom, pk, rf, df)
                v = wrap(df - radar.zero, n_d) * radar.vel_res
                row = {"seq": s, "frame": t, "range_m": rf * radar.range_res, "speed": v, "speed_abs": abs(v),
                       "near_zero": float(abs(wrap(df - radar.zero, n_d)) <= 1), "n_in_frame": len(labels),
                       "cls": float(classes[s, slot]) if classes is not None else np.nan,
                       "over_floor": contrast[k]["over_floor"], "over_clutter": contrast[k]["over_clutter"],
                       "ring_prominence": contrast[k]["prominence"]}
                row |= la | target_response(z, pr, pd)
                objs.append(row)
    table = lambda rows: {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in rows[0]} if rows else {}
    return {"objects": table(objs), "peaks": table(peaks), "frames": table(frames)}


def select(tables, seqs):
    """The rows of the given sequences (a reference subset)."""
    seqs = np.asarray(sorted(seqs))
    return {name: {k: v[np.isin(t["seq"], seqs)] for k, v in t.items()} for name, t in tables.items()}


# ------------------------------------------------------------------ summaries and distances

COVERAGE = ("range_m", "speed_abs", "speed", "n_in_frame")
LABEL = ("peak_dr", "peak_dd", "peak_dist", "cen_dr", "cen_dd", "label_prom", "over_floor",
         "over_clutter", "ring_prominence")
TARGET = tuple(f"width_r{t}" for t in DROPS) + tuple(f"width_d{t}" for t in DROPS) + \
    tuple(f"area{t}" for t in DROPS) + ("near_ext2", "far_ext2", "tail_asym2", "tail_level", "n_peaks2",
                                        "entropy_r", "low_r", "mid_r", "high_r", "entropy_d", "low_d",
                                        "mid_d", "high_d", "acorr_r", "acorr_d")


def _clean(a):
    a = np.asarray(a, dtype=np.float64)
    return a[np.isfinite(a)]


def feature_distance(c, r, min_n=5):
    """Wasserstein-1 and median shift of a candidate's values vs the reference's, in units of the
    reference's standard deviation (pooled when the reference is constant)."""
    c, r = _clean(c), _clean(r)
    if len(c) < min_n or len(r) < min_n:
        return {"w1": np.nan, "shift": np.nan, "n_cand": len(c), "n_ref": len(r)}
    sd = r.std() if r.std() > 0 else np.concatenate([c, r]).std()
    if sd == 0:
        return {"w1": 0.0, "shift": 0.0, "n_cand": len(c), "n_ref": len(r)}
    return {"w1": float(wasserstein_distance(c, r) / sd), "shift": float((np.median(c) - np.median(r)) / sd),
            "cand_median": float(np.median(c)), "ref_median": float(np.median(r)),
            "n_cand": len(c), "n_ref": len(r)}


def energy_distance(a, b, max_n=2000, seed=0):
    """Multivariate energy distance between samples a (n, k) and b (m, k)."""
    g = np.random.default_rng(seed)
    a = a[g.choice(len(a), max_n, replace=False)] if len(a) > max_n else a
    b = b[g.choice(len(b), max_n, replace=False)] if len(b) > max_n else b
    d = lambda x, y: np.sqrt(((x[:, None] - y[None]) ** 2).sum(-1)).mean()
    return float(2 * d(a, b) - d(a, a) - d(b, b))


def _js(p, q):
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda x, y: float((x[x > 0] * np.log(x[x > 0] / y[x > 0])).sum())
    return (kl(p, m) + kl(q, m)) / 2


def thresholds(ref):
    """Target-like prominence per quantile: the q-th percentile of the reference's labelled objects."""
    p = _clean(ref["objects"]["label_prom"])
    return {q: float(np.percentile(p, q)) for q in QUANTILES}


def rates(tables, thr):
    """Threshold-dependent label agreement: labels with a target-like peak within r, target-like
    peaks with no label within UNLABELLED_RADIUS (per frame, and as a share), off the ridge too."""
    o, p, f = tables["objects"], tables["peaks"], tables["frames"]
    out = {f"labels_with_peak_r{r}": float(np.mean(o[f"best_prom_r{r}"] >= thr)) for r in RADII}
    strong = p["prom"] >= thr
    unl = strong & (p["nearest_label"] > UNLABELLED_RADIUS)
    n_frames = len(f["seq"])
    out["unlabelled_per_frame"] = float(unl.sum() / n_frames)
    out["unlabelled_share"] = float(unl.sum() / max(strong.sum(), 1))
    off = p["on_ridge"] == 0
    out["unlabelled_off_ridge_per_frame"] = float((unl & off).sum() / n_frames)
    return out


def per_frame_unlabelled(tables, thr, off_ridge=False):
    """Number of unlabelled target-like peaks in each frame (for a distributional distance);
    off_ridge: only peaks off the zero-Doppler ridge (on a static radar the ridge's static scene
    holds most strong unlabelled peaks and hides objects that are missing labels)."""
    p, f = tables["peaks"], tables["frames"]
    unl = (p["prom"] >= thr) & (p["nearest_label"] > UNLABELLED_RADIUS)
    if off_ridge:
        unl &= p["on_ridge"] == 0
    key = lambda s, t: (int(s), int(t))
    counts = {key(s, t): 0 for s, t in zip(f["seq"], f["frame"])}
    for s, t in zip(p["seq"][unl], p["frame"][unl]):
        counts[key(s, t)] += 1
    return np.asarray(list(counts.values()), dtype=np.float64)


def summarise(cand, ref):
    """The adequacy vector of a candidate against a reference (both raw tables). Components hold
    per-feature distances; a component's score is the mean W1 over its features (unweighted,
    Phase 1 diagnostic only)."""
    co, ro = cand["objects"], ref["objects"]
    out = {"coverage": {k: feature_distance(co[k], ro[k]) for k in COVERAGE}}
    cj, rj = np.column_stack([co["range_m"], co["speed_abs"]]), np.column_stack([ro["range_m"], ro["speed_abs"]])
    sd = rj.std(0)
    sd[sd == 0] = 1
    out["coverage"]["joint_energy"] = {"w1": energy_distance(cj / sd, rj / sd)}
    out["coverage"]["near_zero_share"] = {"cand": float(co["near_zero"].mean()), "ref": float(ro["near_zero"].mean())}
    out["coverage"]["near_zero_share"]["w1"] = abs(out["coverage"]["near_zero_share"]["cand"] - out["coverage"]["near_zero_share"]["ref"])
    if np.isfinite(co["cls"]).all() and np.isfinite(ro["cls"]).all():
        k = int(max(co["cls"].max(), ro["cls"].max())) + 1
        out["coverage"]["class_js"] = {"w1": _js(np.bincount(co["cls"].astype(int), minlength=k) + 1e-9,
                                                 np.bincount(ro["cls"].astype(int), minlength=k) + 1e-9)}
    out["label"] = {k: feature_distance(co[k], ro[k]) for k in LABEL}
    thr = thresholds(ref)
    out["thresholds"] = thr
    out["label_rates"] = {q: {"cand": rates(cand, t), "ref": rates(ref, t)} for q, t in thr.items()}
    t = thr[PRIMARY_Q]
    out["label"]["unlabelled_per_frame"] = feature_distance(per_frame_unlabelled(cand, t), per_frame_unlabelled(ref, t))
    out["label"]["unlabelled_off_ridge_per_frame"] = feature_distance(per_frame_unlabelled(cand, t, True),
                                                                      per_frame_unlabelled(ref, t, True))
    for r in RADII:
        c, rr = out["label_rates"][PRIMARY_Q]["cand"][f"labels_with_peak_r{r}"], out["label_rates"][PRIMARY_Q]["ref"][f"labels_with_peak_r{r}"]
        out["label"][f"labels_with_peak_r{r}"] = {"cand": c, "ref": rr, "w1": abs(c - rr)}
    out["target"] = {k: feature_distance(co[k], ro[k]) for k in TARGET}
    out["target_strata"] = strata(cand, ref)
    out["scores"] = {comp: float(np.nanmean([v["w1"] for v in out[comp].values()]))
                     for comp in ("coverage", "label", "target")}
    return out


def strata(cand, ref, min_n=10):
    """Target-response distances within range terciles (edges from the reference) x near-zero Doppler;
    a stratum needs min_n objects on both sides."""
    co, ro = cand["objects"], ref["objects"]
    edges = np.percentile(_clean(ro["range_m"]), (100 / 3, 200 / 3))
    bin_ = lambda o: np.digitize(o["range_m"], edges) * 2 + o["near_zero"].astype(int)
    cb, rb = bin_(co), bin_(ro)
    out = {}
    for b in range(6):
        cm, rm = cb == b, rb == b
        name = f"range{b // 2}_{'zero' if b % 2 else 'moving'}"
        if cm.sum() < min_n or rm.sum() < min_n:
            out[name] = {"n_cand": int(cm.sum()), "n_ref": int(rm.sum())}
            continue
        d = {k: feature_distance(co[k][cm], ro[k][rm]) for k in TARGET}
        out[name] = {"n_cand": int(cm.sum()), "n_ref": int(rm.sum()),
                     "score": float(np.nanmean([v["w1"] for v in d.values()])), "features": d}
    return out
