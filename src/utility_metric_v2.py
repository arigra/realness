"""Metric v2 raw features and probes (docs/notes/2026-11-06-metric-v2-preregistration.md, M4 / M5).

raw_v2(pool, radar, dets) -> tables (dict of equal-length numpy arrays, tagged with the sequence index):
- objects: one row per labelled object per frame:
    A detectability: strongest cell within Euclidean radius 2 of the label; local background = the 25 x 17
      patch around it minus radius 3 around it, radius 2 around every label, and the zero-Doppler ridge +/-1
      (static radars); local CFAR margin = peak - q95(background), in raw dB and in the detector's z units;
      the strongest cell's prominence over its range row's median (z) for visibility;
    B label task: offsets label -> strongest cell (radius 2); AP@2 energy capture (linear excess over the
      range-row median, clipped at 0, within Euclidean radius 2 / 4 of the label over the excess in the
      25 x 17 patch around the label; NaN when the patch has none);
    C morphology (raw dB relative to the strongest cell; m and m/s): range / Doppler width at -3 / -6 dB,
      near / far range extent at -6 dB, asymmetry, peak count (local maxima >= peak - 6 dB), range / Doppler
      spectral entropy and autocorrelation length of the dB patch;
    M5: the fixed negative detector's best score within radius 2 (0: none, a false negative);
- negatives: the fixed detector's false positives (score >= 0.3, no label within 4; kind 1) and the 3 most
  prominent unlabelled local maxima per frame (kind 0), with range, speed, prominence, ridge flag, CFAR margin
  and -6 dB shape;
- frames: one row per frame (labels, false positives).
Doppler is circular everywhere; range is not.
"""
import dataclasses

import numpy as np
import torch

from src.data import normalised
from src.utility_metric import (PEAK_FLOOR, RADARS, _acorr_len, _run, _spectrum, _window, local_maxima,
                                row_prominence, wrap)

K_R, K_D = 12, 8            # patch half sizes (25 x 17)
GUARD, LABEL_GUARD = 3, 2   # background exclusions (Euclidean cells)
FP_SCORE, FP_RADIUS = 0.3, 4
N_NEG_PEAKS = 3
VERSION = 2


def config():
    return {"version": VERSION, "k_r": K_R, "k_d": K_D, "guard": GUARD, "label_guard": LABEL_GUARD,
            "fp_score": FP_SCORE, "fp_radius": FP_RADIUS, "n_neg_peaks": N_NEG_PEAKS, "peak_floor": PEAK_FLOOR,
            "radars": {k: dataclasses.asdict(v) for k, v in RADARS.items()}}


def _patch_idx(r, d, n_r, n_d):
    rows = np.arange(max(r - K_R, 0), min(r + K_R + 1, n_r))
    cols = (d + np.arange(-K_D, K_D + 1)) % n_d
    return rows, cols


def strongest_within(x, rf, df, radius=2.0):
    """Strongest cell of x within Euclidean `radius` of (rf, df) (Doppler wrapping)."""
    n_r, n_d = x.shape
    best, cell = -np.inf, None
    k = int(np.ceil(radius))
    for dr in range(-k, k + 1):
        r = int(round(rf)) + dr
        if r < 0 or r >= n_r:
            continue
        for dd in range(-k, k + 1):
            d = int(round(df)) + dd
            if np.hypot(r - rf, wrap(d - df, n_d)) <= radius + 1e-9:
                v = x[r, d % n_d]
                if v > best:
                    best, cell = v, (r, d % n_d)
    if cell is None:                                          # label off the map: clamp
        r = min(max(int(round(rf)), 0), n_r - 1)
        cell = (r, int(round(df)) % n_d)
    return cell


def background_mask(pr, pd, labels, shape, radar):
    """Bool mask over the patch around (pr, pd): True = background."""
    n_r, n_d = shape
    rows, cols = _patch_idx(pr, pd, n_r, n_d)
    R, C = np.meshgrid(rows, cols, indexing="ij")
    keep = np.hypot(R - pr, wrap(C - pd, n_d)) > GUARD
    for lr, ld in labels:
        keep &= np.hypot(R - lr, wrap(C - ld, n_d)) > LABEL_GUARD
    if radar.static:
        keep &= np.abs(wrap(C - radar.zero, n_d)) > 1
    return rows, cols, keep


def cfar_margin(x, pr, pd, labels, radar):
    rows, cols, keep = background_mask(pr, pd, labels, x.shape, radar)
    bg = x[np.ix_(rows, cols)][keep]
    return float(x[pr, pd] - np.quantile(bg, 0.95)) if bg.size >= 10 else np.nan


def capture(lin_excess, rf, df, radius):
    """Share of the patch's excess energy within Euclidean `radius` of the label (NaN if none)."""
    n_r, n_d = lin_excess.shape
    ri, di = min(max(int(round(rf)), 0), n_r - 1), int(round(df)) % n_d
    rows, cols = _patch_idx(ri, di, n_r, n_d)
    e = lin_excess[np.ix_(rows, cols)]
    tot = float(e.sum())
    if tot <= 0:
        return np.nan
    R = rows[:, None] - rf
    D = wrap(di + np.arange(-K_D, K_D + 1)[None, :] - df, n_d)
    return float(e[np.hypot(R, D) <= radius].sum() / tot)


def morphology(x, pr, pd, radar):
    """Shape around the strongest cell in raw dB relative to it, physical units. NaN near range edges."""
    names = ["width_r3_m", "width_r6_m", "width_d3_mps", "width_d6_mps", "near6_m", "far6_m", "asym6_m",
             "npk6", "entropy_r", "entropy_d", "acorr_r", "acorr_d"]
    n_r, n_d = x.shape
    if pr < K_R or pr >= n_r - K_R:
        return dict.fromkeys(names, np.nan)
    rows, cols = _patch_idx(pr, pd, n_r, n_d)
    p = x[np.ix_(rows, cols)]
    peak = p[K_R, K_D]
    out = {}
    for t in (3, 6):
        a, b = _run(p[:, K_D], K_R, peak - t)
        out[f"width_r{t}_m"] = (a + b + 1) * radar.range_res
        a, b = _run(p[K_R], K_D, peak - t)
        out[f"width_d{t}_mps"] = (a + b + 1) * radar.vel_res
    near, far = _run(p[:, K_D], K_R, peak - 6)          # range index grows away from the radar
    out["near6_m"], out["far6_m"] = near * radar.range_res, far * radar.range_res
    out["asym6_m"] = (far - near) * radar.range_res
    lm = local_maxima(torch.from_numpy(p)).numpy() & (p >= peak - 6)
    out["npk6"] = float(lm.sum())
    out["entropy_r"] = _spectrum(p, 0)[0]
    out["entropy_d"] = _spectrum(p, 1)[0]
    out["acorr_r"], out["acorr_d"] = _acorr_len(p, 0), _acorr_len(p, 1)
    return {k: float(v) for k, v in out.items()}


OBJECT_KEYS = ["seq", "frame", "range_m", "speed", "speed_abs", "near_zero", "peak_db", "peak_z", "peak_prom",
               "cfar_db", "cfar_z", "off_r", "off_d", "capture_r2", "capture_r4", "det_score_r2",
               "width_r3_m", "width_r6_m", "width_d3_mps", "width_d6_mps", "near6_m", "far6_m", "asym6_m", "npk6",
               "entropy_r", "entropy_d", "acorr_r", "acorr_d"]
NEG_KEYS = ["seq", "frame", "kind", "score", "range_m", "speed", "speed_abs", "near_zero", "prom", "on_ridge",
            "cfar_db", "width_r6_m", "width_d6_mps", "near6_m", "far6_m"]
FRAME_KEYS = ["seq", "frame", "n_labels", "n_fp"]


def raw_v2(pool, radar, dets):
    """dets: the fixed negative detector's detections, one list of (score, r, d) per frame in pool order."""
    objs, negs, frames = [], [], []
    f_i = 0
    for s, item in enumerate(pool):
        for t in range(item["x"].shape[0]):
            zt = normalised(item, t)
            z, x = zt.numpy().astype(np.float64), item["x"][t].float().numpy().astype(np.float64)
            n_r, n_d = x.shape
            prom = row_prominence(zt).numpy()
            rowmed_db = np.median(x, axis=1, keepdims=True)
            lin_excess = np.clip(10 ** (x / 10) - 10 ** (rowmed_db / 10), 0, None)
            labels = item["traj"][item["present"][:, t], t].tolist()
            fd = dets[f_i] if dets is not None else []
            f_i += 1
            fdet = np.asarray([(r, d, sc) for sc, r, d in fd], dtype=np.float64).reshape(-1, 3)
            for rf, df in labels:
                pr, pd = strongest_within(x, rf, df, 2.0)
                v = wrap(df - radar.zero, n_d) * radar.vel_res
                if len(fdet):
                    dist = np.hypot(fdet[:, 0] - rf, wrap(fdet[:, 1] - df, n_d))
                    best = float(fdet[dist <= 2.0, 2].max()) if (dist <= 2.0).any() else 0.0
                else:
                    best = 0.0
                row = {"seq": s, "frame": t, "range_m": rf * radar.range_res, "speed": v, "speed_abs": abs(v),
                       "near_zero": float(abs(wrap(df - radar.zero, n_d)) <= 1), "peak_db": x[pr, pd],
                       "peak_z": z[pr, pd], "peak_prom": prom[pr, pd],
                       "cfar_db": cfar_margin(x, pr, pd, labels, radar), "cfar_z": cfar_margin(z, pr, pd, labels, radar),
                       "off_r": pr - rf, "off_d": wrap(pd - df, n_d),
                       "capture_r2": capture(lin_excess, rf, df, 2.0), "capture_r4": capture(lin_excess, rf, df, 4.0),
                       "det_score_r2": best}
                objs.append(row | morphology(x, pr, pd, radar))
            # negatives
            lab = np.asarray(labels, dtype=np.float64).reshape(-1, 2)

            def neg(kind, score, r, d):
                v = wrap(d - radar.zero, n_d) * radar.vel_res
                m = morphology(x, r, d, radar)
                return {"seq": s, "frame": t, "kind": kind, "score": score, "range_m": r * radar.range_res,
                        "speed": v, "speed_abs": abs(v), "near_zero": float(abs(wrap(d - radar.zero, n_d)) <= 1),
                        "prom": prom[r, d], "on_ridge": float(radar.static and abs(wrap(d - radar.zero, n_d)) <= 1),
                        "cfar_db": cfar_margin(x, r, d, labels, radar), "width_r6_m": m["width_r6_m"],
                        "width_d6_mps": m["width_d6_mps"], "near6_m": m["near6_m"], "far6_m": m["far6_m"]}
            lm = local_maxima(zt).numpy() & (prom >= PEAK_FLOOR)
            rr, dd = np.nonzero(lm)
            if len(rr):
                far = np.ones(len(rr), bool)
                if len(lab):
                    far = np.hypot(rr[:, None] - lab[None, :, 0], wrap(dd[:, None] - lab[None, :, 1], n_d)).min(1) > FP_RADIUS
                idx = np.flatnonzero(far)
                for i in idx[np.argsort(-prom[rr[idx], dd[idx]], kind="stable")[:N_NEG_PEAKS]]:
                    negs.append(neg(0, np.nan, int(rr[i]), int(dd[i])))
            n_fp = 0
            for sc, r, d in fd:
                if sc < FP_SCORE:
                    continue
                if len(lab) and np.hypot(lab[:, 0] - r, wrap(lab[:, 1] - d, n_d)).min() <= FP_RADIUS:
                    continue
                n_fp += 1
                # measured at the strongest cell within +/-1 of the detection (as v1)
                w, o_r, o_d = _window(x, int(r), int(d) % n_d, 1, 1)
                i = np.unravel_index(int(np.argmax(w)), w.shape)
                negs.append(neg(1, float(sc), int(r) + int(o_r[i[0]]), (int(d) + int(o_d[i[1]])) % n_d))
            frames.append({"seq": s, "frame": t, "n_labels": len(labels), "n_fp": float(n_fp)})

    def table(rows, keys):
        return {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in keys}
    return {"objects": table(objs, OBJECT_KEYS), "negatives": table(negs, NEG_KEYS), "frames": table(frames, FRAME_KEYS)}


# ------------------------------------------------------------------ probes (candidate vs reference)

def _c(a):
    a = np.asarray(a, dtype=np.float64)
    return a[np.isfinite(a)]


def shift(c, r, stat=np.median, min_n=5):
    """(stat(candidate) - stat(reference)) / sd(reference)."""
    c, r = _c(c), _c(r)
    if len(c) < min_n or len(r) < min_n:
        return np.nan
    sd = r.std() if r.std() > 0 else np.concatenate([c, r]).std()
    return 0.0 if sd == 0 else float((stat(c) - stat(r)) / sd)


q10 = lambda a: np.quantile(a, 0.10)
q25 = lambda a: np.quantile(a, 0.25)


def strata_of(range_m, near_zero, median_range):
    """0 near-moving, 1 near-zero, 2 far-moving, 3 far-zero."""
    return (np.asarray(range_m) > median_range).astype(int) * 2 + np.asarray(near_zero).astype(int)


def _nn(a, b, chunk=256):
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        out[i:i + chunk] = np.sqrt(((a[i:i + chunk, None] - b[None]) ** 2).sum(-1).min(1))
    return out


def coverage(real_q, real_q_seq, real_pool, real_pool_seq, cand, min_n=5, max_cand=6000, seed=0):
    """Per real query example: distance to the nearest candidate example and to the nearest real pool example
    of another sequence. Inputs already standardised. Returns (to_cand, to_real)."""
    g = np.random.default_rng(seed)
    if len(cand) > max_cand:
        cand = cand[g.choice(len(cand), max_cand, replace=False)]
    to_cand = _nn(real_q, cand) if len(cand) else np.full(len(real_q), np.inf)
    to_real = np.full(len(real_q), np.nan)
    for s in np.unique(real_q_seq):
        m, other = real_q_seq == s, real_pool_seq != s
        if other.any():
            to_real[m] = _nn(real_q[m], real_pool[other])
    return to_cand, to_real


def coverage_summary(to_cand, to_real, strata=None, min_n=5):
    ok = np.isfinite(to_real) & np.isfinite(to_cand)
    if ok.sum() < 2:
        return {"n": int(ok.sum())}
    tc, tr = to_cand[ok], to_real[ok]
    lr = np.log((tc + 1e-6) / (tr + 1e-6))
    thr = np.quantile(tr, 0.95)
    unc = tc > thr
    out = {"n": int(ok.sum()), "median_log_ratio": float(np.median(lr)), "q90_log_ratio": float(np.quantile(lr, 0.9)),
           "uncovered": float(unc.mean())}
    if strata is not None:
        st = np.asarray(strata)[ok]
        per = {int(b): float(unc[st == b].mean()) for b in np.unique(st) if (st == b).sum() >= min_n}
        out["strata_uncovered"] = per
        out["worst_stratum_uncovered"] = max(per.values()) if per else np.nan
    return out


POS_FEATURES = ("range_m", "speed", "cfar_db", "capture_r2", "off_r", "off_d", "width_r6_m", "width_d6_mps",
                "near6_m", "far6_m", "peak_prom", "near_zero")
NEG_FEATURES = ("range_m", "speed", "cfar_db", "width_r6_m", "width_d6_mps", "near6_m", "far6_m", "prom", "on_ridge")


def _feat(t, keys, m=None):
    """Feature matrix of a table (NaN kept: morphology is NaN where the patch is cut at a range edge)."""
    X = np.column_stack([t[k] for k in keys]) if len(t["seq"]) else np.zeros((0, len(keys)))
    return X if m is None else X[m]


def _std(X, ref):
    """Standardise by ref's mean / sd (NaN-aware), clip to [-5, 5]; an unmeasured value is put at the
    reference mean (0), so a cut patch is neither near nor far by itself."""
    if len(ref) and np.isfinite(ref).any():
        with np.errstate(all="ignore"):
            mu, sd = np.nanmean(ref, 0), np.nanstd(ref, 0)
    else:
        mu, sd = np.zeros(X.shape[1]), np.ones(X.shape[1])
    mu, sd = np.where(np.isfinite(mu), mu, 0.0), np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return np.nan_to_num(np.clip((X - mu) / sd, -5, 5), nan=0.0)


def m5(cand, ref, median_range, low_n=20):
    """Real detector-error coverage (M5): false negatives vs candidate positives, false positives vs candidate
    negatives; uncovered shares and the worst range / velocity stratum."""
    ro, rn, co, cn = ref["objects"], ref["negatives"], cand["objects"], cand["negatives"]
    fn = ro["det_score_r2"] == 0
    fp = rn["kind"] == 1
    out = {"n_fn": int(fn.sum()), "n_fp": int(fp.sum()), "n_low_tp": int(((ro["det_score_r2"] > 0) &
                                                                           (ro["det_score_r2"] < FP_SCORE)).sum())}
    unc_all, strata_all = [], []
    for side, q_m, pool, cpool, keys in (("fn", fn, ro, co, POS_FEATURES), ("fp", fp, rn, cn, NEG_FEATURES)):
        if q_m.sum() < 2:
            out[side] = {"n": int(q_m.sum())}
            out[f"{side}_risk"] = np.nan
            continue
        base = _feat(pool, keys, q_m)
        Q = _std(base, base)
        P = _std(_feat(pool, keys), base)
        C = _std(_feat(cpool, keys), base)
        tc, tr = coverage(Q, pool["seq"][q_m], P, pool["seq"], C)
        st = strata_of(pool["range_m"][q_m], pool["near_zero"][q_m], median_range)
        sm = coverage_summary(tc, tr, st)
        sm["low_confidence"] = bool(q_m.sum() < low_n)
        out[side] = sm
        out[f"{side}_risk"] = sm.get("uncovered", np.nan)
        ok = np.isfinite(tr) & np.isfinite(tc)
        if ok.sum() >= 2:
            unc_all.append(tc[ok] > np.quantile(tr[ok], 0.95))
            strata_all.append(st[ok])
    if unc_all:
        u, st = np.concatenate(unc_all), np.concatenate(strata_all)
        per = {int(b): float(u[st == b].mean()) for b in np.unique(st) if (st == b).sum() >= 5}
        out["strata_uncovered"] = per
        out["stratum_risk"] = max(per.values()) if per else np.nan
    else:
        out["stratum_risk"] = np.nan
    vals = [out.get("fn_risk"), out.get("fp_risk"), out.get("stratum_risk")]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    out["M5"] = max(vals) if vals else np.nan
    vals2 = [out.get(f"{s}_risk") for s in ("fn", "fp") if out.get(s, {}).get("low_confidence") is False]
    vals2 = [v for v in vals2 if v is not None and np.isfinite(v)] + ([out["stratum_risk"]] if np.isfinite(out["stratum_risk"]) else [])
    out["M5_no_lowconf"] = max(vals2) if vals2 else np.nan
    return out


def m4_probes(cand, ref, v1_vector, ref_median_range, low_n=20):
    """M4 probe values (signed, candidate vs reference) per block, plus reference counts and flags.
    v1_vector: the v1 adequacy vector of the same (candidate, reference) for v1-measured probes."""
    co, ro = cand["objects"], ref["objects"]
    thr = float(np.quantile(_c(ro["peak_prom"]), 0.25)) if len(_c(ro["peak_prom"])) else np.nan
    vis = lambda o, m=None: float(np.mean((o["peak_prom"] if m is None else o["peak_prom"][m]) >= thr)) \
        if (len(o["peak_prom"]) if m is None else m.sum()) else np.nan
    P = {}
    for u in ("db", "z"):
        P[f"A.cfar_{u}_median"] = shift(co[f"cfar_{u}"], ro[f"cfar_{u}"])
        P[f"A.cfar_{u}_q10"] = shift(co[f"cfar_{u}"], ro[f"cfar_{u}"], q10)
        P[f"A.cfar_{u}_q25"] = shift(co[f"cfar_{u}"], ro[f"cfar_{u}"], q25)
    P["A.visibility"] = vis(co) - vis(ro)
    lab = v1_vector.get("label", {})
    for k in ("peak_dr", "peak_dd", "cen_dr", "cen_dd"):
        P[f"B.{k}"] = lab.get(k, {}).get("shift", np.nan) if lab.get(k) else np.nan
    P["B.capture_r2"] = shift(co["capture_r2"], ro["capture_r2"])
    P["B.capture_r4"] = shift(co["capture_r4"], ro["capture_r4"])
    rates = v1_vector.get("label_rates", {}).get("25", {})
    for r in (2, 4):
        c_, r_ = rates.get("cand", {}).get(f"labels_with_peak_r{r}"), rates.get("ref", {}).get(f"labels_with_peak_r{r}")
        P[f"B.peak_within_{r}"] = (c_ - r_) if c_ is not None and r_ is not None else np.nan
    for k in ("unlabelled_per_frame", "unlabelled_off_ridge_per_frame"):
        P[f"B.{k}"] = lab.get(k, {}).get("shift", np.nan) if lab.get(k) else np.nan
    for k in ("width_r3_m", "width_r6_m", "width_d3_mps", "width_d6_mps", "near6_m", "far6_m", "asym6_m", "npk6",
              "entropy_r", "entropy_d", "acorr_r", "acorr_d"):
        P[f"C.{k}"] = shift(co[k], ro[k])
    # D support
    cs, rs = strata_of(co["range_m"], co["near_zero"], ref_median_range), strata_of(ro["range_m"], ro["near_zero"], ref_median_range)
    counts = {int(b): int((rs == b).sum()) for b in range(4)}
    under = []
    for b in range(4):
        pr, pc = (rs == b).mean() if len(rs) else np.nan, (cs == b).mean() if len(cs) else np.nan
        u = max(0.0, pr - pc) if np.isfinite(pr) and np.isfinite(pc) else np.nan
        P[f"D.under_s{b}"] = u
        under.append(u)
        if counts[b] >= 5:
            P[f"D.vis_s{b}"] = vis(co, cs == b) - vis(ro, rs == b) if (cs == b).sum() else np.nan
            P[f"D.cfar_q10_s{b}"] = shift(co["cfar_db"][cs == b], ro["cfar_db"][rs == b], q10)
        else:
            P[f"D.vis_s{b}"] = P[f"D.cfar_q10_s{b}"] = np.nan
    P["D.under_mean"] = float(np.nanmean(under)) if np.isfinite(under).any() else np.nan
    P["D.under_worst"] = float(np.nanmax(under)) if np.isfinite(under).any() else np.nan
    je = v1_vector.get("coverage", {}).get("joint_energy", {})
    P["D.joint_energy"] = je.get("w1", np.nan) if je else np.nan
    # E hard negatives: real FPs + top unlabelled peaks of the reference vs the candidate's negatives
    rn, cn = ref["negatives"], cand["negatives"]
    if len(rn["seq"]) >= 5 and len(cn["seq"]):
        base = _feat(rn, NEG_FEATURES)
        Q, C = _std(base, base), _std(_feat(cn, NEG_FEATURES), base)
        tc, tr = coverage(Q, rn["seq"], Q, rn["seq"], C)
        far = rn["range_m"] > np.median(rn["range_m"])
        strong = rn["prom"] > np.median(rn["prom"])
        sm = coverage_summary(tc, tr)
        ok = np.isfinite(tr) & np.isfinite(tc)
        thr95 = np.quantile(tr[ok], 0.95) if ok.sum() >= 2 else np.nan
        sub = {}
        for name, m in (("ridge", rn["on_ridge"] == 1), ("far", far), ("strong", strong)):
            mm = m & ok
            if mm.sum() >= 5:
                sub[name] = float((tc[mm] > thr95).mean())
        P["E.median_log_ratio"] = sm.get("median_log_ratio", np.nan)
        P["E.q90_log_ratio"] = sm.get("q90_log_ratio", np.nan)
        P["E.uncovered"] = sm.get("uncovered", np.nan)
        P["E.worst_stratum_uncovered"] = max(sub.values()) if sub else np.nan
    else:
        for k in ("median_log_ratio", "q90_log_ratio", "uncovered", "worst_stratum_uncovered"):
            P[f"E.{k}"] = np.nan
    P["E.fp_per_frame"] = shift(cand["frames"]["n_fp"], ref["frames"]["n_fp"], np.mean)
    # F processing: v1 probes
    proc = v1_vector.get("processing", {})
    for k in ("spread1", "spread4", "corr_range", "corr_doppler", "frame_q99", "ridge_width", "ridge_excess",
              "clip_min", "clip_max"):
        P[f"F.{k}"] = proc.get(k, {}).get("shift", np.nan) if proc.get(k) else np.nan
    for k in ("rowmed_profile", "rowq99_profile"):
        P[f"F.{k}"] = proc.get(k, {}).get("w1", np.nan) if proc.get(k) else np.nan
    P["F.ridge_profile"] = proc.get("ridge_profile", {}).get("shift", np.nan) if proc.get("ridge_profile") else np.nan
    P = {k: (np.nan if v is None else float(v)) for k, v in P.items()}
    meta = {"n_ref_labels": int(len(ro["seq"])), "stratum_counts": counts,
            "low_confidence": bool(len(ro["seq"]) < 100),
            "low_confidence_strata": [b for b, n in counts.items() if n < low_n],
            "visibility_threshold": thr, "ref_median_range": float(ref_median_range)}
    return P, meta


BLOCKS = {"A": "detectability", "B": "label-task", "C": "morphology", "D": "support", "E": "hard negatives",
          "F": "processing"}
