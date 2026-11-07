"""Metric v2 features and selectors (src/utility_metric_v2.py, src/utility_score_v2.py)."""
import numpy as np
import pytest
import torch

from src.utility_metric import Radar
from src.utility_metric_v2 import (background_mask, capture, cfar_margin, coverage, coverage_summary, m4_probes,
                                   raw_v2, strongest_within)
from src.utility_score_v2 import PairwiseRanker, RobustTolerance, m3, m4

RADAR = Radar(0.2, 0.4, 16, True)
R, D = 64, 32


def frame(blobs, noise=1.0, seed=0, extra=None):
    """dB frame: N(0, noise) + Gaussian blobs [(r, d, amp, sr, sd)] (+ an optional function adding more)."""
    g = torch.Generator().manual_seed(seed)
    x = noise * torch.randn(R, D, generator=g, dtype=torch.float64)
    rr = torch.arange(R, dtype=torch.float64)[:, None]
    dd = torch.arange(D, dtype=torch.float64)[None]
    for r, d, a, sr, sd in blobs:
        ddw = torch.remainder(dd - d + D / 2, D) - D / 2
        x = x + a * torch.exp(-((rr - r) ** 2) / (2 * sr ** 2) - ddw ** 2 / (2 * sd ** 2))
    if extra is not None:
        x = extra(x)
    return x.numpy() + 40.0


def item(blobs, labels, frames=2, **kw):
    x = torch.stack([torch.from_numpy(frame(blobs, seed=s, **kw)) for s in range(frames)]).half()
    traj = torch.zeros(8, frames, 2)
    present = torch.zeros(8, frames, dtype=torch.bool)
    for m, (r, d) in enumerate(labels):
        traj[m, :, 0], traj[m, :, 1], present[m] = r, d, True
    return {"x": x, "traj": traj, "present": present, "stats": {"mean": 40.0, "std": 1.0}}


def objects(pool, dets=None):
    return raw_v2(pool, RADAR, dets)["objects"]


def test_circular_doppler_strongest_cell_and_capture():
    x = frame([(30, 0, 12, 1.2, 1.2)])
    assert strongest_within(x, 30, 31.6) == (30, 0)
    lin = np.clip(10 ** (x / 10) - 10 ** (np.median(x, 1, keepdims=True) / 10), 0, None)
    assert capture(lin, 30, 31.6, 2.0) > 0.3          # the energy on the other side of the seam counts


def test_background_excludes_guard_and_labels():
    rows, cols, keep = background_mask(30, 10, [(30, 10), (30, 15)], (R, D), RADAR)
    pos = {(r, c): k for r, cs in zip(rows, np.broadcast_to(cols, (len(rows), len(cols)))) for c, k in zip(cs, keep[list(rows).index(r)])}
    assert not pos[(30, 10)] and not pos[(32, 11)] and not pos[(30, 15)] and not pos[(31, 16)]
    assert not pos[(40, 16)] and not pos[(40, 17)]       # the zero-Doppler ridge (bin 16 +/- 1)
    assert pos[(40, 10)]


def test_stronger_target_raises_cfar_margin():
    weak, strong = frame([(30, 8, 6, 1.2, 1.2)]), frame([(30, 8, 12, 1.2, 1.2)])
    assert cfar_margin(strong, 30, 8, [(30, 8)], RADAR) > cfar_margin(weak, 30, 8, [(30, 8)], RADAR)


def test_more_noise_lowers_cfar_margin():
    quiet, noisy = frame([(30, 8, 12, 1.2, 1.2)], noise=1.0), frame([(30, 8, 12, 1.2, 1.2)], noise=3.0)
    assert cfar_margin(noisy, 30, 8, [(30, 8)], RADAR) < cfar_margin(quiet, 30, 8, [(30, 8)], RADAR)


def test_clutter_lowers_cfar_margin_with_the_same_floor():
    clutter = [(22, 4, 9, 0.8, 0.8), (38, 12, 9, 0.8, 0.8), (25, 13, 9, 0.8, 0.8), (36, 3, 9, 0.8, 0.8)]
    clean = frame([(30, 8, 12, 1.2, 1.2)])
    busy = frame([(30, 8, 12, 1.2, 1.2)] + clutter)
    assert abs(np.median(busy) - np.median(clean)) < 0.2          # floor (map median) unchanged
    assert cfar_margin(busy, 30, 8, [(30, 8)], RADAR) < cfar_margin(clean, 30, 8, [(30, 8)], RADAR)


def test_range_shift_moves_offset_and_lowers_capture():
    base = objects([item([(30, 8, 12, 1.2, 1.2)], [(30, 8)])])
    moved = objects([item([(30, 8, 12, 1.2, 1.2)], [(31.5, 8)])])
    assert np.allclose(base["off_r"], 0) and np.allclose(moved["off_r"], -1.5)
    assert np.all(moved["capture_r2"] < base["capture_r2"])


def test_doppler_offset_wraps():
    o = objects([item([(30, 0, 12, 1.2, 1.2)], [(30, 31.6)])])
    assert np.allclose(o["off_d"], 0.4, atol=1e-4)


def test_unrendered_target_lowers_visibility():
    seen = objects([item([(30, 8, 12, 1.2, 1.2)], [(30, 8)])])
    gone = objects([item([], [(30, 8)])])
    assert np.all(gone["peak_prom"] < seen["peak_prom"]) and np.all(gone["cfar_db"] < seen["cfar_db"])


def test_unlabelled_target_appears_among_negatives():
    t = raw_v2([item([(30, 8, 12, 1.2, 1.2), (45, 25, 12, 1.2, 1.2)], [(30, 8)])], RADAR, None)
    n = t["negatives"]
    top = n["kind"] == 0
    assert np.any(np.isclose(n["range_m"][top], 45 * 0.2))


def test_range_and_doppler_blur_widen_physical_widths():
    narrow = objects([item([(30, 8, 12, 1.0, 1.0)], [(30, 8)], noise=0.01)])
    wide_r = objects([item([(30, 8, 12, 2.5, 1.0)], [(30, 8)], noise=0.01)])
    wide_d = objects([item([(30, 8, 12, 1.0, 2.5)], [(30, 8)], noise=0.01)])
    assert np.all(wide_r["width_r6_m"] > narrow["width_r6_m"]) and np.allclose(wide_r["width_d6_mps"], narrow["width_d6_mps"])
    assert np.all(wide_d["width_d6_mps"] > narrow["width_d6_mps"]) and np.allclose(wide_d["width_r6_m"], narrow["width_r6_m"])


def test_range_tail_raises_far_extent():
    def tail(x):
        x = x.clone()
        for k in range(1, 9):
            x[30 + k, 8] += 12 * np.exp(-k / 4)
        return x
    plain = objects([item([(30, 8, 12, 1.0, 1.0)], [(30, 8)], noise=0.01)])
    tailed = objects([item([(30, 8, 12, 1.0, 1.0)], [(30, 8)], noise=0.01, extra=tail)])
    assert np.all(tailed["far6_m"] > plain["far6_m"]) and np.all(tailed["asym6_m"] > plain["asym6_m"])


def test_identical_candidate_and_reference_give_zero_risk():
    pool = [item([(30, 8, 12, 1.2, 1.2), (45, 25, 9, 1.0, 1.0)], [(30, 8)], frames=8, seed=s) if False else
            item([(30 + s, 8, 12, 1.2, 1.2), (45, 25, 9, 1.0, 1.0)], [(30 + s, 8)], frames=8) for s in range(6)]
    t = raw_v2(pool, RADAR, None)
    P, _ = m4_probes(t, t, {}, float(np.median(t["objects"]["range_m"])))
    for k in ("A.cfar_db_median", "A.visibility", "B.capture_r2", "C.width_r6_m", "D.under_worst", "E.uncovered"):
        assert P[k] == pytest.approx(0.0, abs=1e-9), k
    names = list(P)
    X = np.array([[P[n] for n in names]])
    tol = RobustTolerance().fit(np.vstack([X, X + 0.1]), np.array([True, True]), np.full(len(names), 0.05))
    pen = tol.penalties(X)[0]
    assert np.nansum(pen) == 0 and m4(pen, names)["M4"] == 0 and m3(pen, ["x." + n for n in names])["max"] == 0


def test_real_to_real_coverage_median_ratio_near_one():
    g = np.random.default_rng(0)
    real = g.normal(size=(400, 4)); seq = np.repeat(np.arange(20), 20)
    cand = g.normal(size=(400, 4))
    tc, tr = coverage(real, seq, real, seq, cand)
    s = coverage_summary(tc, tr)
    assert abs(s["median_log_ratio"]) < 0.3 and s["uncovered"] < 0.15


def test_robust_tolerance_scale_and_clip():
    X = np.array([[0.0], [0.1], [0.2], [5.0]])
    tol = RobustTolerance().fit(X, np.array([True, True, True, False]), np.array([0.0]))
    assert tol.scale[0] == pytest.approx(0.1)
    assert tol.penalties(np.array([[100.0]]))[0, 0] == 5.0
    tol0 = RobustTolerance().fit(np.zeros((3, 1)), np.ones(3, bool), np.array([0.0]))
    assert np.isnan(tol0.penalties(np.array([[1.0]]))[0, 0])     # non-informative probe excluded


def test_pairwise_ranker_learns_the_harmful_component():
    g = np.random.default_rng(0)
    R = g.uniform(size=(40, 8))
    val = 0.5 - 0.3 * R[:, 1] + 0.01 * g.normal(size=40)
    rk = PairwiseRanker().fit(R, val, np.array(["a"] * 40), np.array([f"f{i % 4}" for i in range(40)]), {"a": 0.02})
    assert np.all(rk.w >= 0) and np.argmax(rk.w) == 1


def test_deterministic():
    pool = [item([(30, 8, 12, 1.2, 1.2)], [(30, 8)])]
    a, b = raw_v2(pool, RADAR, None), raw_v2(pool, RADAR, None)
    for t in a:
        for k in a[t]:
            assert np.array_equal(a[t][k], b[t][k], equal_nan=True), (t, k)


def test_builder_reads_training_splits_only():
    import scripts.build_metric_v2_features as b
    with pytest.raises(RuntimeError):
        b.train_only("data/x", "val")
    with pytest.raises(RuntimeError):
        b.train_only("data/x", "test")
