"""Phase 1 probes (src/utility_metric.py, scripts/build_metric_features.py)."""
import numpy as np
import pytest
import torch

from src.utility_metric import Radar, negatives, processing, raw, rates, row_prominence, summarise

RADAR = Radar(0.2, 0.4, 16, True)
R, D = 64, 32


def item(blobs, labels, seed=0, frames=2, noise=1.0):
    """A sequence of `frames` identical-geometry frames: noise (dB) plus Gaussian blobs
    [(r, d, amplitude dB, sigma)], labels [(r, d)] in every frame."""
    g = torch.Generator().manual_seed(seed)
    x = noise * torch.randn(frames, R, D, generator=g)
    rr = torch.arange(R, dtype=torch.float)[:, None]
    dd = torch.arange(D, dtype=torch.float)[None]
    for r, d, a, s in blobs:
        ddw = torch.remainder(dd - d + D / 2, D) - D / 2
        x = x + a * torch.exp(-((rr - r) ** 2 + ddw ** 2) / (2 * s ** 2))
    traj = torch.zeros(8, frames, 2)
    present = torch.zeros(8, frames, dtype=torch.bool)
    for m, (r, d) in enumerate(labels):
        traj[m, :, 0], traj[m, :, 1], present[m] = r, d, True
    return {"x": x.half(), "traj": traj, "present": present, "stats": {"mean": 0.0, "std": 1.0}}


def objects(pool):
    return raw(pool, RADAR)["objects"]


def test_label_shift_moves_offsets_by_the_shift():
    base = objects([item([(30, 10, 12, 1.2)], [(30, 10)])])
    moved = objects([item([(30, 10, 12, 1.2)], [(31, 10)])])
    assert np.allclose(base["peak_dr"], 0) and np.allclose(moved["peak_dr"], -1)
    assert np.allclose(moved["cen_dr"] - base["cen_dr"], -1, atol=1e-4)
    side = objects([item([(30, 10, 12, 1.2)], [(30, 12)])])
    assert np.allclose(side["peak_dd"], -2)


def test_doppler_wraps_in_offsets_and_patches():
    # object straddling the Doppler seam: peak at bin 0, label at bin 31.6
    o = objects([item([(30, 0, 12, 1.2)], [(30, 31.6)])])
    assert np.allclose(o["peak_dd"], 0.4, atol=1e-4)
    assert np.all(np.abs(o["cen_dd"]) < 1.0)
    # the patch wraps: a wide blob on the seam is as wide as the same blob mid-map
    seam = objects([item([(30, 0, 12, 2.5)], [(30, 0)], noise=0.01)])
    mid = objects([item([(30, 12, 12, 2.5)], [(30, 12)], noise=0.01)])
    assert np.all(seam["width_d2"] >= 3) and np.array_equal(seam["width_d4"], mid["width_d4"])


def test_weaker_target_lower_contrast():
    strong = objects([item([(30, 10, 12, 1.2)], [(30, 10)])])
    weak = objects([item([(30, 10, 6, 1.2)], [(30, 10)])])
    for k in ("label_prom", "over_floor", "over_clutter"):
        assert np.all(weak[k] < strong[k]), k


def test_unlabelled_object_raises_unlabelled_peaks():
    one = raw([item([(30, 10, 12, 1.2)], [(30, 10)])], RADAR)
    two = raw([item([(30, 10, 12, 1.2), (45, 22, 12, 1.2)], [(30, 10)])], RADAR)
    thr = 6.0
    assert rates(two, thr)["unlabelled_per_frame"] > rates(one, thr)["unlabelled_per_frame"]
    assert rates(one, thr)["labels_with_peak_r2"] == 1.0


def test_wider_target_wider_response():
    narrow = objects([item([(30, 10, 12, 1.0)], [(30, 10)])])
    wide = objects([item([(30, 10, 12, 2.5)], [(30, 10)])])
    for k in ("width_r2", "width_d2", "area2"):
        assert np.all(wide[k] > narrow[k]), k


def test_identical_inputs_zero_distance():
    pool = [item([(30, 10, 12, 1.2)], [(30, 10)], seed=s, frames=8) for s in range(4)]
    t = raw(pool, RADAR)
    v = summarise(t, t)
    for comp in ("coverage", "label", "target", "processing", "negatives"):
        assert v["scores"][comp] == pytest.approx(0.0, abs=1e-9), comp
    assert v["global"]["sum"] == pytest.approx(0.0, abs=1e-9)


def test_deterministic():
    pool = [item([(30, 10, 12, 1.2)], [(30, 10)], seed=s) for s in range(2)]
    a, b = raw(pool, RADAR), raw(pool, RADAR)
    for t in a:
        for k in a[t]:
            assert np.array_equal(a[t][k], b[t][k], equal_nan=True), (t, k)


def test_builder_reads_training_splits_only(monkeypatch):
    import scripts.build_metric_features as bfm
    from src import versions
    seen = []

    def fake(cache, split, limit=None):
        seen.append(split)
        return [item([(30, 10, 12, 1.2)], [(30, 10)])]
    monkeypatch.setattr(bfm, "_LOAD", fake)
    monkeypatch.setattr(versions, "load", bfm.train_only)
    with pytest.raises(RuntimeError):
        bfm.train_only("data/x", "val")
    with pytest.raises(RuntimeError):
        bfm.train_only("data/x", "test")
    row = {"dataset": "carrada", "suite": "study2", "version": "rangeshift1", "cache": "(on the tuned cache)"}
    bfm.load_candidate(row)
    assert seen and set(seen) == {"train"}


def test_ridge_width_and_excess():
    x = torch.zeros(R, D)
    x[:, RADAR.zero - 1:RADAR.zero + 2] = 5.0          # a 3-bin zero-Doppler ridge
    p = processing(x, RADAR)
    assert p["ridge_width"] == 3 and p["ridge_excess"] == pytest.approx(5.0)
    assert p["ridge_b0"] == pytest.approx(5.0)


def test_false_positives_need_score_and_distance():
    it = item([(30, 10, 12, 1.2), (45, 22, 12, 1.2)], [(30, 10)])
    z = it["x"][0].float().numpy()
    prom = row_prominence(it["x"][0].float()).numpy()
    pk = np.array([[45, 22, prom[45, 22]], [30, 10, prom[30, 10]]], dtype=float)
    nearest = np.array([np.hypot(15, 12), 0.0])
    dets = [(0.9, 30.0, 10.0), (0.5, 45.0, 22.0), (0.2, 50.0, 5.0)]   # on a label / FP / too weak
    rows, n_fp = negatives(z, prom, pk, nearest, [(30, 10)], dets, RADAR)
    assert n_fp == 1
    assert [r["kind"] for r in rows] == [0, 1]       # the unlabelled peak, then the false positive
    assert rows[1]["r"] == 45 and rows[1]["d"] == 22
