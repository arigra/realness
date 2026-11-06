"""Measure what the tuned simulators need from the 50 real training sequences
(src/tuned.py) and write results/tuning/<dataset>.json:

1. scenes: the labelled objects of every real training frame;
2. range gain: real row profile minus the simulator's (with the other fixes on);
3. contrast: CARRADA and RADDet: the mean and sd of the objects' SNR (dB) and the
   static offset; RADIal: target and static offsets (dB) and the SNR under which
   an object is not annotated (none on the grid won); chosen so that the
   quartiles of target over floor and over strong clutter (src/measures.py)
   match the real ones (least summed |quartile gap|, keeping at least 90% of
   the scene objects' labels): a coarse grid, then +/-3 dB around its best;
4. RADIal: the body level (dB) whose mean range cut through labelled objects is
   closest to the real one (src/measures.range_cut);
5. the clutter profile: real minus simulated static level per range
   (src/measures.clutter_rows), zero mean, within +/-12 dB;
6. the range gain measured again with all of these, a +/-1.5 dB search around
   the offsets with it, and a check.
Matching only the medians (the first version) left a tail of labelled objects
buried in noise on CARRADA, where every real labelled object is visible
(docs/notes/2026-11-02-simulators.md).
Simulated pools here use seeds TUNE_SEED0 + i, apart from the training versions'.

    PYTHONPATH=. python3 scripts/tune_simulators.py --dataset carrada
"""
import argparse
import itertools
import json

import numpy as np
import torch

from src.data import load, pack_labels
from src.measures import clutter_rows, measure, range_cut
from src.tuned import TUNING, real_scenes, row_profile, simulator
from src.versions import DATASETS

TUNE_SEED0, N_POOL = 5_000_000, 30
GRID_T, GRID_S = (0.0, -6.0, -12.0, -18.0), (0.0, -6.0, -12.0)
GRID_DET = (None, 6.0, 10.0, 14.0, 18.0)          # annotation threshold (dB SNR); None: all labelled
GRID_SNR, GRID_SD, GRID_S_TRACK = (6.0, 10.0, 14.0, 18.0, 22.0), (2.0, 4.0, 6.0), (-6.0, 0.0, 6.0)
QUARTILES = (25, 50, 75)


def pool(dataset, switches, tables, n=N_POOL, **override):
    """n simulated sequences as src.data items, normalised by their own stats."""
    dev = torch.device("cuda") if dataset == "radial" else None
    items = []
    for i in range(n):
        sim = simulator(dataset, switches, torch.Generator().manual_seed(TUNE_SEED0 + i), dev, tables)
        for k, v in override.items():
            setattr(sim, k, v)
        o = sim.gen_sequence()
        traj, present = pack_labels(o["labels"], 8, slot_by_rank=dataset != "radial")
        items.append({"x": o["x"].half(), "traj": traj, "present": present})
    x = torch.stack([it["x"].float() for it in items])
    stats = {"mean": float(x.mean()), "std": float(x.std())}
    for it in items:
        it["stats"] = stats
    return items


def quartiles(items):
    m = measure(items)
    return {k: np.percentile(m[k], QUARTILES).round(3).tolist() for k in ("over_floor", "over_clutter")}


def gap(q, real):
    return float(sum(abs(a - b) for k in real for a, b in zip(q[k], real[k])))


def labels_per_frame(items):
    return float(sum(int(it["present"].sum()) for it in items)) / (8 * len(items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    a = ap.parse_args()
    ds = DATASETS[a.dataset]
    real = load(ds.real, "train")
    meta = torch.load(f"{ds.real}/train_labels.pt")["meta"]
    tb = {"scenes": real_scenes(a.dataset, meta), "source": f"{ds.real} train (50 sequences)"}
    real_prof = row_profile(real)
    tb["real"] = quartiles(real)
    print("real", tb["real"], flush=True)
    base = ["processing", "scenes", "label"]
    dets = GRID_DET if a.dataset != "radial" else (None,)   # RADIal labels come from lidar and camera

    # The scenes are real labelled objects, all of which the real annotation found:
    # a threshold may drop at most 10% of their labels.
    n_all = labels_per_frame(pool(a.dataset, base, tb))

    track = a.dataset != "radial"

    def search(points):
        """points: (annotation threshold, target, static); for CARRADA and RADDet the target
        is (mean, sd) of the objects' SNR (dB), for RADIal an offset (dB)."""
        out = []
        for d, t, s in points:
            strength = {"target_snr_db": t} if track else {"target_offset_db": t}
            items = pool(a.dataset, base + ["range_gain"], tb, static_offset_db=s, label_min_snr_db=d, **strength)
            q, kept = quartiles(items), labels_per_frame(items) / n_all
            out.append({"det_db": d, "target_db": t, "static_db": s, "quartiles": q, "kept": kept,
                        "gap": gap(q, tb["real"]) + (100.0 if kept < 0.9 else 0.0)})
            print(f"det {d} target {t} static {s:+.1f}: {q} kept {kept:.2f} gap {out[-1]['gap']:.2f}", flush=True)
        return out

    def around(b, step, det_step=None):
        ds_ = [b["det_db"]] if det_step is None or b["det_db"] is None else [b["det_db"] + k * det_step for k in (-1, 0, 1)]
        if track:                    # (mean, sd): mean +/- step, sd +/- 1
            m, sd = b["target_db"]
            ts = [(m + i * step, max(sd + k, 0.5)) for i in (-1, 0, 1) for k in (-1, 0, 1)]
        else:
            ts = [b["target_db"] + i * step for i in (-1, 0, 1)]
        return [(d, t, b["static_db"] + j * step) for d in ds_ for t in ts for j in (-1, 0, 1)]

    tb["range_gain_db"] = (real_prof - row_profile(pool(a.dataset, base, tb))).tolist()
    if track:                        # objects' SNR set from a distribution: no annotation threshold needed
        grid = search([(None, (m, sd), s) for m in GRID_SNR for sd in GRID_SD for s in GRID_S_TRACK])
    else:
        grid = search([(d, t, s) for d in dets for t, s in itertools.product(GRID_T, GRID_S)])
    grid += search(around(min(grid, key=lambda g: g["gap"]), 3.0 if not track else 2.0, det_step=2.0))
    b = min(grid, key=lambda g: g["gap"])
    # the gain absorbs the static scene's level per row: measure it again with these values
    fitted = ({"target_snr_db": b["target_db"]} if track else {"target_offset_db": b["target_db"]}) | {
        "static_offset_db": b["static_db"], "label_min_snr_db": b["det_db"]}
    tb["range_gain_db"] = (real_prof - row_profile(pool(a.dataset, base, tb, **fitted))).tolist()

    if a.dataset == "radial":       # body level: the mean range cut through labelled objects
        real_cut, fits = range_cut(real), []
        for level in (-12.0, -9.0, -6.0, -3.0, 0.0):
            cut = range_cut(pool(a.dataset, base + ["range_gain"], tb, body_db=level, **fitted))
            fits.append((float((cut - real_cut).abs().mean()), level))
            print(f"body {level:+.0f} dB: range-cut L1 {fits[-1][0]:.3f}", flush=True)
        tb["body_db"], tb["body_fit"] = min(fits)[1], fits
        base.append("body")

    zero = None if a.dataset == "radial" else 32
    diff = clutter_rows(real, zero) - clutter_rows(pool(a.dataset, base + ["range_gain"], tb, **fitted), zero)
    tb["clutter_profile_db"] = (diff - diff.mean()).clamp(-12, 12).tolist()
    base.append("clutter_profile")
    tb["range_gain_db"] = (real_prof - row_profile(pool(a.dataset, base, tb, **fitted))).tolist()
    final = search(around(b, 1.5))
    b = min(final, key=lambda g: g["gap"])
    tb["label_min_snr_db"] = b["det_db"]
    tb["target_snr_db"] = list(b["target_db"]) if track else None
    tb["contrast"] = {"target_db": 0.0 if track else b["target_db"], "static_db": b["static_db"], "grid": grid + final,
                      "method": "coarse grid over (annotation threshold, target, static), +/-3 dB around the "
                                "best, range gain measured again, +/-1.5 dB; least summed |quartile gap|"}
    tb["contrast"]["check"] = quartiles(pool(a.dataset, base + ["range_gain", "contrast"], tb))
    tb["fixes_fitted_with"] = base + ["range_gain", "contrast"]
    print("fit det", tb["label_min_snr_db"], {k: v for k, v in tb["contrast"].items() if k != "grid"}, flush=True)
    TUNING.mkdir(parents=True, exist_ok=True)
    (TUNING / f"{a.dataset}.json").write_text(json.dumps(tb))


if __name__ == "__main__":
    main()
