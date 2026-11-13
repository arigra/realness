"""Export RadSimReal (CARLA) frames as a realness range-Doppler simulator cache.

Same simulation as generate_carla_dataset.py (reflection points -> measured PSF + measured noise
-> 3D range x Doppler x azimuth tensor). The RD map is the realness RADDet convention
(realness/src/raddet.py): power summed over azimuth, dB, row 0 near; verified on a real frame:
realness rd_db == canonical RadSimReal tensor summed over azimuth with the Doppler axis reversed
(max abs difference 0.0).

Labels follow RADDet's (every road user, label = centre of the box around its returns): every
CARLA actor whose hits are pedestrian/rider/car/truck/bus/motorcycle/bicycle and that has >= 5
reflection points inside the radar domain; range centre = midpoint of its points' range bins,
Doppler = median of its points' Doppler bins. Class = the actor's majority tag.

Sequences: 8 frames (every other frame) from one CARLA drive; drives from the TRAIN positions of
the RadSimReal split, spread over the towns. Output via realness src.data.write_cache.

  .venv/bin/python scripts/export_rd_realness.py --config runs/raddet_calibration/carla_vehicle_v2_config.json \
      --materials v2_measured_road --out ../data/raddet_carla_v2
"""
import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the realness repo, for src.data
import generate_carla_dataset as G                               # noqa: E402  (same directory)
from radsimreal.radar.coordinates import points_to_indices       # noqa: E402

ROAD_USER_TAGS = {12: "person", 13: "rider", 14: "car", 15: "truck", 16: "bus", 18: "motorcycle", 19: "bicycle"}
FRAMES = list(range(0, 16, 2))


def frame_rd(town, pos, frame, seed_index, seed):
    meta, refl, result = G._simulate(town, pos, frame, seed_index, seed)
    power = np.abs(result.tensor) ** 2 if np.iscomplexobj(result.tensor) else np.maximum(result.tensor, 0)
    rd = 10 * np.log10(power.sum(axis=2) + 1e-12)[:, ::-1]
    radar = G._G["cfg"].radar
    with np.load(G._G["captures"] / town / pos / f"{frame}.npz") as cap:
        tags, ids = cap["tag"], cap["object_idx"]
    labels = []
    oid = np.asarray(refl.object_ids)
    for actor in np.unique(oid):
        hit_tags = tags[ids == actor]
        users = [int(t) for t in hit_tags if int(t) in ROAD_USER_TAGS]
        if actor == 0 or not users:
            continue
        sel = oid == actor
        idx, valid = points_to_indices(refl.xyz[sel], refl.velocity[sel], radar)
        idx = idx[valid]
        if len(idx) < 5:
            continue
        cls = ROAD_USER_TAGS[Counter(users).most_common(1)[0][0]]
        labels.append({"id": len(labels), "cls": "bicycle" if cls == "rider" else cls,
                       "range_bin": float((idx[:, 0].min() + idx[:, 0].max()) / 2),
                       "doppler_bin": float(radar.doppler_bins - 1 - np.median(idx[:, 1]))})
    return rd, labels


def make_sequence(job):
    town, pos, seq_index, seed = job
    xs, ls = [], []
    for k, f in enumerate(FRAMES):
        rd, labels = frame_rd(town, pos, f"frame_{f:03d}", seq_index * 100 + k, seed)
        xs.append(rd); ls.append(labels)
    return np.stack(xs), ls, {"recording": f"{town}/{pos}", "frames": FRAMES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="runs/carla_capture_vehicle")
    ap.add_argument("--config", required=True)
    ap.add_argument("--materials", default="v2_measured_road")
    ap.add_argument("--n-seq", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from src.data import write_cache
    frames = G.list_frames(args.captures)
    drives = sorted({(t, p) for t, p, f in frames if not G.is_val(t, p)
                     and sum(1 for tt, pp, ff in frames if (tt, pp) == (t, p)) >= max(FRAMES) + 1})
    pick = [drives[int(i)] for i in np.linspace(0, len(drives) - 1, args.n_seq).round()]
    with ProcessPoolExecutor(args.workers, initializer=G._init,
                             initargs=(args.config, {"x_min": -50, "x_max": 50, "y_min": 0, "y_max": 50, "height": 256, "width": 512},
                                       args.captures, args.out, False, args.materials)) as pool:
        seqs = list(pool.map(make_sequence, [(t, p, i, args.seed) for i, (t, p) in enumerate(pick)], chunksize=1))
    write_cache(args.out, len(seqs), lambda i: seqs[i], (256, 64), 1, slot_by_rank=True)
    n = Counter(l["cls"] for _, ls, _ in seqs for fr in ls for l in fr)
    print("wrote", args.out, len(seqs), "sequences; labels", dict(n),
          "per frame", round(sum(n.values()) / (8 * len(seqs)), 2))


if __name__ == "__main__":
    main()
