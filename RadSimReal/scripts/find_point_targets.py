"""Search TRAIN scenes for stationary, isolated, point-like reflectors (PSF measurement evidence).

Works on a prepared manifest so only declared training IDs are used. For each
scene, frames spread over the scene are loaded in canonical (range, Doppler,
azimuth) order. Candidates are local maxima of the coherent time average at
zero Doppler. Each candidate is scored by evidence, not brightness alone:

  stationarity  phase coherence |mean z| / mean |z| over frames, amplitude CV
  static        fraction of the peak's Doppler-profile energy at the zero bin
  isolation     peak-to-strongest-other-local-maximum ratio in a range/azimuth window
  point-like    -3 dB main-lobe widths in range and azimuth vs the narrowest found
  unlabeled     peak outside every native RADDet RAD box (+ margin) in all used frames

Writes candidates.json and per-candidate diagnostic PNGs. Nothing is confirmed
automatically; the report is the evidence for a human/agent decision.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def load_manifest(path):
    return json.loads(Path(path).read_text())


def lobe_width(profile, center):
    """Number of contiguous bins around center within -3 dB of the center power."""
    p = np.asarray(profile)
    half = p[center] / 2
    lo = center
    while lo > 0 and p[lo - 1] >= half:
        lo -= 1
    hi = center
    while hi < len(p) - 1 and p[hi + 1] >= half:
        hi += 1
    edge = lo == 0 or hi == len(p) - 1
    return hi - lo + 1, edge


def raddet_label_mask(record, shape_rda, profile, margin=(6, 12)):
    """Canonical (range, azimuth) mask of native RAD boxes for every labelled object."""
    R, D, A = shape_rda
    mask = np.zeros((R, A), bool)
    with open(record.annotation_path, "rb") as f:
        native = pickle.load(f)
    for box in np.asarray(native.get("boxes", []), float).reshape(-1, 6):
        r, a, _, hr, ha, _ = box  # native center (range row, azimuth col, doppler) and sizes
        rc = R - 1 - r if profile.range_reversed else r
        ac = A - 1 - a if profile.azimuth_reversed else a
        r0, r1 = int(np.floor(rc - hr / 2 - margin[0])), int(np.ceil(rc + hr / 2 + margin[0]))
        a0, a1 = int(np.floor(ac - ha / 2 - margin[1])), int(np.ceil(ac + ha / 2 + margin[1]))
        mask[max(r0, 0):min(r1 + 1, R), max(a0, 0):min(a1 + 1, A)] = True
    return mask


_CARRADA_CACHE = {}


def carrada_label_mask(record, shape_rda, profile, margin=(6, 12)):
    """Canonical (range, azimuth) mask of every annotated CARRADA instance (all classes).

    Annotation rows: range = (R - row) * step (verified). Annotation columns use the
    uniform angle step; they are mapped to the tensor's own azimuth grid here.
    """
    R, D, A = shape_rda
    mask = np.zeros((R, A), bool)
    path = str(record.annotation_path)
    if path not in _CARRADA_CACHE:
        _CARRADA_CACHE[path] = json.loads(Path(path).read_text())
    frame = _CARRADA_CACHE[path].get(record.sequence, {}).get(f"{record.frame_id:06d}", {})
    grid = np.asarray(profile.angular_grid)
    for inst in frame.values():
        box = np.asarray(inst["range_angle"]["box"], float)  # [[row0,col0],[row1,col1]]
        rows, cols = box[:, 0], box[:, 1]
        rc = np.sort(profile.shape_native[0] - rows)
        theta = (cols - profile.shape_native[1] // 2) * profile.annotation_angle_step
        u = np.sin(theta) if profile.azimuth_coordinate == "sin" else theta
        ac = np.sort(np.interp(u, grid, np.arange(len(grid))))
        r0, r1 = int(rc[0] - margin[0]), int(np.ceil(rc[1] + margin[0]))
        a0, a1 = int(ac[0] - margin[1]), int(np.ceil(ac[1] + margin[1]))
        mask[max(r0, 0):min(r1 + 1, R), max(a0, 0):min(a1 + 1, A)] = True
    return mask


def analyse_scene(reader, records, zero_bin, frames, iso_window, min_range_bin):
    step = max(1, len(records) // frames)
    used = records[::step][:frames]
    cubes = []
    label_mask = None
    for rec in used:
        cube = np.asarray(reader.load_tensor(rec))  # canonical (R, D, A); complex field or linear power
        if not np.iscomplexobj(cube):
            cube = np.sqrt(cube)  # amplitude, so |z|^2 is power; phase coherence is then uninformative (=1)
        cubes.append(cube)
        label_fn = raddet_label_mask if reader.dataset == "raddet" else carrada_label_mask
        m = label_fn(rec, cube.shape, reader.profile)
        label_mask = m if label_mask is None else (label_mask | m)
    stack = np.stack(cubes)  # (T, R, D, A)
    mean_c = stack.mean(0)
    zero = stack[:, :, zero_bin, :]  # (T, R, A)
    p0 = np.abs(mean_c[:, zero_bin, :]) ** 2
    R, A = p0.shape
    noise_floor = np.median(np.abs(stack[:, :, [0, 1, -2, -1], :]) ** 2)
    # local maxima in (range, azimuth), 3x5 neighbourhood
    from scipy.ndimage import maximum_filter
    locmax = (p0 == maximum_filter(p0, size=(3, 5))) & (p0 > 1000 * noise_floor)
    locmax[:min_range_bin] = False
    rows = []
    for r, a in zip(*np.nonzero(locmax)):
        z = zero[:, r, a]
        coherence = float(np.abs(z.mean()) / np.abs(z).mean())
        cv = float(np.abs(z).std() / np.abs(z).mean())
        dop = np.abs(mean_c[r, :, a]) ** 2
        static_frac = float(dop[zero_bin] / dop.sum())
        wr, wa = iso_window
        r0, r1, a0, a1 = max(r - wr, 0), min(r + wr + 1, R), max(a - wa, 0), min(a + wa + 1, A)
        window = locmax[r0:r1, a0:a1].copy()
        window[r - r0, a - a0] = False
        others = p0[r0:r1, a0:a1][window]
        isolation_db = float(10 * np.log10(p0[r, a] / others.max())) if others.size else 99.0
        width_r, edge_r = lobe_width(p0[:, a], r)
        width_a, edge_a = lobe_width(p0[r, :], a)
        rows.append({"range_bin": int(r), "azimuth_bin": int(a), "doppler_bin": int(zero_bin),
                     "snr_db": float(10 * np.log10(p0[r, a] / noise_floor)),
                     "phase_coherence": coherence, "amplitude_cv": cv, "zero_doppler_fraction": static_frac,
                     "isolation_db": isolation_db, "width_range_bins": width_r, "width_azimuth_bins": width_a,
                     "lobe_touches_edge": bool(edge_r or edge_a),
                     "inside_label_region": bool(label_mask[r, a])})
    return used, mean_c, rows, noise_floor


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="prepared native dataset (manifest.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=24, help="frames per scene, spread over the scene")
    ap.add_argument("--scenes", nargs="*", help="train scenes to search (default: all train)")
    ap.add_argument("--iso-window", nargs=2, type=int, default=[12, 48], help="range, azimuth half-window")
    ap.add_argument("--min-range-bin", type=int, default=12)
    args = ap.parse_args()
    manifest = load_manifest(Path(args.data) / "manifest.json")
    from radsimreal.native.workflows import reader_from_prepared
    reader, _ = reader_from_prepared(args.data)
    train_ids = {row["id"] for row in manifest["splits"]["train"]}
    by_scene = {}
    for rec in reader.records:
        if rec.id in train_ids:
            by_scene.setdefault(rec.sequence, []).append(rec)
    scenes = args.scenes or sorted(by_scene)
    profile = reader.profile
    zero_bin = int(np.argmin(np.abs(np.asarray(profile.velocities))))  # canonical axis may be flipped
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"zero_doppler_canonical_bin": zero_bin, "frames_per_scene": args.frames,
              "isolation_window_half_bins": args.iso_window, "scenes": {}}
    for scene in scenes:
        recs = sorted(by_scene[scene], key=lambda r: r.frame_id)
        used, mean_c, rows, floor = analyse_scene(reader, recs, zero_bin, args.frames, args.iso_window, args.min_range_bin)
        report["scenes"][scene] = {"frames_used": [r.id for r in used], "noise_floor_power": float(floor),
                                   "candidates": sorted(rows, key=lambda d: -d["isolation_db"])}
        np.save(out / f"{scene}_mean_zero_doppler_power.npy", np.abs(mean_c[:, zero_bin, :]) ** 2)
        print(scene, len(used), "frames", len(rows), "local maxima", flush=True)
    (out / "candidates.json").write_text(json.dumps(report, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
