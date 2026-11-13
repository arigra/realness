"""Build a RadSimReal synthetic dataset from CARLA captures (scripts/carla_capture.py).

Reflection points (radsimreal.simulation.carla_reflectors) -> measured PSF convolution
+ measured noise (radsimreal.radar.simulate) -> Doppler-max RA image. The output is a
version-2 BEV dataset readable by bev-train. Grid, fixed normalization and native
profile are taken from the prepared real dataset so both domains share preprocessing.

Splits are disjoint by sensor position (val = positions with (hash % 10 == 0)).
Each frame's RNG is seeded by (seed, global frame index): deterministic and resumable.

  --fit-gain : TRAIN-only relative gain = median real RA 0.995-quantile / median synthetic
               (unit gain, no noise) over the given numbers of frames; writes a config copy.
"""
import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np

from radsimreal.config import load_config, save_config
from radsimreal.bev.geometry import BEVGrid
from radsimreal.radar import simulate
from radsimreal.datasets.synthetic import calibrated_psf
from radsimreal.datasets.generation_v2 import materialize_sensor
from radsimreal.native.calibration import json_write
from radsimreal.simulation.carla_reflectors import capture_to_reflections, vehicle_objects, material_table, MATERIAL_SETS


def list_frames(captures):
    frames = []
    for done in sorted(Path(captures).glob("*/pos_*/DONE")):
        pos = done.parent
        for npz in sorted(pos.glob("frame_*.npz")):
            frames.append((pos.parent.name, pos.name, npz.stem))
    return frames


def is_val(town, pos):
    return int(hashlib.sha256(f"{town}/{pos}".encode()).hexdigest(), 16) % 10 == 0


_G = {}


def _init(config_path, grid_dict, captures, out, save_tensor, materials="v2_measured_road"):
    cfg = load_config(config_path)
    _G.update(cfg=cfg, psf=calibrated_psf(cfg.radar), grid=BEVGrid.from_dict(grid_dict),
              captures=Path(captures), out=Path(out), save_tensor=save_tensor, materials=MATERIAL_SETS[materials])


def _simulate(town, pos, frame, seed_index, seed, cfg=None, psf=None):
    cfg = cfg or _G["cfg"]; psf = psf if psf is not None else _G["psf"]
    base = _G["captures"] / town / pos / frame
    meta = json.loads(base.with_suffix(".json").read_text())
    with np.load(base.with_suffix(".npz")) as cap:
        capture = {k: cap[k] for k in cap.files}
    rng = np.random.default_rng([seed, seed_index])
    refl = capture_to_reflections(capture, meta["ray_solid_angle_sr"], cfg.radar, rng, materials=_G.get("materials"))
    return meta, refl, simulate(refl, cfg.radar, psf, rng)


def _write(job):
    split, name, town, pos, frame, seed_index, seed = job
    out = _G["out"]
    npz_path, json_path = out / split / f"{name}.npz", out / split / f"{name}.json"
    if json_path.is_file():
        return split, name
    meta, refl, result = _simulate(town, pos, frame, seed_index, seed)
    objects, excluded = vehicle_objects(meta, refl, _G["cfg"].radar, _G["grid"])
    arrays = {"image": result.image}
    if _G["save_tensor"]:
        arrays["tensor"] = result.tensor.astype(np.complex64)
    tmp = npz_path.with_name(npz_path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays); os.replace(tmp, npz_path)
    json_write(json_path, {"id": name, "objects": objects, "excluded_objects": excluded,
                           "source_capture": f"{town}/{pos}/{frame}", "carla_meta": {k: v for k, v in meta.items() if k != "vehicles"},
                           "simulation": result.metadata, "reflection_metadata": refl.metadata,
                           "seed_sequence": [seed, seed_index]})
    return split, name


def fit_gain(args, prepared_manifest):
    from radsimreal.datasets.bev_dataset import BEVDataset
    cfg = load_config(args.config)
    unit = deepcopy(cfg); unit.radar.scattering_gain = 1.0; unit.radar.noise_model = "none"; unit.radar.noise_std = 0
    psf = calibrated_psf(unit.radar)
    _G.update(captures=Path(args.captures), materials=MATERIAL_SETS[args.materials])
    frames = [f for f in list_frames(args.captures) if not is_val(f[0], f[1])]
    pick = np.linspace(0, len(frames) - 1, min(args.gain_synthetic_frames, len(frames))).astype(int)
    syn = [float(np.quantile(_simulate(*frames[i], 10_000_000 + int(i), args.seed, unit, psf)[2].image, args.quantile)) for i in pick]
    real_ds = BEVDataset(args.prepared, "train", include_radar=False)
    idx = np.linspace(0, len(real_ds) - 1, args.gain_real_frames).astype(int)
    real = [float(np.quantile(real_ds.raw(int(i))["image_ra"], args.quantile)) for i in idx]
    gain = float(np.median(real) / np.median(syn))
    cfg.radar.scattering_gain = gain
    save_config(cfg, args.gain_out)
    json_write(Path(args.gain_out).with_suffix(".gain_fit.json"), {
        "method": "TRAIN-only RA power quantile ratio (median real / median synthetic, unit gain, noise-free)",
        "quantile": args.quantile, "real_train_frames": len(real), "synthetic_train_frames": len(syn),
        "real_statistics": real, "synthetic_unit_gain_statistics": syn, "relative_power_scattering_gain": gain,
        "test_frames_used": 0, "absolute_RCS_calibration": False})
    print("gain", gain)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--config", required=True, help="calibrated sensor_config.json (PSF, noise, gain)")
    ap.add_argument("--prepared", required=True, help="prepared real dataset: grid, normalization, profile")
    ap.add_argument("--out")
    ap.add_argument("--train", type=int, default=10000); ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--save-tensors", action="store_true")
    ap.add_argument("--fit-gain", action="store_true"); ap.add_argument("--gain-out")
    ap.add_argument("--gain-real-frames", type=int, default=64); ap.add_argument("--gain-synthetic-frames", type=int, default=32)
    ap.add_argument("--quantile", type=float, default=0.995)
    ap.add_argument("--materials", choices=sorted(MATERIAL_SETS), default="v2_measured_road")
    args = ap.parse_args()
    prepared = json.loads((Path(args.prepared) / "manifest.json").read_text())
    if args.fit_gain:
        return fit_gain(args, prepared)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    if not (out / "config.json").exists():
        materialize_sensor(cfg, out)
    frames = list_frames(args.captures)
    train = [f for f in frames if not is_val(f[0], f[1])][: args.train]
    val = [f for f in frames if is_val(f[0], f[1])][: args.val]
    if len(train) < args.train or len(val) < args.val:
        print(f"WARNING: only {len(train)} train / {len(val)} val frames available")
    jobs = []
    for split, rows in (("train", train), ("val", val)):
        (out / split).mkdir(exist_ok=True)
        for i, (town, pos, frame) in enumerate(rows):
            seed_index = (0 if split == "train" else 1) * 1_000_000 + i
            jobs.append((split, f"{split}_{i:06d}", town, pos, frame, seed_index, args.seed))
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(str(out / "config.json"), prepared["bev_grid"], args.captures, str(out), args.save_tensors, args.materials)) as pool:
        for k, _ in enumerate(pool.map(_write, jobs, chunksize=4)):
            if (k + 1) % 500 == 0:
                print(k + 1, "/", len(jobs), flush=True)
    splits = {"train": [], "val": [], "test": []}
    for split, name, *_ in jobs:
        splits[split].append({"id": name, "image": f"{split}/{name}.npz", "annotation": f"{split}/{name}.json"})
    manifest = {"format_version": 2, "source": "generated_original_Cartesian_geometry",
                "scene_backend": "carla_semantic_lidar", "axis_order": ["range", "doppler", "azimuth"],
                "bev_grid": prepared["bev_grid"], "normalization": prepared["normalization"],
                "normalization_source": "prepared real TRAIN normalization (shared preprocessing)",
                "label_domain": "synthetic_CARLA_vehicle_footprints_car_truck_bus",
                "metric_domain": "cartesian_bev_metres_axis_aligned",
                "oriented_labels": "not used", "boundary_policy": "reject_outside_grid_no_silent_label_crop",
                "split_policy": "disjoint CARLA sensor positions (sha256(town/pos) % 10 == 0 -> val)",
                "native_profile": prepared["native_profile"], "materials": material_table(MATERIAL_SETS[args.materials]), "material_set": args.materials,
                "captures": str(Path(args.captures).resolve()), "seed": args.seed,
                "tensors_saved": args.save_tensors, "published_result_reproduced": False, "splits": splits}
    json_write(out / "manifest.json", manifest)
    print("wrote", {k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
