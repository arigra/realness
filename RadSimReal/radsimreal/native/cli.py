"""Commands to execute beside native RADDet/CARRADA data on an HPC host."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .workflows import (make_reader, prepare_native, inspect_native, propose_calibration,
                        reader_from_prepared, default_bev_grid, assert_sensor_sampling)
from .calibration import calibrate_native, json_write


def _source_arguments(parser):
    parser.add_argument("--dataset", required=True, choices=("raddet", "carrada"))
    parser.add_argument("--root", required=True, help="Dataset path on the machine executing this command")
    parser.add_argument("--profile", help="Explicit NativeProfile JSON/YAML, overrides source reference defaults")
    parser.add_argument("--representation", choices=("complex", "power", "amplitude", "db"), help="Required for ambiguous real-valued native tensors")
    parser.add_argument("--annotations", help="CARRADA annotations_frame_oriented.json when stored separately")
    parser.add_argument("--trusted-labels", action="store_true", help="Read pickle labels from your trusted original RADDet dataset")
    parser.add_argument("--reference-cartesian", action="store_true", help="Explicitly enable the official-helper-derived RADDet Cartesian affine; verify the overlay locally")


def _reader(args):
    return make_reader(args.root, args.dataset, profile=args.profile, representation=args.representation,
                       trusted_labels=args.trusted_labels, annotation_path=args.annotations,
                       reference_cartesian=args.reference_cartesian)


def _grid(path):
    if not path:
        return None
    from ..bev.geometry import BEVGrid
    return BEVGrid.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _inspect(args):
    report = inspect_native(_reader(args), args.out, args.max_frames, _grid(args.grid))
    print(json.dumps({"frames_found": report["record_count"], "sequences": report["sequences"], "out": args.out}, indent=2))


def _prepare(args):
    path = prepare_native(_reader(args), args.out, val_sequences=args.val_sequences,
                          test_sequences=args.test_sequences, split_file=args.split_file,
                          auto_split=args.auto_split, grid=_grid(args.grid),
                          fit_normalization_frames=args.fit_normalization_frames,
                          verify_all_labels=not args.quick_label_check)
    print(f"Prepared original-data-backed Cartesian dataset at {path}")


def _candidates(args):
    path = propose_calibration(args.data, args.out, args.max_frames, args.half_widths, args.max_candidates)
    print(f"Wrote unconfirmed candidates, diagnostic images and selection template: {path}")


def _calibrate(args):
    reader, manifest = reader_from_prepared(args.data)
    allowed = set(manifest.get("calibration_record_ids", [row["id"] for row in manifest["splits"]["train"]]))
    path = calibrate_native(reader, args.selection, args.out, allowed_ids=allowed,
                            noise_model=args.noise_model, phase_align=args.phase_align,
                            normalize_peak=not args.keep_target_gain)
    print(f"Saved measured native response/noise products and provenance: {path}")


def _gain(args):
    from ..datasets.generation_v2 import fit_relative_scattering_gain
    path = fit_relative_scattering_gain(args.data, args.config, args.out,
        real_frames=args.real_frames, synthetic_frames=args.synthetic_frames,
        scene_backend=args.scene_backend, max_samples=args.max_samples, quantile=args.quantile,
        vehicle_assets_path=args.vehicle_assets)
    print(f"Saved explicit TRAIN-only empirical relative gain fit to {path}")


def _generate(args):
    from ..datasets.generation_v2 import generate_bev_dataset
    from ..native.profiles import NativeProfile
    from ..config import load_config
    grid = _grid(args.grid)
    normalization = None
    profile = NativeProfile.load(args.native_profile) if args.native_profile else None
    if args.data:
        reader, manifest = reader_from_prepared(args.data)
        from ..bev.geometry import BEVGrid
        grid = grid or BEVGrid.from_dict(manifest["bev_grid"])
        normalization = manifest["normalization"]
        if args.export_native:
            profile = reader.profile
    if args.export_native and profile is None:
        raise ValueError("Native export requires --data prepared_native or --native-profile")
    config = load_config(args.config)
    if profile is not None or args.data:
        assert_sensor_sampling(config.radar, profile if profile is not None else reader.profile)
    path = generate_bev_dataset(args.out, config, {"train": args.train, "val": args.val, "test": args.test},
        grid=grid, normalization=normalization, scene_backend=args.scene_backend, visible_only=not args.all_objects,
        samples_per_square_meter=args.samples_per_square_meter, max_samples=args.max_samples,
        static_mesh_path=args.static_mesh, native_profile=profile,
        save_tensors=not args.no_tensors, seed=args.seed, vehicle_assets_path=args.vehicle_assets)
    print(f"Generated Cartesian/native-compatible synthetic data at {path}")


def _validate(args):
    from .validation import validate_distribution
    report = validate_distribution(args.data, args.synthetic, args.out, split=args.split,
                                    max_frames=args.max_frames)
    print(json.dumps({k: v for k, v in report.items() if k != "datasets"}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native RADDet/CARRADA simulation and calibration workflows")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect", help="Read actual native tensors and draw coordinate/label overlays")
    _source_arguments(p)
    p.add_argument("--out", default="runs/native_inspection")
    p.add_argument("--max-frames", type=int, default=4)
    p.add_argument("--grid")
    p.set_defaults(run=_inspect)
    p = sub.add_parser("prepare", help="Index raw arrays and preserve original Cartesian/native annotations")
    _source_arguments(p)
    p.add_argument("--out", required=True)
    p.add_argument("--auto-split", action="store_true", help="Disclosed deterministic scene/sequence pilot split, not an author CARRADA manifest")
    p.add_argument("--val-sequences", nargs="+")
    p.add_argument("--test-sequences", nargs="+")
    p.add_argument("--split-file")
    p.add_argument("--grid", help="BEVGrid JSON; the grid must contain all original target boxes")
    p.add_argument("--fit-normalization-frames", type=int, default=16)
    p.add_argument("--quick-label-check", action="store_true", help="Check only first12 labels; full label checks are default")
    p.set_defaults(run=_prepare)
    p = sub.add_parser("candidates", help="Propose unconfirmed point-target/noise regions and write visual diagnostics")
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="runs/calibration_candidates")
    p.add_argument("--max-frames", type=int, default=16)
    p.add_argument("--half-widths", nargs=3, type=int, default=[4, 2, 10])
    p.add_argument("--max-candidates", type=int, default=8)
    p.set_defaults(run=_candidates)
    p = sub.add_parser("calibrate", help="Measure PSF/noise from selected TRAIN frames, streaming one tensor at a time")
    p.add_argument("--data", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--out", default="runs/native_calibration")
    p.add_argument("--noise-model", choices=("empirical", "gaussian"), default="empirical")
    p.add_argument("--phase-align", action="store_true")
    p.add_argument("--keep-target-gain", action="store_true")
    p.set_defaults(run=_calibrate)
    p = sub.add_parser("gain", help="Optional unlabeled training-only relative power-quantile gain fit")
    p.add_argument("--data", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--real-frames", type=int, default=16)
    p.add_argument("--synthetic-frames", type=int, default=4)
    p.add_argument("--scene-backend", choices=("mesh", "procedural"), default="mesh")
    p.add_argument("--max-samples", type=int, default=10000)
    p.add_argument("--quantile", type=float, default=.995)
    p.add_argument("--vehicle-assets", help="Moving vehicle CAD/RF-material manifest JSON")
    p.set_defaults(run=_gain)
    p = sub.add_parser("generate", help="Generate full tensors, original Cartesian labels and optional native-layout files")
    p.add_argument("--config", required=True)
    p.add_argument("--data", help="Prepared native dataset supplies grid, fixed normalization and native profile")
    p.add_argument("--out", required=True)
    p.add_argument("--train", type=int, default=64)
    p.add_argument("--val", type=int, default=16)
    p.add_argument("--test", type=int, default=16)
    p.add_argument("--grid")
    p.add_argument("--native-profile")
    p.add_argument("--export-native", action="store_true")
    p.add_argument("--scene-backend", choices=("mesh", "procedural"), default="mesh")
    p.add_argument("--static-mesh", help="Optional static OBJ or MeshScene NPZ, in scene-world metres")
    p.add_argument("--vehicle-assets", help="Moving vehicle CAD/RF-material manifest JSON")
    p.add_argument("--samples-per-square-meter", type=float, default=4)
    p.add_argument("--max-samples", type=int, default=20000)
    p.add_argument("--all-objects", action="store_true", help="Include geometrical objects with no simulated return")
    p.add_argument("--no-tensors", action="store_true", help="Saves RA images only; RADDet 3D training then unavailable")
    p.add_argument("--seed", type=int)
    p.set_defaults(run=_generate)
    p = sub.add_parser("validate", help="Compare held-out real/synthetic distributions and physical-coordinate figures")
    p.add_argument("--data", required=True)
    p.add_argument("--synthetic", required=True)
    p.add_argument("--out", default="runs/sim_real_validation")
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--max-frames", type=int, default=32)
    p.set_defaults(run=_validate)
    p = sub.add_parser("run", help="Execute the on-cluster pilot with explicit calibration assumptions")
    _source_arguments(p)
    p.add_argument("--out", required=True)
    p.add_argument("--selection", help="Verified calibration selection JSON; omitted selection only allowed with --allow-provisional")
    p.add_argument("--allow-provisional", action="store_true", help="Run with UNCONFIRMED automatic target/noise candidates; no sensor-realism claim")
    p.add_argument("--split-file")
    p.add_argument("--val-sequences", nargs="+")
    p.add_argument("--test-sequences", nargs="+")
    p.add_argument("--grid")
    p.add_argument("--models", nargs="+", choices=("unet", "raddet", "probabilistic"), default=["unet"])
    p.add_argument("--train", type=int, default=16)
    p.add_argument("--val", type=int, default=4)
    p.add_argument("--test", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=4, help="0 runs all epochs; positive values bound a pilot")
    p.add_argument("--base-channels", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fit-gain", action="store_true")
    p.add_argument("--raddet-pretrain", action="store_true")
    p.add_argument("--train-real-baseline", action="store_true")
    p.add_argument("--max-samples", type=int, default=10000)
    p.add_argument("--half-widths", nargs=3, type=int, default=[4, 2, 10])
    p.add_argument("--resume-run", action="store_true")
    p.add_argument("--vehicle-assets", help="Moving vehicle CAD/RF-material manifest JSON")
    p.add_argument("--matching-policy", choices=("best_overall", "best_unmatched"), default="best_overall")
    p.add_argument("--ap-aggregation", choices=("global_pr", "frame_mean"), default="global_pr")
    p.set_defaults(run=lambda a: __import__("radsimreal.native.runner", fromlist=["run_native_pilot"]).run_native_pilot(a))
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    args.run(args)
