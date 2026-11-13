"""Executable demonstrations, calibration, data and detection workflows."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from .config import load_config, save_config


def _array(path, key="image"):
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        with data:
            return np.array(data[key])
    return data


def _dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _demo(args):
    from .simulation import sample_scene, generate_reflections, save_reflections
    from .radar import simulate
    from .datasets.synthetic import scene_annotations, calibrated_psf
    from .visualization import plot_demo, plot_radar, plot_comparison
    config = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed if args.seed is None else args.seed)
    scene = sample_scene(config.radar, config.scene, rng)
    reflections = generate_reflections(scene, config.radar, config.scene, rng)
    result = simulate(reflections, config.radar, calibrated_psf(config.radar), rng)
    annotations = scene_annotations(scene, config.radar, reflections)
    boxes = [a["box"] for a in annotations]
    np.savez_compressed(out / "radar_arrays.npz", image=result.image, tensor=result.tensor, psf=result.psf)
    save_reflections(out / "reflections.npz", reflections)
    _dump(out / "scene.json", {"scene": scene.to_dict(), "objects": annotations, "simulation": result.metadata})
    save_config(config, out / "config.json")
    plot_demo(scene, reflections, result, config.radar, out / "demo_overview.png", boxes)
    plot_radar(result.image, config.radar, out / "synthetic_radar.png", boxes)
    if args.real_image:
        plot_comparison(result.image, _array(args.real_image), config.radar, out / "synthetic_real_comparison.png")
    print(f"Saved radar images, full 3D tensor, reflections and annotations to {out}")


def _generate(args):
    if not args.legacy_ra:
        from .datasets.generation_v2 import generate_bev_dataset
        path = generate_bev_dataset(args.out, load_config(args.config),
            {"train": args.train, "val": args.val, "test": args.test},
            scene_backend=args.scene_backend, max_samples=args.max_samples, save_tensors=not args.no_tensors)
        print(f"Saved Cartesian v2 {args.train}/{args.val}/{args.test} train/val/test scenes to {path}")
        return
    from .datasets import generate_dataset
    path = generate_dataset(args.out, load_config(args.config),
                            {"train": args.train, "val": args.val, "test": args.test}, args.save_tensors)
    print(f"Saved {args.train}/{args.val}/{args.test} train/val/test scenes to {path}")


def _calibrate(args):
    from .radar import measure_psf, estimate_noise, save_psf
    config = load_config(args.config)
    tensors = _array(args.tensors, "tensors")
    mask = _array(args.noise_mask, "mask")
    if tensors.ndim != 4 or tensors.shape[1:] != config.radar.shape:
        raise ValueError("Calibration input must be (frames,range,doppler,azimuth) matching --config")
    if mask.dtype != np.bool_ or mask.shape != config.radar.shape or not mask.any():
        raise ValueError("Noise mask must be a nonempty bool (range,doppler,azimuth) array")
    if np.iscomplexobj(tensors) != (config.radar.representation == "complex"):
        raise ValueError("Tensor real/complex representation must match configured representation")
    noise = estimate_noise(tensors[:, mask])
    kernel, metadata = measure_psf(tensors, args.target, args.half_widths, config.radar.psf_energy,
                                    noise["variance"], normalize_peak=False)
    metadata["noise"] = noise
    metadata["grid"] = {"range_step": config.radar.range_step,
                        "velocity_step": config.radar.velocity_step,
                        "angular_step": config.radar.angular_step,
                        "azimuth_coordinate": config.radar.azimuth_coordinate,
                        "representation": config.radar.representation}
    metadata["axis_order"] = ["range", "doppler", "azimuth"]
    metadata["complete_sensor_calibration"] = False
    metadata["notes"] = "Scattering scale, noise law/covariance, target gain and stationarity require separate validation."
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_psf(out / "calibration.npz", kernel, metadata)
    _dump(out / "calibration_metadata.json", metadata)
    payload = config.to_dict()
    payload["radar"]["psf_path"] = "calibration.npz"
    payload["radar"]["noise_std"] = noise["std"]
    _dump(out / "calibrated_config.json", payload)
    print(f"Saved stationary-target PSF, noise moments and grid-checked config to {out}")


def _simulate_points(args):
    from .simulation import load_reflections
    from .radar import simulate
    from .datasets.synthetic import calibrated_psf
    from .visualization import plot_radar
    config = load_config(args.config)
    reflections = load_reflections(args.points)
    result = simulate(reflections, config.radar, calibrated_psf(config.radar), np.random.default_rng(config.seed))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "radar_arrays.npz", image=result.image, tensor=result.tensor, psf=result.psf)
    _dump(out / "simulation.json", result.metadata)
    plot_radar(result.image, config.radar, out / "radar.png", title="Radar image from user-supplied reflection coefficients")
    print(f"Saved reflection-to-radar simulation to {out}")


def _import_real(args):
    from .datasets.external import import_manifest
    out = import_manifest(args.manifest, args.out, load_config(args.config))
    print(f"Imported explicitly gridded real radar arrays to {out}")


def _visualize(args):
    from .datasets import RadarDataset
    from .visualization import plot_radar
    dataset = RadarDataset(args.data, args.split)
    record = dataset.samples[args.index]
    annotation = json.loads((dataset.root / record["annotation"]).read_text(encoding="utf-8"))
    boxes = [o["box"] for o in annotation["objects"] if o["label"] == "car"]
    predictions = None
    if args.checkpoint:
        import torch
        from .training import load_checkpoint
        from .models import detect
        torch.set_num_threads(1)
        model, checkpoint = load_checkpoint(args.checkpoint, "cpu")
        dataset.normalization = checkpoint["normalization"]
        item = dataset[args.index]
        result = detect(model, item["image"].unsqueeze(0), score_threshold=args.score_threshold, top_k=args.top_k)[0]
        predictions = {k: v.detach().cpu().numpy() for k, v in result.items()}
    title = record["id"]
    if predictions is not None:
        title += f" · reconstruction ({checkpoint.get('global_step', '?')} training updates)"
    path = plot_radar(_array(dataset.root / record["image"]), dataset.config.radar, args.out,
                       boxes, predictions, title=title)
    print(f"Saved annotation/prediction overlay to {path}")


def _protocol(args):
    from .datasets.protocol import scene_id_for_frame, split_for_frame
    _ids = [int(s.strip()) for s in Path(args.frames).read_text(encoding="utf-8").splitlines() if s.strip()]
    output = {"train": [], "test": [], "unassigned": []}
    for frame in _ids:
        split = split_for_frame(frame)
        output["unassigned" if split is None else split].append({"frame": frame, "scene_id": scene_id_for_frame(frame)})
    _dump(args.out, output)
    print(json.dumps({k: len(v) for k, v in output.items()}))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "native":
        from .native.cli import main as native_main
        return native_main(argv[1:])
    if argv and argv[0] in ("bev-train", "bev-evaluate", "bev-infer", "bev-visualize"):
        from .bev.cli import train_main, evaluate_main, infer_main
        command = {"bev-train": train_main, "bev-evaluate": evaluate_main,
                   "bev-infer": infer_main, "bev-visualize": infer_main}[argv[0]]
        return command(argv[1:])
    if argv and argv[0] == "bev-pretrain":
        from .bev.pretrain import main as pretrain_main
        return pretrain_main(argv[1:])
    # The main training commands use Cartesian v2 data. Preserve explicitly
    # labelled compatibility when an existing v1 dataset is supplied.
    if argv and argv[0] in ("train", "evaluate", "infer", "visualize"):
        data_path = None
        for i, value in enumerate(argv[1:], 1):
            if value == "--data" and i + 1 < len(argv):
                data_path = argv[i + 1]
            elif value.startswith("--data="):
                data_path = value.split("=", 1)[1]
        version = None
        if data_path is not None and (Path(data_path) / "manifest.json").is_file():
            version = json.loads((Path(data_path) / "manifest.json").read_text(encoding="utf-8")).get("format_version")
        if version == 2 or (data_path is None and argv[0] != "visualize"):
            from .bev.cli import train_main, evaluate_main, infer_main
            dispatch = {"train": train_main, "evaluate": evaluate_main, "infer": infer_main, "visualize": infer_main}
            return dispatch[argv[0]](argv[1:])
    # Delegate these parsers to the modules also exposed as standalone scripts.
    if argv and argv[0] == "train":
        from .training.engine import main as train_main
        return train_main(argv[1:])
    if argv and argv[0] in ("evaluate", "infer"):
        from .evaluation.runner import main as evaluate_main, infer_main
        return (evaluate_main if argv[0] == "evaluate" else infer_main)(argv[1:])
    parser = argparse.ArgumentParser(description="RadSimReal independent reconstruction; see REPRODUCTION_NOTES.md")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("demo", help="Create one scene, reflections, 3D tensor and radar figures")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="runs/demo")
    p.add_argument("--seed", type=int)
    p.add_argument("--real-image", help="Optional linear real image on exactly the same grid")
    p.set_defaults(run=_demo)
    p = commands.add_parser("generate", help="Generate independently sampled split datasets")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--train", type=int, default=64)
    p.add_argument("--val", type=int, default=16)
    p.add_argument("--test", type=int, default=16)
    p.add_argument("--save-tensors", action="store_true")
    p.add_argument("--no-tensors", action="store_true", help="v2: omit full tensors; RADDet training unavailable")
    p.add_argument("--legacy-ra", action="store_true", help="Generate the former v1 RA-label dataset explicitly")
    p.add_argument("--scene-backend", choices=("mesh", "procedural"), default="mesh")
    p.add_argument("--max-samples", type=int, default=20000)
    p.set_defaults(run=_generate)
    p = commands.add_parser("calibrate", help="Average stationary target tensors and estimate noise from a supplied ROI")
    p.add_argument("--config", required=True)
    p.add_argument("--tensors", required=True)
    p.add_argument("--noise-mask", required=True)
    p.add_argument("--target", nargs=3, type=int, required=True, metavar=("R", "D", "A"))
    p.add_argument("--half-widths", nargs=3, type=int)
    p.add_argument("--out", default="runs/calibration")
    p.set_defaults(run=_calibrate)
    p = commands.add_parser("simulate-points", help="Form radar from calibrated external reflector NPZ")
    p.add_argument("--points", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="runs/points")
    p.set_defaults(run=_simulate_points)
    p = commands.add_parser("import-real", help="Import a user-converted, scene-disjoint JSONL manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(run=_import_real)
    p = commands.add_parser("visualize", help="Draw labels and optional checkpoint predictions")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--checkpoint")
    p.add_argument("--score-threshold", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=20, help="Maximum pre-NMS candidates for a legible overlay")
    p.add_argument("--out", default="runs/predictions.png")
    p.set_defaults(run=_visualize)
    p = commands.add_parser("protocol", help="Apply published scene ranges to an observed RADDet frame list")
    p.add_argument("--frames", required=True, help="Text file: one observed integer frame ID per line")
    p.add_argument("--out", default="split_manifest.json")
    p.set_defaults(run=_protocol)
    for name in ("train", "evaluate", "infer"):
        commands.add_parser(name, help=f"Use radsimreal {name} --help for arguments")
    for name in ("native", "bev-train", "bev-evaluate", "bev-infer", "bev-visualize", "bev-pretrain"):
        commands.add_parser(name, help=f"Use radsimreal {name} --help for v2 native/Cartesian workflows")
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
