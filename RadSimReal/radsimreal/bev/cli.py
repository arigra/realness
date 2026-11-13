"""Training, evaluation and visualization commands for Cartesian v2 models."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .training import train_bev, load_bev_checkpoint, compare_sensor_sampling, enforce_sensor_sampling, collate_bev
from .evaluation import evaluate_bev
from .inference import detect_bev
from .geometry import BEVGrid, oriented_corners
from ..datasets.bev_dataset import BEVDataset


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Train Cartesian U-Net, published RADDet port or probabilistic detector")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", choices=("unet", "unet_zhang", "raddet", "probabilistic"), default="unet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (speed only)")
    parser.add_argument("--oriented", action="store_true", help="Requires genuine yaw labels for every object")
    parser.add_argument("--resume")
    parser.add_argument("--pretrained-backbone", help="RADDet 3D pretraining checkpoint from bev-pretrain")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--allow-sensor-change", action="store_true", help="Explicit native-RAD coordinate transfer experiment")
    args = parser.parse_args(argv)
    normalization = None
    model_name = args.model
    if args.resume:
        model, checkpoint = load_bev_checkpoint(args.resume, args.device)
        normalization = checkpoint["normalization"]
        model_name = checkpoint["model_config"]["name"]
    train = BEVDataset(args.data, "train", normalization=normalization, include_radar=model_name == "raddet")
    validation = BEVDataset(args.data, "val", normalization=train.normalization, include_radar=model_name == "raddet")
    path, history = train_bev(train, validation, args.out, model_name=model_name, epochs=args.epochs,
        batch_size=args.batch_size, max_steps=args.max_steps, base_channels=args.base_channels,
        device=args.device, seed=args.seed, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        resume=args.resume, num_threads=args.threads, oriented=args.oriented,
        pretrained_backbone=args.pretrained_backbone, freeze_backbone=args.freeze_backbone,
        allow_sensor_change=args.allow_sensor_change, num_workers=args.workers)
    print(json.dumps({"checkpoint": str(path), "last_epoch": history[-1], "published_result_reproduced": False}, indent=2))


def _evaluation_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=.05)
    parser.add_argument("--nms-threshold", type=float, default=.3)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--allow-label-domain-change", action="store_true",
                        help="Explicit sim-to-real or cross-dataset label-provenance comparison")
    parser.add_argument("--allow-sensor-change", action="store_true", help="Explicitly allow changed native-RAD sampling, reported as a transfer experiment")
    return parser


def evaluate_main(argv=None):
    parser = _evaluation_parser("Evaluate physical Cartesian car IoU/AP, preserving native annotation provenance")
    parser.add_argument("--out", default="runs/bev_metrics.json")
    parser.add_argument("--matching-policy", choices=("best_overall", "best_unmatched"), default="best_overall")
    parser.add_argument("--ap-aggregation", choices=("global_pr", "frame_mean"), default="global_pr",
                        help="global_pr: global PR area; frame_mean: original RADDet reference frame averaging")
    args = parser.parse_args(argv)
    model, payload = load_bev_checkpoint(args.checkpoint, args.device)
    dataset = BEVDataset(args.data, args.split, normalization=payload["normalization"],
                         include_radar=model.requires_radar, grid=model.grid)
    report = evaluate_bev(args.checkpoint, dataset, batch_size=args.batch_size, device=args.device,
        score_threshold=args.score_threshold, nms_threshold=args.nms_threshold, top_k=args.top_k,
        output=args.out, num_threads=args.threads, allow_label_domain_change=args.allow_label_domain_change,
        allow_sensor_change=args.allow_sensor_change, matching_policy=args.matching_policy,
        ap_aggregation=args.ap_aggregation)
    print(json.dumps({k: v for k, v in report.items() if k != "curves"}, indent=2))


def infer_main(argv=None):
    parser = _evaluation_parser("Save physical Cartesian predictions or original-label/prediction overlays")
    parser.add_argument("--out", default="runs/bev_predictions.json")
    parser.add_argument("--index", type=int, help="One frame, with a PNG overlay when --out ends in .png")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    model, payload = load_bev_checkpoint(args.checkpoint, args.device)
    dataset = BEVDataset(args.data, args.split, normalization=payload["normalization"],
                         include_radar=model.requires_radar, grid=model.grid)
    if dataset.label_domain != payload["label_domain"] and not args.allow_label_domain_change:
        raise ValueError("Cross-label-domain inference needs explicit --allow-label-domain-change")
    comparison = compare_sensor_sampling(payload.get("dataset_meta", {}), dataset)
    enforce_sensor_sampling(model, comparison, dataset, args.allow_sensor_change)
    indices = list(range(len(dataset))) if args.index is None else [args.index]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".png" and args.index is None:
        raise ValueError("A single PNG overlay requires --index")
    rows = []
    from torch.utils.data import DataLoader, Subset
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, collate_fn=collate_bev)
    for batch in loader:
        radar = batch.get("radar")
        predictions = detect_bev(model, batch["image"].to(args.device),
            radar.to(args.device) if model.requires_radar else None,
            args.score_threshold, args.nms_threshold, args.top_k)
        for prediction, frame_id in zip(predictions, batch["id"]):
            row = {k: v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v for k, v in prediction.items()}
            row.update(id=frame_id, label_domain=dataset.label_domain, sensor_sampling_comparison=comparison,
                       checkpoint_global_step=payload.get("global_step"))
            rows.append(row)
            if out.suffix.lower() == ".png":
                plot_bev(dataset, args.index, row, out, title=f"{frame_id} · {model.architecture} · {payload.get('global_step', '?')} updates · score ≥ {args.score_threshold:g}")
    if out.suffix.lower() != ".png":
        out.write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    else:
        out.with_suffix(".predictions.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(rows)} Cartesian prediction records to {out}")


def plot_bev(dataset, index, predictions, output, title="Cartesian radar detection"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    raw = dataset.raw(index)
    with torch.no_grad():
        power = dataset.project(torch.as_tensor(raw["image_ra"], dtype=torch.float64)).numpy()
    grid = dataset.grid
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    norm = dataset.normalization
    im = ax.imshow(10 * np.log10(np.maximum(power, norm["power_floor"])), origin="lower",
        extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max], cmap="magma",
        vmin=norm["db_min"], vmax=norm["db_max"], aspect="equal")
    for i, (x1, y1, x2, y2) in enumerate(raw["boxes_physical"]):
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "c-", lw=1.5,
                label="Original Cartesian labels" if i == 0 else None)
    oriented = predictions.get("oriented_boxes")
    if oriented is not None:
        for i, box in enumerate(oriented):
            corners = oriented_corners(box)
            corners = np.concatenate([corners, corners[:1]])
            ax.plot(corners[:, 0], corners[:, 1], "g-", label="Predicted oriented boxes" if i == 0 else None)
    else:
        for i, ((x1, y1, x2, y2), score) in enumerate(zip(predictions["boxes"], predictions["scores"])):
            ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "g-", lw=1,
                    label="Predictions" if i == 0 else None)
            ax.text(x1, y2, f"{score:.2f}", color="lime", fontsize=7)
    ax.set(xlabel="Lateral x (m)", ylabel="Forward y (m)", title=title)
    ax.set_xlim(grid.x_min, grid.x_max)
    ax.set_ylim(grid.y_min, grid.y_max)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    fig.colorbar(im, ax=ax, label="Linear-power response (dB relative units)", shrink=.8, pad=.025)
    fig.savefig(output, dpi=150)
    plt.close(fig)
