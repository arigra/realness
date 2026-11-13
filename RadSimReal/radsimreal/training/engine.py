"""Coordinate-U-Net training with a disclosed, chosen AdamW recipe.

RadSimReal Section 4 names architectures but does not disclose this recipe.
This implementation uses fixed preprocessing, shuffled training batches,
unshuffled validation, best-validation-loss selection, AdamW, clipped gradients,
and complete optimizer checkpoints. A smoke run is not a published experiment.
"""

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import random
import shutil
import uuid
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models import CoordinateUNet, detection_loss


def collate_detection(samples):
    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "boxes": [sample["boxes"] for sample in samples],
        "id": [sample["id"] for sample in samples],
    }


def load_checkpoint(path, device="cpu"):
    """Load a state-dictionary checkpoint with PyTorch's restricted unpickler."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("model_name") != "coordinate_unet_reconstruction":
        raise ValueError("unsupported checkpoint format/model")
    model = CoordinateUNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _validation_loss(model, loader, device):
    model.eval()
    loss_sum = 0.0
    n_images = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            boxes = [box.to(device) for box in batch["boxes"]]
            loss = detection_loss(model(image), boxes)["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite validation loss")
            loss_sum += float(loss) * image.shape[0]
            n_images += image.shape[0]
    return loss_sum / n_images


def _restore_historical_best(resume_path, checkpoint, destination):
    """Retain/copy prior best weights only when their lineage is established.

    A last checkpoint can record an earlier best loss without containing those
    earlier weights. Never assign that loss to the current model. Prefer the
    sibling best.pt; a resumed best checkpoint itself is also a valid source.
    Legacy checkpoints are checked against their configuration and exact
    history prefix; newer ones additionally carry a run identifier.
    """
    prior_best = float(checkpoint.get("best_val_loss", float("inf")))
    source_history = checkpoint.get("history", [])
    for candidate_path in (Path(resume_path).parent / "best.pt", Path(resume_path)):
        if not candidate_path.is_file():
            continue
        try:
            candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError):
            continue
        candidate_history = candidate.get("history", [])
        same_config = all(candidate.get(key) == checkpoint.get(key) for key in (
            "format_version", "model_name", "model_config", "normalization", "dataset_config",
        ))
        same_run = not checkpoint.get("run_id") or candidate.get("run_id") == checkpoint["run_id"]
        is_prefix = bool(candidate_history) and candidate_history == source_history[:len(candidate_history)]
        contains_best_weights = (
            candidate.get("best_val_loss") == prior_best
            and candidate_history and candidate_history[-1].get("val_loss") == prior_best
            and int(candidate.get("global_step", -1)) <= int(checkpoint["global_step"])
        )
        if same_config and same_run and is_prefix and contains_best_weights:
            if candidate_path.resolve() != destination.resolve():
                shutil.copy2(candidate_path, destination)
            scope = ("continued_resumed_segment_with_historical_best_weights"
                     if "segment" in checkpoint.get("best_selection_scope", "")
                     else "continued_run_with_historical_best_weights")
            return prior_best, scope, prior_best
    # The earlier best model is unavailable: begin a disclosed segment-level
    # selection rather than returning a nonexistent or incorrectly labeled file.
    if destination.exists():
        raise ValueError("Cannot validate existing best.pt against resumed history; use a new output directory")
    return float("inf"), "resumed_segment_only_historical_best_weights_unavailable", prior_best


def train(
    data_root, output_dir, epochs=10, batch_size=4, max_steps=None, seed=0,
    device="cpu", base_channels=16, learning_rate=1e-3, weight_decay=1e-4,
    normalization=None, resume=None, num_threads=1,
):
    """Return (best_checkpoint_path, history); max_steps bounds total updates.

    Train and val splits must both be nonempty and distinct. Test samples never
    enter training, normalization fitting, validation or model selection.
    Resume restores model, optimizer, epoch and global step. A resume starts at
    the next epoch; interrupted epochs are not replayed exactly. RNG/shuffle
    state is reseeded, so resume is functional rather than bitwise continuation.
    Historical best weights are retained or copied into a new output directory
    when available. If only a non-best last checkpoint remains, model selection
    covers the resumed segment and records the unavailable earlier best loss.
    """
    from ..datasets import RadarDataset

    if epochs < 1 or batch_size < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError("epochs, batch_size and max_steps must be positive")
    if num_threads < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid threads or optimizer settings")
    torch.set_num_threads(num_threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if resume is not None:
        model, checkpoint = load_checkpoint(resume, device)
        normalization = checkpoint["normalization"]
        base_channels = checkpoint["model_config"]["base_channels"]
    else:
        model = CoordinateUNet(base_channels=base_channels).to(device)
    train_set = RadarDataset(data_root, split="train", normalization=normalization)
    normalization = dict(train_set.normalization)
    val_set = RadarDataset(data_root, split="val", normalization=normalization)
    if not len(train_set) or not len(val_set):
        raise ValueError("training requires nonempty train and val splits")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator,
                              collate_fn=collate_detection, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_detection, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    start_epoch, global_step, best_val = 0, 0, float("inf")
    run_id = str(uuid.uuid4())
    best_selection_scope = "all_epochs_in_run"
    prior_best_val = None
    history = []
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        history = list(checkpoint.get("history", []))
        # Keep legacy lineage without an ID history-based across later resumes.
        run_id = checkpoint.get("run_id")
    if max_steps is not None and global_step >= max_steps:
        raise ValueError("max_steps must exceed the resumed global_step")
    best_path = output_dir / "best.pt"
    if checkpoint is not None:
        best_val, best_selection_scope, prior_best_val = _restore_historical_best(resume, checkpoint, best_path)
    dataset_config = getattr(train_set, "config", None)
    if is_dataclass(dataset_config):
        dataset_config = asdict(dataset_config)
    elif hasattr(dataset_config, "to_dict"):
        dataset_config = dataset_config.to_dict()
    elif not isinstance(dataset_config, (dict, type(None))):
        dataset_config = None
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        sums = {"total": 0.0, "heatmap": 0.0, "box": 0.0, "collisions": 0.0}
        n_images = 0
        updates = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            boxes = [box.to(device) for box in batch["boxes"]]
            optimizer.zero_grad(set_to_none=True)
            losses = detection_loss(model(images), boxes)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("nonfinite training loss")
            losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            for key in ("total", "heatmap", "box"):
                sums[key] += float(losses[key].detach()) * images.shape[0]
            sums["collisions"] += int(losses["collisions"])
            n_images += images.shape[0]
            global_step += 1
            updates += 1
            if max_steps is not None and global_step >= max_steps:
                break
        val_loss = _validation_loss(model, val_loader, device)
        record = {
            "epoch": epoch, "global_step": global_step, "updates": updates,
            "train_loss": sums["total"] / n_images,
            "train_heatmap_loss": sums["heatmap"] / n_images,
            "train_box_loss": sums["box"] / n_images,
            "target_collisions": int(sums["collisions"]), "val_loss": val_loss,
            "last_gradient_norm": float(gradient_norm),
        }
        history.append(record)
        improved = val_loss < best_val
        best_val = min(best_val, val_loss)
        payload = {
            "format_version": 1, "model_name": "coordinate_unet_reconstruction",
            "fidelity": model.fidelity,
            "model_config": {"base_channels": base_channels, "in_channels": 3},
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "normalization": normalization, "dataset_config": dataset_config,
            "epoch": epoch, "global_step": global_step, "best_val_loss": best_val,
            "seed": seed, "history": history, "run_id": run_id,
            "best_selection_scope": best_selection_scope,
            "prior_best_val_loss": prior_best_val,
            "training_recipe": {
                "optimizer": "AdamW", "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
                "effective_parameter_groups": [
                    {"learning_rate": float(group["lr"]), "weight_decay": float(group["weight_decay"]),
                     "betas": tuple(group["betas"]), "eps": float(group["eps"])}
                    for group in optimizer.param_groups
                ],
                "gradient_clip_norm": 10.0,
                "loss": "gaussian_focal_plus_masked_L1_offset_plus_5x_L1_size",
                "published_recipe": False,
            },
        }
        torch.save(payload, output_dir / "last.pt")
        if improved:
            torch.save(payload, best_path)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if max_steps is not None and global_step >= max_steps:
            break
    return best_path, history


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Train the documented U-Net reconstruction (not authors' weights)")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--resume")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    checkpoint, history = train(
        args.data, args.out, epochs=args.epochs, batch_size=args.batch_size,
        max_steps=args.max_steps, seed=args.seed, device=args.device,
        base_channels=args.base_channels, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, resume=args.resume, num_threads=args.threads,
    )
    print(json.dumps({"checkpoint": str(checkpoint), "last_epoch": history[-1]}, indent=2))


if __name__ == "__main__":
    main()
