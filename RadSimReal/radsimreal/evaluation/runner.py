"""Evaluation and inference for generated or explicitly converted real datasets."""

import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from .metrics import evaluate_predictions
from ..models.inference import detect
from ..training.engine import collate_detection, load_checkpoint


def predict_dataset(checkpoint, data_root, split="test", batch_size=4, device="cpu", score_threshold=0.05,
                    nms_threshold=0.3, top_k=100, num_threads=1):
    from ..datasets import RadarDataset

    if batch_size < 1 or num_threads < 1:
        raise ValueError("batch_size and num_threads must be positive")
    torch.set_num_threads(num_threads)
    model, checkpoint_data = load_checkpoint(checkpoint, device)
    dataset = RadarDataset(data_root, split=split, normalization=checkpoint_data["normalization"])
    if not len(dataset):
        raise ValueError(f"split {split!r} is empty")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_detection, num_workers=0)
    predictions, targets, ids = [], [], []
    for batch in loader:
        detected = detect(model, batch["image"].to(device), score_threshold=score_threshold,
                          nms_threshold=nms_threshold, top_k=top_k)
        predictions.extend({"boxes": p["boxes"].cpu(), "scores": p["scores"].cpu()} for p in detected)
        targets.extend(batch["boxes"])
        ids.extend(batch["id"])
    return predictions, targets, ids


def evaluate(checkpoint, data_root, split="test", batch_size=4, device="cpu", score_threshold=0.05,
             nms_threshold=0.3, top_k=100, output=None, num_threads=1):
    predictions, targets, ids = predict_dataset(
        checkpoint, data_root, split, batch_size, device, score_threshold, nms_threshold, top_k, num_threads,
    )
    metrics = evaluate_predictions(predictions, targets)
    metrics.update({
        "checkpoint": str(checkpoint), "split": split, "image_ids": ids,
        "score_threshold": score_threshold, "nms_threshold": nms_threshold,
        "top_k": top_k, "published_result_reproduced": False,
        "purpose": "evaluation_of_reconstructed_baseline_on_supplied_dataset",
    })
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    return metrics


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate car AP at IoU .1/.3/.5 on normalized RA image boxes")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", default="metrics.json")
    args = parser.parse_args(argv)
    result = evaluate(args.checkpoint, args.data, split=args.split, batch_size=args.batch_size,
                      device=args.device, score_threshold=args.score_threshold,
                      nms_threshold=args.nms_threshold, top_k=args.top_k, output=args.out,
                      num_threads=args.threads)
    print(json.dumps({key: value for key, value in result.items() if key != "curves"}, indent=2))


def infer_main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Save detections for a dataset split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="predictions.json")
    args = parser.parse_args(argv)
    predictions, _, ids = predict_dataset(args.checkpoint, args.data, args.split,
                                           device=args.device, score_threshold=args.score_threshold)
    rows = [{"id": frame_id, "boxes": prediction["boxes"].tolist(), "scores": prediction["scores"].tolist()}
            for frame_id, prediction in zip(ids, predictions)]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved predictions for {len(rows)} images to {output}")


if __name__ == "__main__":
    main()
