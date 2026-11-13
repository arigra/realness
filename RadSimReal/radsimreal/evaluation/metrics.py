"""Global-score-ranked single-class AP with one-to-one per-frame matching.

RadSimReal Sec.4 reports car AP at IoU 0.1/0.3/0.5 as PR area but does not
release evaluation code. Here the area is the integral of the monotonic
precision envelope at all recall changes (VOC all-points convention). This
explicit convention is a reproducibility choice, not a claimed exact match.
Coordinates must share a domain: the built-in dataset uses normalized RA image
XYXY boxes. This metric does NOT silently convert polar boxes to Cartesian BEV.
"""

import numpy as np
import torch


def box_iou(boxes1, boxes2):
    """Pairwise nonnegative XYXY area IoU, including empty and degenerate boxes."""
    boxes1 = torch.as_tensor(boxes1).reshape(-1, 4)
    boxes2 = torch.as_tensor(boxes2, device=boxes1.device, dtype=boxes1.dtype).reshape(-1, 4)
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(dim=-1)
    union = area1[:, None] + area2[None, :] - intersection
    return torch.where(union > 0, intersection / union.clamp_min(1e-12), torch.zeros_like(union))


def nms(boxes, scores, iou_threshold=0.3):
    boxes = torch.as_tensor(boxes).reshape(-1, 4)
    scores = torch.as_tensor(scores, device=boxes.device).reshape(-1)
    if scores.numel() != boxes.shape[0] or not 0 <= iou_threshold <= 1:
        raise ValueError("invalid NMS input")
    ordering = torch.argsort(scores, descending=True, stable=True)
    kept = []
    while ordering.numel():
        index = ordering[0]
        kept.append(index)
        if ordering.numel() == 1:
            break
        remaining = ordering[1:]
        ordering = remaining[box_iou(boxes[index].reshape(1, 4), boxes[remaining])[0] <= iou_threshold]
    return torch.stack(kept).to(dtype=torch.long) if kept else torch.empty(0, dtype=torch.long, device=boxes.device)


def _numpy(value, shape):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(shape)


def average_precision(predictions, targets, iou_threshold=0.5, matching_policy="best_unmatched"):
    """Return AP, PR arrays and counts. AP=None for datasets with zero GT cars.

    predictions: one {'boxes':Nx4,'scores':N} mapping per image.
    targets: one Nx4 array/tensor or {'boxes':Nx4} mapping per image.
    Duplicate predictions cannot claim the same GT. Tied scores preserve input
    order. The legacy default maximizes overlap over unmatched GT. Explicit
    best_overall maximizes over all GT and rejects an already matched maximum,
    matching RADDet's released getTruePositive convention.
    """
    if len(predictions) != len(targets) or not 0 <= iou_threshold <= 1:
        raise ValueError("predictions/targets length or IoU threshold invalid")
    if matching_policy not in ("best_unmatched","best_overall"):
        raise ValueError("matching_policy must be best_unmatched or best_overall")
    truths = [_numpy(t["boxes"] if isinstance(t, dict) else t, (-1, 4)) for t in targets]
    if any(not np.isfinite(t).all() for t in truths):
        raise ValueError("targets contain nonfinite values")
    detections = []
    for frame, prediction in enumerate(predictions):
        boxes = _numpy(prediction["boxes"], (-1, 4))
        scores = _numpy(prediction["scores"], (-1,))
        if len(boxes) != len(scores) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError("invalid prediction boxes/scores")
        detections.extend((float(score), frame, box) for box, score in zip(boxes, scores))
    detections.sort(key=lambda item: -item[0])
    matched = [np.zeros(len(t), dtype=bool) for t in truths]
    tp = np.zeros(len(detections), dtype=float)
    fp = np.zeros(len(detections), dtype=float)
    for i, (_, frame, box) in enumerate(detections):
        candidates = np.flatnonzero(~matched[frame]) if matching_policy == "best_unmatched" else np.arange(len(truths[frame]))
        if len(candidates) == 0:
            fp[i] = 1
            continue
        overlaps = box_iou(torch.as_tensor(box).reshape(1, 4), torch.as_tensor(truths[frame][candidates]))[0].numpy()
        best = int(np.argmax(overlaps))
        if overlaps[best] >= iou_threshold and not matched[frame][candidates[best]]:
            tp[i] = 1
            matched[frame][candidates[best]] = True
        else:
            fp[i] = 1
    n_gt = sum(len(t) for t in truths)
    cumulative_tp, cumulative_fp = np.cumsum(tp), np.cumsum(fp)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    recall = cumulative_tp / max(n_gt, 1)
    ap = None
    if n_gt:
        r = np.concatenate(([0.0], recall, [1.0]))
        p = np.concatenate(([0.0], precision, [0.0]))
        p = np.maximum.accumulate(p[::-1])[::-1]
        changes = np.flatnonzero(r[1:] != r[:-1]) + 1
        ap = float(np.sum((r[changes] - r[changes - 1]) * p[changes]))
    return {
        "ap": ap, "precision": precision.tolist(), "recall": recall.tolist(),
        "num_ground_truth": n_gt, "num_predictions": len(detections),
        "true_positives": int(tp.sum()), "false_positives": int(fp.sum()),
        "iou_threshold": float(iou_threshold),
        "matching_policy": matching_policy,
    }


def evaluate_predictions(predictions, targets, iou_thresholds=(0.1, 0.3, 0.5)):
    curves = {f"{t:g}": average_precision(predictions, targets, t) for t in iou_thresholds}
    return {
        **{f"AP@{t}": result["ap"] for t, result in curves.items()},
        "num_images": len(targets), "curves": curves,
        "class": "car", "box_domain": "normalized_range_azimuth_image",
        "metric_domain": "normalized_RA_bin_area_not_Cartesian_BEV",
        "ap_convention": "all_points_monotonic_precision_envelope",
    }
