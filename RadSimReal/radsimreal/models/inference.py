"""Heatmap peak decoding and class-car nonmaximum suppression."""

import torch
from torch.nn import functional as F
from ..evaluation.metrics import nms


def decode_detections(outputs, score_threshold=0.2, nms_threshold=0.3, top_k=100, clip_boxes=True):
    """Decode peaks; physical callers explicitly retain unbounded box extents."""
    if not 0 <= score_threshold <= 1 or not 0 <= nms_threshold <= 1 or top_k < 1:
        raise ValueError("invalid detection thresholds or top_k")
    probabilities = outputs["heatmap"].sigmoid()
    local_maximum = F.max_pool2d(probabilities, 3, stride=1, padding=1)
    result = []
    for b, heatmap in enumerate(probabilities[:, 0]):
        height, width = heatmap.shape
        candidates = (heatmap >= score_threshold) & (heatmap == local_maximum[b, 0])
        rows, cols = torch.where(candidates)
        scores = heatmap[rows, cols]
        if scores.numel() == 0:
            result.append({"boxes": heatmap.new_empty((0, 4)), "scores": heatmap.new_empty((0,))})
            continue
        ordering = torch.argsort(scores, descending=True, stable=True)[:top_k]
        rows, cols, scores = rows[ordering], cols[ordering], scores[ordering]
        offset = outputs["offset"][b, :, rows, cols].T
        size = outputs["size"][b, :, rows, cols].T
        center = torch.stack(((cols + offset[:, 0]) / width, (rows + offset[:, 1]) / height), dim=1)
        boxes = torch.cat((center - size * 0.5, center + size * 0.5), dim=1)
        if clip_boxes:
            boxes = boxes.clamp(0, 1)
        valid = (boxes[:, 2:] > boxes[:, :2]).all(dim=1)
        boxes, scores = boxes[valid], scores[valid]
        kept = nms(boxes, scores, nms_threshold)
        result.append({"boxes": boxes[kept], "scores": scores[kept]})
    return result


@torch.no_grad()
def detect(model, images, score_threshold=0.2, nms_threshold=0.3, top_k=100):
    """Run inference without modifying the caller's train/eval mode."""
    was_training = model.training
    model.eval()
    try:
        return decode_detections(model(images), score_threshold, nms_threshold, top_k)
    finally:
        model.train(was_training)
