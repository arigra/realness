"""Explicit chosen center detector losses, not published RadSimReal losses.

The paper leaves the range-azimuth box heads and loss unspecified. This baseline
uses a Gaussian center heatmap, focal loss (alpha=2,beta=4), and center-masked L1
size/offset terms. An exact center collision keeps the smaller box and reports a
collision count; no model can represent two centers in one cell with this head.
"""

import math
import torch


def build_targets(boxes, height: int, width: int, device=None, dtype=torch.float32):
    if height < 1 or width < 1:
        raise ValueError("target dimensions must be positive")
    batch_size = len(boxes)
    heatmap = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
    offset = torch.zeros(batch_size, 2, height, width, device=device, dtype=dtype)
    size = torch.zeros_like(offset)
    mask = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
    collisions = 0
    for b, frame_boxes in enumerate(boxes):
        frame_boxes = torch.as_tensor(frame_boxes, device=device, dtype=dtype).reshape(-1, 4)
        if frame_boxes.numel() and (
            not torch.isfinite(frame_boxes).all() or (frame_boxes < 0).any()
            or (frame_boxes > 1).any()
            or (frame_boxes[:, 2:] <= frame_boxes[:, :2]).any()
        ):
            raise ValueError("boxes must be finite nondegenerate normalized XYXY")
        areas = (frame_boxes[:, 2] - frame_boxes[:, 0]) * (frame_boxes[:, 3] - frame_boxes[:, 1])
        # Descending area makes smaller collided targets overwrite larger ones.
        for box in frame_boxes[torch.argsort(areas, descending=True, stable=True)]:
            wh = box[2:] - box[:2]
            center = (box[2:] + box[:2]) * 0.5
            px = float(center[0]) * width
            py = float(center[1]) * height
            ix, iy = min(int(px), width - 1), min(int(py), height - 1)
            sx, sy = max(float(wh[0]) * width / 6, 1.0), max(float(wh[1]) * height / 6, 1.0)
            rx, ry = math.ceil(3 * sx), math.ceil(3 * sy)
            x0, x1 = max(0, ix - rx), min(width, ix + rx + 1)
            y0, y1 = max(0, iy - ry), min(height, iy + ry + 1)
            xs = torch.arange(x0, x1, device=device, dtype=dtype) - ix
            ys = torch.arange(y0, y1, device=device, dtype=dtype) - iy
            gaussian = torch.exp(-0.5 * ((ys[:, None] / sy) ** 2 + (xs[None, :] / sx) ** 2))
            heatmap[b, 0, y0:y1, x0:x1] = torch.maximum(heatmap[b, 0, y0:y1, x0:x1], gaussian)
            collisions += int(mask[b, 0, iy, ix].item() != 0)
            mask[b, 0, iy, ix] = 1
            offset[b, :, iy, ix] = torch.tensor((px - ix, py - iy), device=device, dtype=dtype)
            size[b, :, iy, ix] = wh
    return {"heatmap": heatmap, "offset": offset, "size": size, "mask": mask, "collisions": collisions}


def detection_loss(outputs, boxes, size_weight: float = 5.0, offset_weight: float = 1.0):
    logits = outputs["heatmap"]
    targets = build_targets(boxes, *logits.shape[-2:], device=logits.device, dtype=logits.dtype)
    probability = logits.sigmoid().clamp(1e-6, 1 - 1e-6)
    truth = targets["heatmap"]
    positive = truth.eq(1).to(logits.dtype)
    negative = truth.lt(1).to(logits.dtype)
    count = positive.sum().clamp_min(1)
    heatmap_loss = -(
        positive * (1 - probability).pow(2) * probability.log()
        + negative * (1 - truth).pow(4) * probability.pow(2) * (1 - probability).log()
    ).sum() / count
    mask = targets["mask"]
    offset_loss = ((outputs["offset"] - targets["offset"]).abs() * mask).sum() / count
    size_loss = ((outputs["size"] - targets["size"]).abs() * mask).sum() / count
    box_loss = offset_weight * offset_loss + size_weight * size_loss
    return {
        "total": heatmap_loss + box_loss,
        "heatmap": heatmap_loss,
        "box": box_loss,
        "offset": offset_loss,
        "size": size_loss,
        "collisions": torch.as_tensor(targets["collisions"], device=logits.device),
    }
