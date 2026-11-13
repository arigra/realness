"""Model-specific Cartesian training losses with explicit anchor assignment.

Center U-Net uses the documented reconstruction focal/L1 recipe. RADDet uses
released Cartesian YOLO square-root box loss, focal objectness and class BCE.
Probabilistic uses Dong2020 Eq.(1)-(2): exp(-log variance)*smoothL1 + log variance,
localization weight100, binary focal alpha.25 gamma2. Matching rules are stated
engineering choices because exact RadSimReal adaptations were not released.
"""
import math
import torch
from torch.nn import functional as F
from ..models.losses import detection_loss
from .geometry import BEVGrid, rotated_iou


def _validate_boxes(boxes, device, dtype):
    result = []
    for value in boxes:
        b = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1,4)
        if len(b) and (not torch.isfinite(b).all() or (b < 0).any() or (b > 1).any() or (b[:,2:] <= b[:,:2]).any()):
            raise ValueError("Targets require finite nondegenerate normalized Cartesian XYXY")
        result.append(b)
    return result


def _grids(raw):
    h, w = raw.shape[1:3]
    rows, columns = torch.meshgrid(torch.arange(h, device=raw.device, dtype=raw.dtype),
                                   torch.arange(w, device=raw.device, dtype=raw.dtype), indexing="ij")
    return torch.stack([columns, rows], -1)[None,:,:,None,:], raw.new_tensor([w,h])


def decode_anchor_parameters(outputs):
    raw, anchors = outputs["raw"], outputs["anchors"]
    cells, dimensions = _grids(raw)
    if outputs["head_type"] == "raddet_anchor":
        center = (raw[...,:2].sigmoid() + cells) / dimensions
        # Released source clamps log sizes at +1e-12, disallowing shrinkage.
        # Use a disclosed bounded signed exponent instead to permit fitted anchors.
        log_wh = raw[...,2:4].clamp(1e-12,1e12) if outputs.get("upstream_size_clamp",False) else raw[...,2:4].clamp(-12,12)
        wh = log_wh.exp() * anchors
        # Released RADDet ranks/thresholds objectness alone; the car-class logit
        # is still supervised by its separate classification loss.
        confidence = raw[...,4].sigmoid()
        return center, wh, confidence, None
    anchor_center = (cells + .5) / dimensions
    center = anchor_center + raw[...,1:3] * anchors
    wh = raw[...,3:5].clamp(-12,12).exp() * anchors
    angles = outputs["anchor_angles"] + torch.atan2(raw[...,6], raw[...,5])
    return center, wh, raw[...,0].sigmoid(), angles


def _size_iou(wh, anchors):
    intersection = torch.minimum(wh[None], anchors).prod(-1)
    return intersection / (wh.prod() + anchors.prod(-1) - intersection).clamp_min(1e-12)


def anchor_targets(outputs, boxes, oriented_boxes=None, grid=None):
    raw, anchors = outputs["raw"], outputs["anchors"]
    batch, height, width, n_anchors = raw.shape[:4]
    boxes = _validate_boxes(boxes, raw.device, raw.dtype)
    if len(boxes) != batch:
        raise ValueError("Target count does not match model batch")
    confidence = raw.new_zeros(batch,height,width,n_anchors)
    targets = raw.new_zeros(batch,height,width,n_anchors,6)
    wh_targets = raw.new_zeros(batch,height,width,n_anchors,2)
    xy_targets = raw.new_zeros(batch,height,width,n_anchors,2)
    assignments = []
    collisions = 0
    is_prob = outputs["head_type"] == "probabilistic_anchor"
    use_angle = is_prob and bool(outputs.get("oriented", False))
    grid = grid or BEVGrid()
    for b, frame in enumerate(boxes):
        oriented = None
        if use_angle:
            if oriented_boxes is None:
                raise ValueError("Oriented training requires genuine oriented labels for every object")
            oriented = torch.as_tensor(oriented_boxes[b], device=raw.device, dtype=raw.dtype).reshape(-1,5)
            if len(oriented) != len(frame) or (len(oriented) and (not torch.isfinite(oriented).all() or (oriented[:,2:4] <= 0).any())):
                raise ValueError("Oriented labels must align one-to-one with Cartesian targets")
        if len(frame):
            sizes = frame[:,2:] - frame[:,:2]
            order = torch.argsort(sizes.prod(-1), descending=True, stable=True)
        else:
            order = []
        for index in order:
            box = frame[index]
            center, wh = (box[:2]+box[2:])*.5, box[2:]-box[:2]
            yaw = None
            if use_angle:
                ob = oriented[index]
                center = raw.new_tensor([(ob[0]-grid.x_min)/(grid.x_max-grid.x_min),
                                         (ob[1]-grid.y_min)/(grid.y_max-grid.y_min)])
                wh = ob[2:4] / raw.new_tensor([grid.x_max-grid.x_min,grid.y_max-grid.y_min])
                yaw = ob[4]
            if ((center < 0) | (center > 1)).any():
                raise ValueError("Oriented target center is outside declared BEV grid")
            ix, iy = min(int(center[0]*width),width-1), min(int(center[1]*height),height-1)
            match = _size_iou(wh, anchors)
            if use_angle:
                # Exact physical rectangle overlap, including anisotropic pixels.
                cell_center = raw.new_tensor([grid.x_min+(ix+.5)/width*(grid.x_max-grid.x_min),
                                              grid.y_min+(iy+.5)/height*(grid.y_max-grid.y_min)])
                anchor_wh = anchors*raw.new_tensor([grid.x_max-grid.x_min,grid.y_max-grid.y_min])
                anchor_boxes = torch.cat([cell_center.expand(len(anchors),2),anchor_wh,
                                          outputs["anchor_angles"][:,None]],-1)
                match = raw.new_tensor(rotated_iou(ob.detach().cpu().numpy()[None],anchor_boxes.detach().cpu().numpy())[0])
            anchor = int(match.argmax())
            collisions += int(confidence[b,iy,ix,anchor] != 0)
            confidence[b,iy,ix,anchor] = 1
            xy_targets[b,iy,ix,anchor] = center
            wh_targets[b,iy,ix,anchor] = wh
            if is_prob:
                anchor_center = raw.new_tensor([(ix+.5)/width,(iy+.5)/height])
                targets[b,iy,ix,anchor,:2] = (center-anchor_center)/anchors[anchor]
                targets[b,iy,ix,anchor,2:4] = torch.log(wh/anchors[anchor])
                if use_angle:
                    delta = yaw - outputs["anchor_angles"][anchor]
                    targets[b,iy,ix,anchor,4:] = torch.stack([torch.cos(delta),torch.sin(delta)])
            assignments.append((b,iy,ix,anchor))
    return {"confidence":confidence,"regression":targets,"xy":xy_targets,"wh":wh_targets,
            "collisions":collisions,"angle_supervised":use_angle}


def presence_targets(outputs, boxes):
    """Per-pixel presence mask and box targets for the Zhang head.

    A pixel is positive when its centre lies inside a normalized XYXY box; where boxes overlap the
    smallest box owns the pixel. A box covering no pixel centre owns the pixel holding its centre, so
    every ground-truth car is supervised. Box target = (cx-px, cy-py, w, h) in pixels / box_scale.
    """
    logits = outputs["presence"]
    b, _, h, w = logits.shape
    device, dtype = logits.device, logits.dtype
    px = (torch.arange(w, device=device, dtype=dtype) + .5) / w
    py = (torch.arange(h, device=device, dtype=dtype) + .5) / h
    mask = torch.zeros(b, h, w, device=device, dtype=dtype)
    target = torch.zeros(b, 4, h, w, device=device, dtype=dtype)
    scale = outputs["box_scale"]
    for i, frame in enumerate(_validate_boxes(boxes, device, dtype)):
        if not len(frame):
            continue
        area = (frame[:, 2] - frame[:, 0]) * (frame[:, 3] - frame[:, 1])
        for k in torch.argsort(area, descending=True):  # smaller boxes written last own overlaps
            x0, y0, x1, y1 = frame[k]
            inside = ((py[:, None] >= y0) & (py[:, None] <= y1)) & ((px[None, :] >= x0) & (px[None, :] <= x1))
            if not inside.any():
                inside[min(int((y0 + y1) / 2 * h), h - 1), min(int((x0 + x1) / 2 * w), w - 1)] = True
            mask[i][inside] = 1
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            target[i, 0][inside] = ((cx - px[None, :]).expand(h, w) * w)[inside] / scale
            target[i, 1][inside] = ((cy - py[:, None]).expand(h, w) * h)[inside] / scale
            target[i, 2][inside] = (x1 - x0) * w / scale
            target[i, 3][inside] = (y1 - y0) * h / scale
    return mask, target


def presence_loss(outputs, boxes):
    """Zhang et al. 2020 Sec. 4.3.2: BCE + Dice on the presence map, plus MSE on regression; summed."""
    mask, target = presence_targets(outputs, boxes)
    logits = outputs["presence"][:, 0]
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    p = logits.sigmoid()
    dice = 1 - (2 * (p * mask).sum() + 1) / (p.sum() + mask.sum() + 1)
    positive = mask.bool()
    mse = ((outputs["box"] - target) ** 2).permute(0, 2, 3, 1)[positive].mean() if positive.any() else outputs["box"].sum() * 0
    seg = bce + dice
    return {"total": seg + mse, "heatmap": seg, "box": mse, "bce": bce, "dice": dice,
            "collisions": logits.new_tensor(0)}


def bev_detection_loss(outputs, boxes, oriented_boxes=None, grid=None):
    kind = outputs["head_type"]
    if kind == "center":
        return detection_loss(outputs, boxes)
    if kind == "presence":
        return presence_loss(outputs, boxes)
    targets = anchor_targets(outputs, boxes, oriented_boxes, grid)
    raw, positive = outputs["raw"], targets["confidence"]
    count = positive.sum().clamp_min(1)
    if kind == "raddet_anchor":
        center, wh, _, _ = decode_anchor_parameters(outputs)
        pixel_scale = raw.new_tensor([grid.width,grid.height]) if grid is not None else raw.new_tensor([512,256])
        box = ((((center-targets["xy"])*pixel_scale)**2 +
                ((wh*pixel_scale).clamp_min(1e-12).sqrt()-(targets["wh"]*pixel_scale).clamp_min(0).sqrt())**2).sum(-1)*positive).sum()/len(raw)
        probability = raw[...,4].sigmoid()
        bce = F.binary_cross_entropy_with_logits(raw[...,4],positive,reduction="none")
        negative_mask = torch.ones_like(positive)
        # Official confidence ignore rule excludes negative anchors overlapping GT.
        # Compute in normalized Cartesian coordinates, not in RA bins.
        from ..evaluation.metrics import box_iou
        prediction = torch.cat([center-wh*.5,center+wh*.5],-1).detach()
        for b, frame in enumerate(boxes):
            gt = torch.as_tensor(frame,device=raw.device,dtype=raw.dtype).reshape(-1,4)
            if len(gt):
                ious = box_iou(prediction[b].reshape(-1,4),gt).max(-1).values.reshape_as(positive[b])
                negative_mask[b] = (ious < .3).to(raw.dtype)
        object_loss = (((positive-probability)**2)*bce*(positive+.01*(1-positive)*negative_mask)).sum()/len(raw)
        class_loss = (F.binary_cross_entropy_with_logits(raw[...,5],torch.ones_like(positive),reduction="none")*positive).sum()/len(raw)
        total = .1*box+object_loss+class_loss
        return {"total":total,"heatmap":object_loss,"box":.1*box+class_loss,
                "objectness":object_loss,"classification":class_loss,"collisions":raw.new_tensor(targets["collisions"])}
    logits = raw[...,0]
    probability = logits.sigmoid()
    focal = F.binary_cross_entropy_with_logits(logits,positive,reduction="none")
    pt = torch.where(positive.bool(),probability,1-probability)
    alpha = torch.where(positive.bool(),.25,.75)
    object_loss = (alpha*(1-pt).pow(2)*focal).sum()/count
    regression = raw[...,1:7]
    log_variance = raw[...,7:13].clamp(-8,8)
    residual = F.smooth_l1_loss(regression,targets["regression"],reduction="none")
    dimension_mask = raw.new_ones(6)
    if not targets["angle_supervised"]:
        dimension_mask[4:] = 0
    localization = ((torch.exp(-log_variance)*residual+log_variance)*positive[...,None]*dimension_mask).sum()/count
    total = object_loss+100*localization
    return {"total":total,"heatmap":object_loss,"box":100*localization,
            "localization":localization,"collisions":raw.new_tensor(targets["collisions"])}
