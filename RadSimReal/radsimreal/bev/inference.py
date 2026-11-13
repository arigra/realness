"""Physical Cartesian box decoding and suppression, including oriented boxes."""
import numpy as np
import torch
from ..models.inference import decode_detections
from ..evaluation.metrics import nms
from .geometry import rotated_iou, oriented_enclosures
from .losses import decode_anchor_parameters


def rotated_nms(boxes, scores, iou_threshold=.1):
    boxes = torch.as_tensor(boxes)
    scores = torch.as_tensor(scores,device=boxes.device)
    if boxes.ndim != 2 or boxes.shape[1] != 5 or scores.shape != (len(boxes),) or not 0 <= iou_threshold <= 1:
        raise ValueError("Invalid rotated NMS input")
    order = torch.argsort(scores,descending=True,stable=True).tolist()
    kept = []
    values = boxes.detach().cpu().numpy()
    while order:
        i = order.pop(0)
        kept.append(i)
        if order:
            overlap = rotated_iou(values[i:i+1],values[order])[0]
            order = [j for j,value in zip(order,overlap) if value <= iou_threshold]
    return torch.tensor(kept,device=boxes.device,dtype=torch.long)


def _probabilistic_uncertainty(outputs,batch,indices,grid,centers,sizes):
    """Predicted parameter scales and explicit first-order physical propagation.

    Dong's six sigmas parameterize the offset/log-size/cos/sin regression loss.
    A delta-method transformation is useful for consumers but is not a proven
    calibrated Gaussian covariance. For axis-aligned training the two angular
    scales are unsupervised and are masked as such; yaw variance is undefined.
    """
    raw = outputs["raw"][batch].reshape(-1,13)[indices]
    variance = raw[:,7:13].clamp(-8,8).exp()
    anchors = outputs["anchors"][indices % len(outputs["anchors"])]
    span = torch.tensor([grid.x_max-grid.x_min,grid.y_max-grid.y_min],device=raw.device,dtype=torch.float64)
    anchor_physical = anchors.double()*span
    size_physical = sizes[batch].reshape(-1,2)[indices].double()*span
    center_physical = centers[batch].reshape(-1,2)[indices].double()*span+span.new_tensor([grid.x_min,grid.y_min])
    physical_variance = torch.cat([variance[:,:2].double()*anchor_physical.square(),
                                   variance[:,2:4].double()*size_physical.square()],-1)
    if not torch.isfinite(variance).all() or not torch.isfinite(physical_variance).all():
        raise FloatingPointError("Predicted localization uncertainty exceeds finite physical units")
    cosine,sine = raw[:,5].double(),raw[:,6].double()
    norm_squared = cosine.square()+sine.square()
    angle_variance = (sine.square()*variance[:,4].double()+cosine.square()*variance[:,5].double())/norm_squared.square().clamp_min(1e-30)
    supervised = bool(outputs.get("oriented",False))
    valid = (norm_squared > 1e-8) & torch.isfinite(angle_variance) & supervised
    angle_list = [float(v) if bool(ok) else None for v,ok in zip(angle_variance.detach().cpu(),valid.detach().cpu())]
    return {"localization_variance":variance,
            "localization_variance_parameters":["dx_over_anchor_width","dy_over_anchor_height","log_width_ratio","log_height_ratio","cos_yaw_offset","sin_yaw_offset"],
            "localization_variance_supervision_mask":[True,True,True,True,supervised,supervised],
            "box_parameter_variance_m2":physical_variance,
            "box_parameter_variance_order":["center_x","center_y","width","height"],
            "unclipped_box_parameters_m":torch.cat([center_physical,size_physical],-1),
            "orientation_variance_rad2":angle_list,"orientation_uncertainty_valid":valid,
            "uncertainty_propagation":"first-order delta method; independent regression scales; yaw uses atan2 Jacobian",
            "uncertainty_scope":"decoded Cartesian box parameters, with no clipping to the observation grid",
            "uncertainty_calibrated_on_real_data":False}


def decode_presence(outputs, grid, score_threshold=.2, nms_threshold=.3, top_k=100, pre_nms=4000):
    """Every pixel with presence >= threshold proposes the box its regression maps carry; score =
    its presence probability; then NMS. (Box read-out is our choice; Zhang/RadSimReal do not state it.)"""
    probability = outputs["presence"][:, 0].sigmoid()
    b, h, w = probability.shape
    scale = outputs["box_scale"]
    px = ((torch.arange(w, device=probability.device, dtype=probability.dtype) + .5) / w)[None, :].expand(h, w).reshape(-1)
    py = ((torch.arange(h, device=probability.device, dtype=probability.dtype) + .5) / h)[:, None].expand(h, w).reshape(-1)
    result = []
    for i in range(b):
        score = probability[i].reshape(-1)
        selection = torch.where(score >= score_threshold)[0]
        selection = selection[torch.argsort(score[selection], descending=True, stable=True)[:pre_nms]]
        reg = outputs["box"][i].reshape(4, -1)[:, selection] * scale
        cx, cy = px[selection] + reg[0] / w, py[selection] + reg[1] / h
        bw, bh = reg[2].clamp_min(1e-3) / w, reg[3].clamp_min(1e-3) / h
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)
        s = score[selection]
        kept = nms(boxes, s, nms_threshold)[:top_k]
        result.append({"boxes": grid.denormalize_boxes(boxes[kept]), "scores": s[kept],
                       "box_domain": "cartesian_bev_metres_axis_aligned",
                       "score_definition": "sigmoid of per-pixel presence logit", "prediction_clipping": "none"})
    return result


def decode_bev(outputs, grid, score_threshold=.2, nms_threshold=.3, top_k=100):
    if not 0 <= score_threshold <= 1 or not 0 <= nms_threshold <= 1 or top_k < 1:
        raise ValueError("Invalid detection cutoff or candidate limit")
    if outputs["head_type"] == "presence":
        return decode_presence(outputs,grid,score_threshold,nms_threshold,top_k)
    if outputs["head_type"] == "center":
        result = decode_detections(outputs,score_threshold,nms_threshold,top_k,clip_boxes=False)
        return [{**p,"boxes":grid.denormalize_boxes(p["boxes"]),"box_domain":"cartesian_bev_metres_axis_aligned",
                 "score_definition":"sigmoid of center heatmap logit","prediction_clipping":"none"} for p in result]
    centers,sizes,confidence,angles = decode_anchor_parameters(outputs)
    result = []
    use_oriented = outputs["head_type"] == "probabilistic_anchor" and bool(outputs.get("oriented",False))
    for b in range(len(confidence)):
        score = confidence[b].reshape(-1)
        selection = torch.where(score >= score_threshold)[0]
        selection = selection[torch.argsort(score[selection],descending=True,stable=True)[:top_k]]
        center = centers[b].reshape(-1,2)[selection]
        wh = sizes[b].reshape(-1,2)[selection]
        score = score[selection]
        if use_oriented:
            physical_center = center*center.new_tensor([grid.x_max-grid.x_min,grid.y_max-grid.y_min])+center.new_tensor([grid.x_min,grid.y_min])
            physical_wh = wh*wh.new_tensor([grid.x_max-grid.x_min,grid.y_max-grid.y_min])
            oriented = torch.cat([physical_center,physical_wh,angles[b].reshape(-1)[selection,None]],-1)
            valid = torch.isfinite(oriented).all(-1) & (physical_wh > 0).all(-1)
            oriented,score = oriented[valid],score[valid]
            retained_indices = selection[valid]
            kept = rotated_nms(oriented,score,nms_threshold)
            oriented,score = oriented[kept],score[kept]
            enclosures = oriented_enclosures(oriented.detach().cpu().numpy())
            item = {"boxes":torch.as_tensor(enclosures,device=oriented.device,dtype=oriented.dtype),
                    "oriented_boxes":oriented,"scores":score,"box_domain":"cartesian_bev_metres_oriented"}
            item.update(_probabilistic_uncertainty(outputs,b,retained_indices[kept],grid,centers,sizes))
            result.append(item)
        else:
            boxes = torch.cat([center-wh*.5,center+wh*.5],-1)
            valid = torch.isfinite(boxes).all(-1) & (boxes[:,2:] > boxes[:,:2]).all(-1)
            boxes,score = boxes[valid],score[valid]
            kept = nms(boxes,score,nms_threshold)
            item = {"boxes":grid.denormalize_boxes(boxes[kept]),"scores":score[kept],
                    "box_domain":"cartesian_bev_metres_axis_aligned"}
            if outputs["head_type"] == "probabilistic_anchor":
                item.update(_probabilistic_uncertainty(outputs,b,selection[valid][kept],grid,centers,sizes))
            result.append(item)
        result[-1].update({"score_definition":"sigmoid of objectness logit","prediction_clipping":"none"})
    return result


@torch.no_grad()
def detect_bev(model, images, radar=None, score_threshold=.2, nms_threshold=.3, top_k=100):
    training = model.training
    model.eval()
    try:
        return decode_bev(model(images,radar=radar),model.grid,score_threshold,nms_threshold,top_k)
    finally:
        model.train(training)
