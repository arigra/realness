"""Class-car AP measured in physical Cartesian coordinates, with declared labels."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..evaluation.metrics import average_precision
from .geometry import rotated_iou
from .inference import detect_bev


def _np(value, dimensions):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value,dtype=np.float64).reshape(-1,dimensions)


def rotated_average_precision(predictions, targets, iou_threshold=.5,matching_policy="best_overall"):
    if len(predictions) != len(targets) or not 0 <= iou_threshold <= 1:
        raise ValueError("Invalid prediction count or IoU threshold")
    if matching_policy not in ("best_overall","best_unmatched"):
        raise ValueError("Unknown matching policy")
    if any(isinstance(t,dict) and "oriented_boxes" not in t for t in targets):
        raise ValueError("Oriented AP requires explicit oriented_boxes targets, not axis-aligned boxes")
    ground_truth = [_np(t["oriented_boxes"] if isinstance(t,dict) else t,5) for t in targets]
    detections = []
    for frame,p in enumerate(predictions):
        boxes = _np(p["oriented_boxes"],5)
        scores = np.asarray(p["scores"].detach().cpu() if isinstance(p["scores"],torch.Tensor) else p["scores"],dtype=float).reshape(-1)
        if len(boxes) != len(scores) or not np.isfinite(scores).all():
            raise ValueError("Invalid oriented prediction scores")
        # rotated_iou validates finite boxes and positive extents.
        if len(boxes):
            rotated_iou(boxes,boxes[:1])
        detections.extend((float(score),frame,box) for score,box in zip(scores,boxes))
    for boxes in ground_truth:
        if len(boxes):
            rotated_iou(boxes,boxes[:1])
    detections.sort(key=lambda row:-row[0])
    matched = [np.zeros(len(t),dtype=bool) for t in ground_truth]
    tp,fp = np.zeros(len(detections)),np.zeros(len(detections))
    for i,(_,frame,box) in enumerate(detections):
        available = np.flatnonzero(~matched[frame]) if matching_policy == "best_unmatched" else np.arange(len(ground_truth[frame]))
        if not len(available):
            fp[i] = 1
            continue
        overlaps = rotated_iou(box[None],ground_truth[frame][available])[0]
        best = int(overlaps.argmax())
        if overlaps[best] >= iou_threshold and not matched[frame][available[best]]:
            matched[frame][available[best]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    n_gt = sum(len(t) for t in ground_truth)
    precision = np.cumsum(tp)/np.maximum(np.cumsum(tp+fp),1)
    recall = np.cumsum(tp)/max(n_gt,1)
    ap = None
    if n_gt:
        r,p = np.r_[0,recall,1],np.r_[0,precision,0]
        p = np.maximum.accumulate(p[::-1])[::-1]
        changes = np.flatnonzero(r[1:] != r[:-1])+1
        ap = float(np.sum((r[changes]-r[changes-1])*p[changes]))
    return {"ap":ap,"precision":precision.tolist(),"recall":recall.tolist(),
            "num_ground_truth":n_gt,"num_predictions":len(detections),
            "true_positives":int(tp.sum()),"false_positives":int(fp.sum()),"iou_threshold":iou_threshold,
            "matching_policy":matching_policy}


def frame_mean_average_precision(predictions,targets,iou_threshold=.5,oriented=False,matching_policy="best_overall"):
    """Released RADDet class-specific aggregation, extended explicitly to yaw.

    The upstream mAP2D appends per-image AP only for classes with GT in that
    image; evaluate.py then averages each class list. Thus GT-empty car frames
    are excluded even if they contain false car predictions. If no frame has
    car GT, the upstream evaluation returns class AP0. These rules are preserved
    here and reported because they differ materially from global PR/AP.
    """
    if len(predictions) != len(targets) or not 0 <= iou_threshold <= 1:
        raise ValueError("Prediction/target image counts differ")
    function = rotated_average_precision if oriented else average_precision
    dimension = 5 if oriented else 4
    values,rows = [],[]
    ignored_predictions = 0
    for prediction,truth in zip(predictions,targets):
        key = "oriented_boxes" if oriented else "boxes"
        if isinstance(truth,dict) and key not in truth:
            raise ValueError(f"Frame AP requires explicit {key} target coordinates")
        boxes = _np(truth[key] if isinstance(truth,dict) else truth,dimension)
        result = function([prediction],[boxes],iou_threshold,matching_policy=matching_policy)
        if not len(boxes):
            values.append(None)
            ignored_predictions += result["num_predictions"]
            continue
        values.append(result["ap"])
        rows.append(result)
    return {"ap":float(np.mean([r["ap"] for r in rows])) if rows else 0.,
            "precision":[],"recall":[],"per_frame_ap":values,
            "num_ground_truth":sum(r["num_ground_truth"] for r in rows),
            "num_predictions":sum(r["num_predictions"] for r in rows),
            "true_positives":sum(r["true_positives"] for r in rows),
            "false_positives":sum(r["false_positives"] for r in rows),
            "num_frames_scored":len(rows),"num_gt_empty_frames_excluded":len(targets)-len(rows),
            "num_predictions_ignored_in_gt_empty_frames":ignored_predictions,
            "iou_threshold":iou_threshold,"matching_policy":matching_policy,
            "zero_gt_policy":"exclude GT-empty frames; return class AP0 when no positive frame exists",
            "precision_recall_scope":"no single PR curve exists for a mean of per-frame AP values"}


def evaluate_predictions_bev(predictions, targets, iou_thresholds=(.1,.3,.5), oriented=False, label_domain="unspecified_cartesian_labels",
                             matching_policy="best_overall",ap_aggregation="global_pr"):
    for p in predictions:
        domain = p.get("box_domain", "cartesian_bev_metres")
        if not domain.startswith("cartesian_bev_metres"):
            raise ValueError("BEV evaluator rejects non-Cartesian prediction domains")
    if matching_policy not in ("best_overall","best_unmatched") or ap_aggregation not in ("global_pr","frame_mean"):
        raise ValueError("Unknown AP matching or aggregation convention")
    function = rotated_average_precision if oriented else average_precision
    curves = {f"{t:g}":(function(predictions,targets,t,matching_policy=matching_policy) if ap_aggregation == "global_pr" else
                        frame_mean_average_precision(predictions,targets,t,oriented,matching_policy)) for t in iou_thresholds}
    return {**{f"AP@{t}":r["ap"] for t,r in curves.items()},"curves":curves,
            "num_images":len(targets),"class":"car","label_domain":label_domain,
            "metric_domain":"cartesian_bev_metres_oriented" if oriented else "cartesian_bev_metres_axis_aligned",
            "ap_convention":"all_points_monotonic_precision_envelope","matching_policy":matching_policy,
            "ap_aggregation":ap_aggregation,"score_tie_policy":"stable_input_order",
            "zero_gt_policy":"undefined class AP without GT; include GT-empty-frame detections as false positives" if ap_aggregation == "global_pr" else
                             "exclude GT-empty frames; class AP0 if no positive frame exists (RADDet released per-class evaluator)",
            "radsimreal_evaluator_parity":"unverified; its exact evaluator was not released",
            "published_result_reproduced":False}


def evaluate_bev(checkpoint, dataset, batch_size=4, device="cpu", score_threshold=.05,
                 nms_threshold=.3, top_k=100, output=None, num_threads=1,
                 allow_label_domain_change=False,allow_sensor_change=False,
                 matching_policy="best_overall",ap_aggregation="global_pr"):
    from .training import load_bev_checkpoint, collate_bev,compare_sensor_sampling,enforce_sensor_sampling
    if batch_size < 1 or num_threads < 1:
        raise ValueError("Batch size and threads must be positive")
    torch.set_num_threads(num_threads)
    model,payload = load_bev_checkpoint(checkpoint,device)
    if not len(dataset):
        raise ValueError("Evaluation dataset is empty")
    if dataset.grid.to_dict() != model.grid.to_dict():
        raise ValueError("Evaluation grid differs from checkpoint grid")
    label_domain = str(getattr(dataset,"label_domain","unspecified_cartesian_labels"))
    training_domain = payload.get("label_domain",label_domain)
    if label_domain != training_domain and not allow_label_domain_change:
        raise ValueError("Evaluation label domain differs; explicitly acknowledge a cross-domain comparison")
    if dict(getattr(dataset,"normalization",{})) != payload.get("normalization",{}):
        raise ValueError("Dataset normalization differs from checkpoint; construct it with saved preprocessing")
    sensor_comparison = compare_sensor_sampling(payload.get("dataset_meta",{}),dataset)
    enforce_sensor_sampling(model,sensor_comparison,dataset,allow_sensor_change)
    predictions,targets,ids = [],[],[]
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_bev,num_workers=0)
    for batch in loader:
        image = batch["image"].to(device)
        radar = batch["radar"].to(device) if model.requires_radar else None
        detected = detect_bev(model,image,radar,score_threshold,nms_threshold,top_k)
        predictions.extend({k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in p.items()} for p in detected)
        if model.oriented:
            physical_targets = batch.get("oriented_boxes_physical",batch["oriented_boxes"])
            physical_boxes = batch.get("boxes_physical",batch["boxes"])
            for boxes,ob in zip(physical_boxes,physical_targets):
                if len(ob) != len(boxes):
                    raise ValueError("Oriented evaluation requires a genuine yaw label for every object")
            targets.extend(physical_targets)
        else:
            targets.extend(batch["boxes_physical"] if "boxes_physical" in batch else
                           [model.grid.denormalize_boxes(boxes) for boxes in batch["boxes"]])
        ids.extend(batch["id"])
    result = evaluate_predictions_bev(predictions,targets,oriented=model.oriented,label_domain=label_domain,
                                     matching_policy=matching_policy,ap_aggregation=ap_aggregation)
    result.update({"checkpoint":str(checkpoint),"architecture":model.architecture,
                   "score_definition":"sigmoid of center heatmap logit" if model.head_type == "center" else "sigmoid of objectness logit",
                   "prediction_clipping":"none",
                   "training_label_domain":training_domain,"cross_label_domain_comparison":label_domain != training_domain,
                   "sensor_sampling_comparison":sensor_comparison,"sensor_change_explicitly_allowed":bool(allow_sensor_change),
                   "image_ids":ids,"score_threshold":score_threshold,"nms_threshold":nms_threshold,"top_k":top_k})
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    return result
