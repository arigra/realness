"""Which ground-truth cars does a checkpoint miss? Error analysis only; nothing here is tuned.

For every GT car: its best IoU with any prediction (score >= --score), the score of the best
prediction overlapping it at IoU >= 0.1, and its distance, azimuth, size and scene. For every
prediction with no GT overlap (IoU < 0.1): its score, distance and scene. Then recall per bin
and a figure.

  python scripts/diagnose_detections.py --checkpoint runs/step1/unet_real/best.pt --split test \
      --out runs/step1/unet_real/diag/test
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from radsimreal.bev.training import load_bev_checkpoint, collate_bev
from radsimreal.bev.inference import detect_bev
from radsimreal.datasets.bev_dataset import BEVDataset
from radsimreal.evaluation.metrics import box_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/raddet_prepared_v")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--score", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, payload = load_bev_checkpoint(args.checkpoint, args.device)
    ds = BEVDataset(args.data, args.split, normalization=payload["normalization"], include_radar=False, grid=model.grid)
    scene = {r["id"]: r["sequence"] for r in ds.samples}
    gts, fps, frame_rows = [], [], []
    for batch in DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_bev):
        preds = detect_bev(model, batch["image"].to(args.device), None, args.score, 0.3, 100)
        for p, gt, fid, img in zip(preds, batch["boxes_physical"], batch["id"], batch["image"]):
            pb, ps = p["boxes"].cpu().double(), p["scores"].cpu().double()
            gt = gt.double().reshape(-1, 4)
            iou = box_iou(gt, pb).numpy() if len(gt) and len(pb) else np.zeros((len(gt), len(pb)))
            frame_rows.append({"id": fid, "scene": scene[fid], "n_gt": len(gt), "n_pred": len(pb),
                               "mean_intensity": float(img[0][img[0] > 0].mean())})
            for g, b in enumerate(gt.numpy()):
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                hit = iou[g] >= 0.1
                gts.append({"id": fid, "scene": scene[fid], "dist": float(np.hypot(cx, cy)),
                            "az": float(np.degrees(np.arctan2(cx, cy))), "w": float(b[2] - b[0]), "h": float(b[3] - b[1]),
                            "best_iou": float(iou[g].max()) if len(pb) else 0.0,
                            "score_at_iou01": float(ps.numpy()[hit].max()) if hit.any() else 0.0})
            for k in range(len(pb)):
                if not len(gt) or iou[:, k].max() < 0.1:
                    b = pb[k].numpy()
                    fps.append({"id": fid, "scene": scene[fid], "score": float(ps[k]),
                                "dist": float(np.hypot((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "gt_cars.json").write_text(json.dumps(gts)); (out / "unmatched_predictions.json").write_text(json.dumps(fps))
    (out / "frames.json").write_text(json.dumps(frame_rows))
    G = {k: np.array([g[k] for g in gts]) for k in ("dist", "az", "w", "h", "best_iou", "score_at_iou01")}
    sc = np.array([g["scene"] for g in gts])
    found01 = G["best_iou"] >= 0.1; found05 = G["best_iou"] >= 0.5
    summary = {"n_gt": len(gts), "recall_iou0.1_any_score": float(found01.mean()), "recall_iou0.5_any_score": float(found05.mean()),
               "recall_iou0.1_score>=0.5": float((G["score_at_iou01"] >= 0.5).mean()),
               "unmatched_predictions_score>=0.5": int(sum(f["score"] >= 0.5 for f in fps)), "by_scene": {}, "by_distance": {}, "by_azimuth": {}}
    for s in sorted(set(sc)):
        m = sc == s
        fr = [f for f in frame_rows if f["scene"] == s]
        summary["by_scene"][s] = {"cars": int(m.sum()), "frames": len(fr), "recall_iou0.1": float(found01[m].mean()),
                                  "recall_iou0.5": float(found05[m].mean()),
                                  "fp_score>=0.5_per_frame": sum(f["score"] >= 0.5 for f in fps if f["scene"] == s) / len(fr),
                                  "mean_input_intensity": float(np.mean([f["mean_intensity"] for f in fr]))}
    for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 55)]:
        m = (G["dist"] >= lo) & (G["dist"] < hi)
        summary["by_distance"][f"{lo}-{hi} m"] = {"cars": int(m.sum()), "recall_iou0.1": float(found01[m].mean()) if m.any() else None,
                                                   "recall_iou0.5": float(found05[m].mean()) if m.any() else None}
    for lo, hi in [(-90, -45), (-45, -15), (-15, 15), (15, 45), (45, 90)]:
        m = (G["az"] >= lo) & (G["az"] < hi)
        summary["by_azimuth"][f"{lo}..{hi} deg"] = {"cars": int(m.sum()), "recall_iou0.1": float(found01[m].mean()) if m.any() else None}
    hit = found01
    summary["box_size_m_found_vs_missed"] = {"found_median_wh": [float(np.median(G["w"][hit])), float(np.median(G["h"][hit]))],
                                             "missed_median_wh": [float(np.median(G["w"][~hit])), float(np.median(G["h"][~hit]))] if (~hit).any() else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    bins = np.arange(0, 52, 4)
    for s in sorted(set(sc)):
        m = sc == s
        idx = np.digitize(G["dist"][m], bins)
        rec = [found01[m][idx == i].mean() if (idx == i).sum() >= 10 else np.nan for i in range(1, len(bins))]
        ax[0].plot(bins[:-1] + 2, rec, marker="o", label=s)
    ax[0].set(xlabel="car distance (m)", ylabel="recall at IoU 0.1 (any score)", ylim=(0, 1.02), title="Found cars vs distance")
    ax[0].legend()
    ax[1].scatter(G["az"][found01], G["dist"][found01], s=3, c="tab:green", label="found")
    ax[1].scatter(G["az"][~found01], G["dist"][~found01], s=3, c="tab:red", label="missed")
    ax[1].set(xlabel="azimuth (deg)", ylabel="distance (m)", title="Where the missed cars are"); ax[1].legend()
    ax[2].hist([G["best_iou"][found01]], bins=np.linspace(0.1, 1, 19), color="tab:green")
    ax[2].set(xlabel="best IoU of found cars", ylabel="cars", title="Box overlap of found cars")
    fig.suptitle(f"{Path(args.checkpoint).parent.name} on {args.split}: {len(gts)} cars")
    fig.savefig(out / "misses.png", dpi=100)
    print("wrote", out)


if __name__ == "__main__":
    main()
