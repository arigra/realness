"""Target crops for the classification study (docs/notes/2026-11-05-classification.md).

A crop is the range-Doppler patch around one labelled object in one frame, centred on its
label, so the classifier never sees where in Doppler (how fast) or in range (how far) the
object is: classes differ in typical speed and a network would learn that shortcut instead
of the target's signature (Ari, 2026-11-05). Two more leaks are closed:
- the static radars' zero-Doppler ridge (CARRADA, RADDet: bin 32) shows inside the crop of a
  slow object and would give its speed away: those columns (+/-1 bin) are set to the crop's median;
- the level of the crop follows range and the radar's gain: each crop is standardised
  (minus its median, over its standard deviation).
Classes: CARRADA pedestrian / cyclist / car; RADDet's six merged to the simulator's three
models: person; bicycle + motorcycle; car + bus + truck. Label 0..2 = class 1..3.
"""
import pickle

import numpy as np
import torch

from src.data import load

K_R, K_D, ZERO = 12, 8, 32            # crop half-sizes (cells), zero-Doppler bin
GROUP = {"person": 1, "bicycle": 2, "motorcycle": 2, "car": 3, "bus": 3, "truck": 3}
CLASSES = ("person", "two-wheeler", "vehicle")


# How a crop is made (controls of docs/notes/2026-11-05-classification.md):
#   ridge: "constant" sets the zero-Doppler columns to the crop's median (the first version: the
#          flat stripe's position inside the crop gives the target's speed away), "noise" fills
#          them with values drawn from the crop's other cells, "keep" leaves them;
#   align: "label" centres the crop on the label, "peak" on the brightest cell within +/-3 cells
#          of it (target shape without the dataset's label-to-return offset);
#   erase: the crop is taken ERASE_ROWS rows away from the target in range, at the same Doppler:
#          the ridge and background are there, the target is not (a classifier must be at chance).
#   kr, kd: half sizes (Phase 8 of docs/notes/2026-11-06-metric-experiments.md: smaller crops);
#   mask: None, or "component": keep only the connected cells around the target's strongest cell
#         that stand >= max(peak - 2, 1) crop-sd over the crop median; bg "noise" then replaces
#         everything outside the mask by N(0, 1) (background randomised);
#   ridge_only: the zero-Doppler columns kept, every other cell N(0, 1) (does the ridge alone tell
#         the class?).
MODE = {"ridge": "noise", "align": "label", "erase": False,    # "constant" leaked speed (results/crop_controls.json)
        "kr": K_R, "kd": K_D, "mask": None, "bg": "keep", "ridge_only": False}
ERASE_ROWS = 40


def crop(frame_db, r, d, n_doppler=64, mode=None, generator=None):
    """The standardised (2 K_R + 1, 2 K_D + 1) crop around (r, d) of one dB frame."""
    m = MODE | (mode or {})
    R = frame_db.shape[0]
    ri, di = int(round(r)), int(round(d))
    if m["align"] == "peak":
        rr = torch.arange(ri - 3, ri + 4).clamp(0, R - 1)
        dd = torch.remainder(torch.arange(di - 3, di + 4), n_doppler)
        k = int(frame_db[rr][:, dd].float().argmax())
        ri, di = int(rr[k // 7]), int(dd[k % 7])
    if m["erase"]:
        ri = ri + ERASE_ROWS if ri + ERASE_ROWS + K_R < R else ri - ERASE_ROWS
    kr, kd = m["kr"], m["kd"]
    rows = torch.arange(ri - kr, ri + kr + 1).clamp(0, R - 1)
    bins = torch.remainder(torch.arange(di - kd, di + kd + 1), n_doppler)
    c = frame_db[rows][:, bins].float().clone()
    ridge = (torch.remainder(bins - ZERO + n_doppler // 2, n_doppler) - n_doppler // 2).abs() <= 1
    if bool(ridge.any()) and m["ridge"] != "keep":
        rest = c[:, ~ridge]
        if m["ridge"] == "constant":
            c[:, ridge] = rest.median()
        else:                     # values of the crop's other cells, drawn at random
            pick = torch.randint(rest.numel(), (int(ridge.sum()) * c.shape[0],), generator=generator)
            c[:, ridge] = rest.flatten()[pick].view(c.shape[0], -1)
    c = (c - c.median()) / (c.std() + 1e-6)
    if m["ridge_only"]:
        out = torch.randn(c.shape, generator=generator)
        out[:, ridge] = c[:, ridge]
        return out
    if m["mask"] == "component":
        from scipy import ndimage
        w = c[kr - 1: kr + 2, kd - 1: kd + 2]
        k = int(w.argmax())
        pr, pd = kr - 1 + k // 3, kd - 1 + k % 3
        level = max(float(c[pr, pd]) - 2.0, 1.0)
        lab, _ = ndimage.label((c >= level).numpy())
        keep = torch.from_numpy(lab == lab[pr, pd]) if lab[pr, pd] > 0 else torch.zeros(c.shape, dtype=torch.bool)
        if m["bg"] == "noise":
            c = torch.where(keep, c, torch.randn(c.shape, generator=generator))
        else:
            c = torch.where(keep, c, torch.zeros_like(c))
    elif m["bg"] == "noise":
        raise ValueError("bg noise needs a mask")
    return c


def speed_range(r, d, range_res, vel_res):
    """The shortcut the crops hide: |radial speed| (m/s) and range (m) of a label."""
    return [abs((d - ZERO) * vel_res), r * range_res]


def from_frames(frames, range_res, vel_res, mode=None):
    """frames: iterable of (dB frame (R, D), [(range_bin, doppler_bin, class 1..3)]) ->
    crops (N, 1, h, w), classes (N,) 0..2, shortcut features (N, 2)."""
    xs, ys, fs = [], [], []
    g = torch.Generator().manual_seed(0)
    for x, objs in frames:
        for r, d, c in objs:
            xs.append(crop(x, r, d, x.shape[1], mode, g))
            ys.append(c - 1)
            fs.append(speed_range(r, d, range_res, vel_res))
    if not xs:
        return torch.zeros(0, 1, 2 * K_R + 1, 2 * K_D + 1), torch.zeros(0, dtype=torch.long), torch.zeros(0, 2)
    return torch.stack(xs)[:, None], torch.tensor(ys), torch.tensor(fs, dtype=torch.float)


def real_frames(dataset, cache, split, limit=None):
    """(frame, objects) of a real split; classes from the cache (CARRADA) or RADDet's raw labels."""
    items = load(cache, split, limit=limit)
    labels = torch.load(f"{cache}/{split}_labels.pt")
    if dataset == "raddet":
        from src import raddet
        part = {o: {f: p for p, f in raddet.frames(o)} for o in ("train", "test")}
    for i, it in enumerate(items):
        meta = labels["meta"][i]
        for t in range(it["x"].shape[0]):
            if dataset == "carrada":
                objs = [(float(it["traj"][m, t, 0]), float(it["traj"][m, t, 1]), int(labels["cls"][i, m]))
                        for m in range(it["traj"].shape[0]) if it["present"][m, t]]
            else:                       # slot m of frame t = the frame's m-th label (src/raddet.labels)
                o, f = meta["samples"][t]
                raw = raddet.labels(o, part[o][f], f)
                objs = [(float(it["traj"][m, t, 0]), float(it["traj"][m, t, 1]), GROUP[raw[m]["cls"]])
                        for m in range(min(len(raw), it["traj"].shape[0])) if it["present"][m, t]]
            yield it["x"][t], objs


def sim_frames(sequences):
    """(frame, objects) of simulated sequences (gen_sequence outputs; label 'cls' = model 1..3)."""
    for o in sequences:
        for t, fl in enumerate(o["labels"]):
            yield o["x"][t], [(lab["range_bin"], lab["doppler_bin"], lab["cls"]) for lab in fl]
