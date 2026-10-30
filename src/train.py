"""Training the frame detector, and its detections on a real split.

Batches are single frames, each normalised by its own source's statistics.
Training stops when AP on real validation sequences stops improving (checked
every VAL_EVERY steps, PATIENCE checks without a gain, at most MAX_STEPS
steps), and the best weights are kept.
"""
import torch

from src.data import normalised
from src.detector import average_precision, detect, focal_loss, heatmap_targets

LR = 1e-3
VAL_EVERY, PATIENCE, MAX_STEPS = 250, 4, 6000
N_VAL = 32          # validation sequences, spread evenly over the real val split


def frame_batch(pool, n, device, generator):
    """n random frames from a pool, normalised, with their heatmap targets."""
    idx = torch.randint(len(pool), (n,), generator=generator).tolist()
    n_frames = pool[0]["x"].shape[0]
    frames = torch.randint(n_frames, (n,), generator=generator).tolist()
    x = torch.stack([normalised(pool[i], t) for i, t in zip(idx, frames)])[:, None]
    traj = torch.stack([pool[i]["traj"][:, t] for i, t in zip(idx, frames)])[:, :, None]
    present = torch.stack([pool[i]["present"][:, t] for i, t in zip(idx, frames)])[:, :, None]
    return x.to(device), heatmap_targets(traj.to(device), present.to(device),
                                         n_range=x.shape[-2], n_doppler=x.shape[-1])


@torch.no_grad()
def detections(model, pool, device):
    """Per frame of every sequence: the detections and the labelled objects."""
    model.eval()
    dets, tgts = [], []
    for item in pool:
        probs = [torch.sigmoid(model(normalised(item, t)[None, None].to(device)).float())[0, 0].cpu()
                 for t in range(item["x"].shape[0])]
        dets += detect(torch.stack(probs), threshold=0.05, top_k=10)
        for t in range(item["traj"].shape[1]):
            tgts.append(item["traj"][item["present"][:, t], t].float())
    model.train()
    return dets, tgts


def val_ap(model, pool, device):
    dets, tgts = detections(model, pool, device)
    return average_precision(dets, tgts, pool[0]["x"].shape[-1], radius=2.0)


def train(model, draw, val, device, generator, lr=LR):
    """draw(generator) -> (x, heatmap). Returns how training stopped."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    best, best_state, best_step, bad = -1.0, None, 0, 0
    for step in range(1, MAX_STEPS + 1):
        x, heat = draw(generator)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        loss = focal_loss(logits, heat)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % VAL_EVERY == 0:
            v = val_ap(model, val, device)
            if v > best:
                best, best_step, bad = v, step, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return {"val_ap": best, "best_step": best_step, "stopped_at": step}
