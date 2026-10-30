"""The frame detector and how its detections are scored.

A small U-Net reads one range-Doppler frame and predicts a heatmap with a peak
at each object (CenterNet style). It works frame by frame, so no training
source can gain from reading neighbouring frames. The Doppler axis is circular
(unshifted FFT order), so targets, convolutions, peak picking and matching all
wrap around it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def heatmap_targets(traj, present, n_range=512, n_doppler=256, sigma=2.0):
    """Gaussian blobs at each present object: (B, L, n_range, n_doppler).

    Doppler distance wraps; an object near bin 255 must light up bin 0 as well,
    or the detector is taught a seam the radar does not have.
    """
    B, M, L, _ = traj.shape
    device = traj.device
    r = torch.arange(n_range, device=device, dtype=torch.float).view(1, 1, 1, n_range, 1)
    d = torch.arange(n_doppler, device=device, dtype=torch.float).view(1, 1, 1, 1, n_doppler)
    # Centres are rounded to a cell: the focal loss counts a cell as an object
    # only if its target is exactly 1.0, and labelled ranges are fractional.
    tr = torch.round(traj[..., 0].float()).view(B, M, L, 1, 1)
    td = torch.remainder(torch.round(traj[..., 1].float()), n_doppler).view(B, M, L, 1, 1)
    dd = torch.remainder(d - td + n_doppler / 2, n_doppler) - n_doppler / 2
    blobs = torch.exp(-((r - tr) ** 2 + dd ** 2) / (2 * sigma ** 2))
    blobs = blobs * present.to(device).view(B, M, L, 1, 1).float()
    return blobs.amax(dim=1)


def _conv(cin, cout):
    """3x3 convolution that wraps in Doppler and pads normally in range."""
    return nn.Sequential(
        _WrapPad(), nn.Conv2d(cin, cout, 3, padding=0), nn.GroupNorm(1, cout), nn.SiLU(),
        _WrapPad(), nn.Conv2d(cout, cout, 3, padding=0), nn.GroupNorm(1, cout), nn.SiLU())


class _WrapPad(nn.Module):
    def forward(self, x):
        x = F.pad(x, (1, 1, 0, 0), mode="circular")     # Doppler: circular
        return F.pad(x, (0, 0, 1, 1), mode="replicate")  # range: replicate


class FrameDetector(nn.Module):
    """U-Net over one frame; (B, L, R, D) inputs are folded into the batch."""

    def __init__(self, width=32):
        super().__init__()
        w = width
        self.enc1, self.enc2, self.enc3 = _conv(1, w), _conv(w, 2 * w), _conv(2 * w, 4 * w)
        self.mid = _conv(4 * w, 4 * w)
        self.dec3, self.dec2, self.dec1 = _conv(8 * w, 2 * w), _conv(4 * w, w), _conv(2 * w, w)
        self.head = nn.Conv2d(w, 1, 1)
        self.head.bias.data.fill_(-4.0)                 # rare positives

    def forward(self, x):
        b, l = x.shape[:2]
        h = x.reshape(b * l, 1, *x.shape[2:])
        e1 = self.enc1(h)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        m = self.mid(F.avg_pool2d(e3, 2))
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="nearest")
        d3 = self.dec3(torch.cat([up(m, e3), e3], 1))
        d2 = self.dec2(torch.cat([up(d3, e2), e2], 1))
        d1 = self.dec1(torch.cat([up(d2, e1), e1], 1))
        return self.head(d1).reshape(b, l, *x.shape[2:])


def focal_loss(logits, target, alpha=2, beta=4):
    """CenterNet penalty-reduced focal loss, normalised by the number of target centres."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    pos_loss = -(torch.log(p) * (1 - p) ** alpha * pos).sum()
    neg_loss = -(torch.log(1 - p) * p ** alpha * (1 - target) ** beta * (1 - pos)).sum()
    return (pos_loss + neg_loss) / pos.sum().clamp_min(1)


def detect(prob, threshold=0.05, top_k=10):
    """prob (F, R, D) in [0, 1] -> per frame a list of (score, range, doppler):
    local maxima above threshold, Doppler wrapping, the top_k strongest."""
    p = prob[:, None]
    p = torch.cat([p[..., -1:], p, p[..., :1]], dim=-1)
    local_max = F.max_pool2d(p, 3, 1, padding=(1, 0))[:, 0]
    peaks = (prob == local_max) & (prob > threshold)
    out = []
    for f in range(prob.shape[0]):
        rs, ds = torch.nonzero(peaks[f], as_tuple=True)
        scores = prob[f, rs, ds]
        order = torch.argsort(scores, descending=True)[:top_k]
        out.append([(float(scores[i]), float(rs[i]), float(ds[i])) for i in order])
    return out


def _distances(tgt, r, d, n_doppler):
    dd = torch.remainder(tgt[:, 1] - d + n_doppler / 2, n_doppler) - n_doppler / 2
    return torch.sqrt((tgt[:, 0] - r) ** 2 + dd ** 2)


def average_precision(detections, targets, n_doppler, radius=2.0):
    """All-point interpolated AP over all frames, class-agnostic. A detection is
    a hit if an unmatched labelled object lies within `radius` cells (Doppler
    wrapping); strongest detections are matched first.
    detections: per frame a list of (score, r, d); targets: per frame (m, 2)."""
    scored, n_true = [], 0
    for frame_dets, tgt in zip(detections, targets):
        n_true += len(tgt)
        used = torch.zeros(len(tgt), dtype=torch.bool)
        for score, r, d in sorted(frame_dets, key=lambda t: -t[0]):
            hit = -1
            if len(tgt):
                dist = _distances(tgt, r, d, n_doppler)
                dist[used] = float("inf")
                best = int(torch.argmin(dist))
                if float(dist[best]) <= radius:
                    hit = best
            if hit >= 0:
                used[hit] = True
            scored.append((score, hit >= 0))
    if n_true == 0:
        return float("nan")
    scored.sort(key=lambda t: -t[0])
    tp = fp = 0
    points = []
    for _, is_tp in scored:
        tp, fp = tp + int(is_tp), fp + int(not is_tp)
        points.append((tp / n_true, tp / (tp + fp)))
    # precision envelope, then sum over recall steps (all-point interpolation)
    envelope, best = [], 0.0
    for recall, precision in reversed(points):
        best = max(best, precision)
        envelope.append((recall, best))
    ap, prev_recall = 0.0, 0.0
    for recall, precision in reversed(envelope):
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def recall(detections, targets, n_doppler, radius=2.0):
    """Share of labelled objects with a detection within radius (matched as in the AP)."""
    hit = n = 0
    for frame_dets, tgt in zip(detections, targets):
        n += len(tgt)
        used = torch.zeros(len(tgt), dtype=torch.bool)
        for _, r, d in sorted(frame_dets, key=lambda t: -t[0]):
            if not len(tgt):
                break
            dist = _distances(tgt, r, d, n_doppler)
            dist[used] = float("inf")
            best = int(torch.argmin(dist))
            if float(dist[best]) <= radius:
                used[best] = True
        hit += int(used.sum())
    return hit / max(n, 1)


def scores(detections, targets, n_doppler):
    """The study's three scores: AP within 2 cells (primary), AP within 4, recall within 2."""
    return {"ap2": average_precision(detections, targets, n_doppler, radius=2.0),
            "ap4": average_precision(detections, targets, n_doppler, radius=4.0),
            "recall2": recall(detections, targets, n_doppler, radius=2.0)}
