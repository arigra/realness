"""A compact transformer heatmap detector (docs/notes/2026-11-06-metric-experiments.md, Phase 7.1).

Same input, output and training as src/detector.FrameDetector (one frame -> object heatmap logits),
about the same parameter count (~0.85M vs 0.84M). A convolutional stem (Doppler wraps), patches of
P x P cells as tokens, a 4-layer transformer encoder, and an upsampling decoder joined with the stem.
Positions: range sinusoids; Doppler sinusoids whose periods divide the number of Doppler patches,
so the encoding is circular like the radar's Doppler axis (no seam). Patch size: 4 for 256 x 64
maps (1,024 tokens), 8 for 512 x 256 (2,048 tokens).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.detector import _conv


def _positions(n_r, n_d, dim):
    """(n_r * n_d, dim): half range sinusoids, half circular Doppler harmonics."""
    half = dim // 2
    r = torch.arange(n_r, dtype=torch.float)[:, None]
    freq = torch.exp(torch.arange(0, half, 2, dtype=torch.float) * (-math.log(1000.0) / half))
    pr = torch.cat([torch.sin(r * freq), torch.cos(r * freq)], 1)                  # (n_r, half)
    d = torch.arange(n_d, dtype=torch.float)[:, None]
    # the lowest integer harmonics, amplitude 1/k (all harmonics up to Nyquist at equal weight sum to
    # a flat kernel and erase locality); unused dims are zero
    k = torch.arange(1, min(4, n_d // 2) + 1, dtype=torch.float)[None]
    pd = torch.cat([torch.sin(2 * math.pi * k * d / n_d), torch.cos(2 * math.pi * k * d / n_d)], 1) / k.repeat(1, 2)
    pd = F.pad(pd, (0, half - pd.shape[1]))
    return torch.cat([pr[:, None].expand(n_r, n_d, half), pd[None].expand(n_r, n_d, half)], -1).reshape(-1, dim)


class TransformerDetector(nn.Module):
    def __init__(self, dim=128, depth=4, heads=4, stem=16, patch=None):
        super().__init__()
        self.dim, self.patch = dim, patch
        self.stem = _conv(1, stem)
        self.embed = nn.LazyConv2d(dim, 1)       # replaced per patch size at first call
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True,
                                       activation="gelu"), depth)
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Conv2d(dim, stem, 1)
        self.dec = _conv(2 * stem, stem)
        self.head = nn.Conv2d(stem, 1, 1)
        self.head.bias.data.fill_(-4.0)
        self._pos = {}
        self._stem = stem

    def _patchify(self, h, p):
        if not isinstance(self.embed, nn.Conv2d) or self.embed.kernel_size != (p, p):
            self.embed = nn.Conv2d(self._stem, self.dim, p, stride=p).to(h.device)
        return self.embed(h)

    def forward(self, x):
        b, l = x.shape[:2]
        R, D = x.shape[2:]
        p = self.patch or (4 if R * D <= 256 * 64 else 8)
        h = self.stem(x.reshape(b * l, 1, R, D))
        t = self._patchify(h, p)                                  # (B, dim, R/p, D/p)
        n_r, n_d = t.shape[-2:]
        key = (n_r, n_d)
        if key not in self._pos:
            self._pos[key] = _positions(n_r, n_d, self.dim).to(x.device)
        tok = t.flatten(2).transpose(1, 2) + self._pos[key].to(t.dtype)
        tok = self.norm(self.layers(tok))
        t = tok.transpose(1, 2).reshape(b * l, self.dim, n_r, n_d)
        u = F.interpolate(self.up(t), size=(R, D), mode="nearest")
        out = self.head(self.dec(torch.cat([u, h], 1)))
        return out.reshape(b, l, R, D)


def build(n_range, n_doppler, device):
    """A TransformerDetector with its patch embedding created for this map size."""
    m = TransformerDetector().to(device)
    with torch.no_grad():
        m(torch.zeros(1, 1, n_range, n_doppler, device=device))
    return m
