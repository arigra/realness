"""The RADDet simulator tuned to the real RADDet maps (docs/notes/2026-11-01-realness-score.md,
step 6; Ari 2026-11-01: tune to fix the specific gaps). Each fix is a switch, so
a version can carry one fix or all of them:
- processing: RADDet's own FFT windows, found by inverting its cubes: Hann in
  range, none in Doppler (the simulator used Hamming in both, which spreads the
  static scene over +/-1 Doppler bin; real maps hold it in one bin);
- range_gain: the receiver's filters, as the gain per range row measured on the
  50 real training sequences (median over frames and Doppler bins off the
  static ridge, relative to the map median): ~10 rows of roll-off near, a dip to
  -9 dB around row 243 far; applied to signal and noise;
- target_offset_db, static_offset_db: moving objects and the static scene
  (ground, poles, parked cars, facades) weaker, fitted together so that target
  over floor and target over strong clutter (src/measures.py) match the real
  training sequences; target strength alone cannot match both.
"""
import json
from pathlib import Path

import numpy as np
import torch

from src.radar import window
from src.raddet_sim import RaddetSimulator

GAIN_FILE = Path("results/tuning/raddet_range_gain.json")


def measure_range_gain(real_pool):
    """Gain per range row (dB) of real maps; real_pool = src.data.load items."""
    x = torch.stack([it["x"].float() for it in real_pool]).flatten(0, 1)
    off = torch.cat([x[:, :, :28], x[:, :, 37:]], 2)
    row = off.median(dim=2).values.median(dim=0).values
    return (row - off.median()).tolist()


def range_gain():
    return torch.tensor(json.loads(GAIN_FILE.read_text())["gain_db"])


class TunedRaddetSimulator(RaddetSimulator):
    def __init__(self, *args, processing=False, range_gain_db=None, target_offset_db=0.0,
                 static_offset_db=0.0, **kw):
        super().__init__(*args, **kw)
        if processing:
            self.win_r = window(self.spec.n_range, "hann").to(self.device)
            self.win_d = window(self.spec.n_doppler, "none").to(self.device)
        self.range_gain_db, self.target_offset_db = range_gain_db, target_offset_db
        self.static_offset_db = static_offset_db

    def _static(self):
        x, y, rcs = super()._static()
        return x, y, rcs + self.static_offset_db

    def _returns(self, o, t):
        ret = super()._returns(o, t)
        if ret is None or not self.target_offset_db:
            return ret
        px, py, v_app, rcs = ret
        return px, py, v_app, rcs + self.target_offset_db

    def gen_sequence(self):
        o = super().gen_sequence()
        if self.range_gain_db is not None:
            o["x"] = o["x"] + self.range_gain_db[None, :, None]
        return o
