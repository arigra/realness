"""Controlled defects in synthetic radar data (docs/notes/2026-10-27-damage-spread.md,
docs/notes/2026-10-28-damage-spread-fair.md).

Label defects act on a cache's traj/present; map defects re-render the same
simulator scene (same seed) with one physical change, so everything else in the
scene is identical to the clean sequence. Damage draws use their own generator.
"""
import dataclasses

import torch

from src.carrada_sim import CarradaSimulator, CarradaSpec, TrackScenario
from src.radial_sim import SceneSimulator, _elevation_gain_db, _power_db as _scene_power_db

GONE_DB = -200.0      # an RCS this low leaves nothing in the map


def drop_labels(present, frac, generator):
    """Remove each (object, frame) label with probability frac; the map is unchanged."""
    return present & (torch.rand(present.shape, generator=generator) >= frac)


def shift_labels(traj, axis, bins, n_bins, wrap):
    """Move every label `bins` cells along axis 0 (range, clamped) or 1 (Doppler, circular)."""
    t = traj.clone()
    v = t[..., axis] + bins
    t[..., axis] = v % n_bins if wrap else v.clamp(0, n_bins - 1)
    return t


def smooth_background(x, traj, present, frac, win_r=3, win_d=2, protect=True):
    """Over-smoothing, as the generator does: pull every cell frac of the way to
    its 4-neighbour mean, except (protect=True, Study 1) around labelled objects;
    protect=False smooths the whole map, objects included. x (L, R, D) dB."""
    nb = (torch.roll(x, 1, -2) + torch.roll(x, -1, -2) + torch.roll(x, 1, -1) + torch.roll(x, -1, -1)) / 4
    if not protect:
        return x - frac * (x - nb)
    keep = torch.zeros_like(x, dtype=torch.bool)
    n_d = x.shape[-1]
    for m, t in present.nonzero().tolist():
        r, d = (int(round(float(v))) for v in traj[m, t])
        rows = slice(max(r - win_r, 0), r + win_r + 1)
        keep[t, rows, [(d + k) % n_d for k in range(-win_d, win_d + 1)]] = True
    return torch.where(keep, x, x - frac * (x - nb))


def clutter_scenario(offset_db, base=TrackScenario()):
    """Ground and poles `offset_db` stronger (negative: weaker)."""
    g, p = base.ground_rcs_dbsm, base.pole_rcs_dbsm
    return dataclasses.replace(base, ground_rcs_dbsm=(g[0] + offset_db, g[1]),
                               pole_rcs_dbsm=(p[0] + offset_db, p[1]))


def noisy_spec(offset_db, base=CarradaSpec()):
    """Receiver noise figure `offset_db` worse: everything sinks towards the noise."""
    return dataclasses.replace(base, noise_figure_db=base.noise_figure_db + offset_db)


class DamagedCarradaSimulator(CarradaSimulator):
    """The engineer simulator with optional defects:
    - target_offset_db: moving objects weaker or stronger;
    - blink_frac: each (object, frame) removed from map and label;
    - unrender_frac: each (object, frame) removed from the map, label kept;
    - extra_frac: a sequence gets one extra moving object with no label.
    The parent's random draws always happen in the same order, so the scene is the
    clean one; the extra object is drawn and rendered with the damage generator and
    its power added to the frame."""

    def __init__(self, *args, target_offset_db=0.0, blink_frac=0.0, unrender_frac=0.0,
                 extra_frac=0.0, blink_generator=None, **kw):
        super().__init__(*args, **kw)
        self.target_offset_db, self.blink_frac = target_offset_db, blink_frac
        self.unrender_frac, self.extra_frac = unrender_frac, extra_frac
        self.blink_g = blink_generator
        self._extra, self._frame = None, 0

    def _coin(self, p):
        return p > 0 and float(torch.rand((), generator=self.blink_g)) < p

    def _with_damage_generator(self, fn, *args):
        main, self.g = self.g, self.blink_g
        try:
            return fn(*args)
        finally:
            self.g = main

    def _returns(self, o, t):
        ret = super()._returns(o, t)
        if ret is None or self._coin(self.blink_frac):
            return None
        px, py, v_app, rcs = ret
        off = GONE_DB if self._coin(self.unrender_frac) else self.target_offset_db
        return px, py, v_app, rcs + off

    def _render(self, x, y, v_app, rcs):
        power = super()._render(x, y, v_app, rcs)
        if self._extra is not None:
            t = self._frame * self.spec.frame_interval_s
            ret = self._with_damage_generator(CarradaSimulator._returns, self, self._extra, t)
            if ret is not None:
                power = power + self._with_damage_generator(CarradaSimulator._render, self, *ret)
        self._frame += 1
        return power

    def gen_sequence(self):
        self._frame = 0
        if self._coin(self.extra_frac):
            cls = 1 + int(torch.randint(3, (), generator=self.blink_g))
            self._extra = self._with_damage_generator(self._object, cls, 99)
        return super().gen_sequence()


# ---------------------------------------------------------------- RADIal


class DamagedSceneSimulator(SceneSimulator):
    """The RADIal engineer simulator with the same defects as DamagedCarradaSimulator,
    on labelled (moving) vehicles; parked cars are scene and stay as they are.
    clutter_offset_db: every static scatterer (kerbs, rails, facades, trees, poles,
    ground) stronger or weaker. Re-rendering must run on the device the cache was
    built on (its noise generator is a device generator)."""

    def __init__(self, *args, target_offset_db=0.0, unrender_frac=0.0, extra_frac=0.0,
                 clutter_offset_db=0.0, blink_generator=None, **kw):
        super().__init__(*args, **kw)
        self.target_offset_db, self.unrender_frac = target_offset_db, unrender_frac
        self.extra_frac, self.clutter_offset_db = extra_frac, clutter_offset_db
        self.blink_g = blink_generator
        self._extra, self._frame, self._v_ego = None, 0, 0.0

    _coin = DamagedCarradaSimulator._coin
    _with_damage_generator = DamagedCarradaSimulator._with_damage_generator

    def _car_returns(self, v, px, py):
        visible, rcs = super()._car_returns(v, px, py)
        if not v["moving"]:
            return visible, rcs
        return visible, rcs + (GONE_DB if self._coin(self.unrender_frac) else self.target_offset_db)

    def _static_scene(self, road, half_width, ymax):
        x, y, rcs, surf, parked = super()._static_scene(road, half_width, ymax)
        return x, y, rcs + self.clutter_offset_db, surf, parked

    def _vehicles(self, road, half_width, v_ego):
        out = super()._vehicles(road, half_width, v_ego)
        self._v_ego = v_ego
        if self._coin(self.extra_frac):          # one more vehicle, drawn like _vehicles draws one
            def draw():
                oncoming = bool(torch.rand((), generator=self.g) < road.p_oncoming)
                lane = float(self._u(0.3, 0.9)) * half_width
                x = -lane if oncoming else lane * float(self._u(-1.0, 1.0))
                speed = (float(self._u(*road.vehicle_speed_mps)) if oncoming else
                         max(v_ego + float(self._n(0.0, self.sc.traffic_speed_spread_mps
                                                   + self.sc.traffic_speed_spread_frac * v_ego)), 0.0))
                return self._car(x, float(self._u(8.0, self.spec.max_range_m - 10)),
                                 self.sc.car_rcs_dbsm, vy=-speed if oncoming else speed)
            self._extra = self._with_damage_generator(draw)
        return out

    def _render(self, rng_m, az, vr, power_db, looks_phase):
        power = super()._render(rng_m, az, vr, power_db, looks_phase)
        if self._extra is not None:            # the extra vehicle, as gen_sequence renders one
            spec, sc, v = self.spec, self.sc, self._extra
            t = self._frame * spec.frame_interval_s
            px, py = v["x"], v["y"] + v["vy"] * t - self._v_ego * t
            visible, rcs = SceneSimulator._car_returns(self, v, px, py)
            if bool(visible.any()):
                x, y = px[visible], py[visible]
                r, a = torch.sqrt(x ** 2 + y ** 2), torch.atan2(x, y)
                p = (_scene_power_db(rcs, r, spec) + _elevation_gain_db(r, spec, sc)
                     + 10 * sc.azimuth_pattern_power * torch.log10(torch.cos(a).abs().clamp(min=1e-3))
                     + sc.snr_offset_db)
                vr_x = (torch.full_like(r, v["vy"]) - self._v_ego) * torch.cos(a)
                power = power + self._with_damage_generator(SceneSimulator._render, self, r, a, vr_x, p, None)
        self._frame += 1
        return power

    def gen_sequence(self):
        self._frame, self._extra = 0, None
        return super().gen_sequence()
