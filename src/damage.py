"""Controlled defects in synthetic radar data (docs/notes/2026-10-27-damage-spread.md,
docs/notes/2026-10-28-damage-spread-fair.md).

Label defects act on a cache's traj/present; map defects re-render the same
simulator scene (same seed) with one physical change, so everything else in the
scene is identical to the clean sequence. Damage draws use their own generator.
"""
import dataclasses
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal.windows import chebwin

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
        # the spec simulator has no real scenes; a subclass that adds them may draw its own
        if getattr(self, "scenes", None) is not None and getattr(self, "_own_scenario", False):
            keep, self.scenes = self.scenes, None
            try:
                out = super()._vehicles(road, half_width, v_ego)
            finally:
                self.scenes = keep
        else:
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


# ------------------------------------------------- on the tuned simulators
# Study 2 (tolerance curves) on the simulators of src/tuned.py. Strength
# defects (contrast, clutter, noise) are the tuned offsets moved further
# (noise +d dB = targets and static scene both -d dB: maps are normalised, so
# only levels relative to the noise count). The rest are here. Damage draws use
# their own generator, so a damaged sequence shows the clean one's scene.


def _spectrum(w, pad=64):
    s = torch.fft.fft(w, n=len(w) * pad).abs() ** 2
    return 10 * torch.log10(s / s.max() + 1e-30), pad


def mainlobe_fwhm(w):
    """Half-power width of a window's response, in bins."""
    s, pad = _spectrum(w)
    half = int((s[: len(s) // 2] >= -3.0103).nonzero().max())
    return 2 * (half + 0.5) / pad


def peak_sidelobe_db(w):
    """The highest sidelobe of a window's response (dB under the peak)."""
    s, pad = _spectrum(w)
    first = s[: len(s) // 2]
    null = int(torch.nonzero(first[1:] > first[:-1]).min())       # first rise = first null
    return -float(first[null:].max())


def sidelobe_window(w, delta_db):
    """A Chebyshev window whose sidelobes are delta_db higher than w's (lower if negative)."""
    at = max(peak_sidelobe_db(w) - delta_db, 3.0)
    with warnings.catch_warnings():          # chebwin warns below 45 dB; the window is still what we ask
        warnings.simplefilter("ignore")
        c = torch.tensor(chebwin(len(w), at), dtype=torch.float)
    return c.to(w.device)


def blur_sigma(w, factor):
    """Gaussian sigma (bins) that makes the response `factor` x wider at half power."""
    return 0.0 if factor <= 1 else mainlobe_fwhm(w.cpu()) * math.sqrt(factor ** 2 - 1) / 2.3548


def gaussian_blur(power, sr, sd):
    """Blur a (R, D) power map; Doppler circular, range zero beyond the edges."""
    out = power[None, None]
    if sr > 0:
        k = max(int(math.ceil(3 * sr)), 1)
        g = torch.exp(-torch.arange(-k, k + 1, device=power.device, dtype=power.dtype) ** 2 / (2 * sr ** 2))
        out = F.conv2d(F.pad(out, (0, 0, k, k)), (g / g.sum()).view(1, 1, -1, 1))
    if sd > 0:
        k = max(int(math.ceil(3 * sd)), 1)
        g = torch.exp(-torch.arange(-k, k + 1, device=power.device, dtype=power.dtype) ** 2 / (2 * sd ** 2))
        out = F.conv2d(F.pad(out, (k, k, 0, 0), mode="circular"), (g / g.sum()).view(1, 1, 1, -1))
    return out[0, 0]


def range_tail(power, length):
    """Smear a (R, D) power map away from the radar (towards higher rows) with a
    one-sided exponential of this decay length (cells); total power is kept."""
    k = int(math.ceil(5 * length))
    h = torch.exp(-torch.arange(k + 1, device=power.device, dtype=power.dtype) / length)
    out = F.conv2d(F.pad(power[None, None], (0, 0, k, 0)), (h / h.sum()).flip(0).view(1, 1, -1, 1))
    return out[0, 0]


class _Damage:
    """Shared by the two damage mixins: damage coins, windows, blur and tail.
    Study 2b (docs/notes/2026-11-02-study2b.md) adds own_scenario_frac (share of
    sequences drawn from the simulator's own scenario instead of a real frame's
    labels), annotation_db (objects under the annotation threshold + this many
    dB left unlabelled), tail (cells) and confusers (x discrete static structures)."""

    def _setup(self, unrender_frac, extra_frac, blur, sidelobes_db, damage_generator,
               own_scenario_frac=0.0, annotation_db=None, tail=0.0, confusers=1.0):
        self.unrender_frac, self.extra_frac, self.dg = unrender_frac, extra_frac, damage_generator
        if sidelobes_db is not None:
            self.win_r = sidelobe_window(self.win_r, sidelobes_db)
            self.win_d = sidelobe_window(self.win_d, sidelobes_db)
        self.sigma = (blur_sigma(self.win_r, blur), blur_sigma(self.win_d, blur))
        self.tail, self.own_scenario_frac, self.annotation_db = tail, own_scenario_frac, annotation_db
        self.confusers = confusers
        self._extra, self._frame = [], 0

    def _split_targets(self):
        """Targets rendered on their own (to blur or smear them)."""
        return self.sigma != (0.0, 0.0) or self.tail > 0

    def _coin(self, p):
        return p > 0 and float(torch.rand((), generator=self.dg)) < p

    def _n_extra(self):
        whole = int(self.extra_frac)
        return whole + int(self._coin(self.extra_frac - whole))

    def _damage_draws(self, fn, *args):
        main, self.g = self.g, self.dg
        try:
            return fn(*args)
        finally:
            self.g = main

    def _blurred(self, power):
        if self.sigma != (0.0, 0.0):
            power = gaussian_blur(power, *self.sigma)
        if self.tail > 0:
            power = range_tail(power, self.tail)
        return power


class TrackDamage(_Damage):
    """Defects for TunedCarrada / TunedRaddet (mixed in before them):
    unrender_frac: each (object, frame) gone from the map, label kept;
    extra_frac: unlabelled extra objects per sequence (1.0 = one, 2.0 = two;
      fractions are a chance), placed like a real frame's objects;
    blur: the target response this many times wider (targets only);
    sidelobes_db: Chebyshev windows with sidelobes this much higher than the
      tuned processing's, both axes."""

    def __init__(self, *args, unrender_frac=0.0, extra_frac=0.0, blur=1.0, sidelobes_db=None,
                 damage_generator=None, own_scenario_frac=0.0, annotation_db=None, tail=0.0,
                 confusers=1.0, micro=1.0, **kw):
        super().__init__(*args, **kw)
        self._setup(unrender_frac, extra_frac, blur, sidelobes_db, damage_generator,
                    own_scenario_frac, annotation_db, tail, confusers)
        if micro != 1.0:      # limbs and wheels move less (0: every part at the body's speed)
            self.sc = dataclasses.replace(self.sc, limb_swing=self.sc.limb_swing * micro, wheel_micro=micro)
        if confusers != 1.0:
            more = {"poles_mean": self.sc.poles_mean * confusers}
            if hasattr(self.sc, "parked_mean"):
                more["parked_mean"] = self.sc.parked_mean * confusers
            self.sc = dataclasses.replace(self.sc, **more)
        if annotation_db is not None:
            self.label_min_snr_db = (self.label_min_snr_db or 0.0) + annotation_db

    def _objects(self):
        if self.scenes is None or not self._own_scenario:
            return super()._objects()
        keep, self.scenes = self.scenes, None
        try:
            return super()._objects()
        finally:
            self.scenes = keep

    def _static(self):
        x, y, rcs = super()._static()
        self._n_static = len(x)
        return x, y, rcs

    def _returns(self, o, t):
        ret = super()._returns(o, t)
        if ret is None or not self._coin(self.unrender_frac):
            return ret
        px, py, v_app, rcs = ret
        return px, py, v_app, rcs + GONE_DB

    def _label(self, o, t, px, py, v_app, rcs):
        if bool((rcs < GONE_DB / 2).all()):          # unrendered: label where it would be
            rcs = rcs - GONE_DB
        return super()._label(o, t, px, py, v_app, rcs)

    def _render(self, x, y, v_app, rcs):
        n, base = self._n_static, super()._render
        if not self._split_targets():
            power = base(x, y, v_app, rcs)
        else:                       # targets rendered on their own to blur them
            power = base(x[:n], y[:n], v_app[:n], rcs[:n]) + self._blurred(base(x[n:], y[n:], v_app[n:], rcs[n:]))
        t = self._frame * self.spec.frame_interval_s
        for o in self._extra:
            ret = self._damage_draws(super()._returns, o, t)
            if ret is not None:
                power = power + self._blurred(self._damage_draws(base, *ret))
        self._frame += 1
        return power

    def _extra_object(self, k):
        scenes = [s for s in (self.scenes or []) if s]
        if scenes:
            scene = scenes[int(torch.randint(len(scenes), (), generator=self.g))]
            return self._from_label(scene[int(torch.randint(len(scene), (), generator=self.g))], 90 + k)
        return self._object(1 + int(torch.randint(3, (), generator=self.g)), 90 + k)

    def gen_sequence(self):
        self._frame = 0
        self._own_scenario = self._coin(self.own_scenario_frac)
        self._extra = [self._damage_draws(self._extra_object, k) for k in range(self._n_extra())]
        return super().gen_sequence()


class RadialDamage(_Damage):
    """The same defects for TunedRadial, on labelled (moving) vehicles; the extra
    vehicles are placed like a real frame's labelled ones and left unlabelled."""

    def __init__(self, *args, unrender_frac=0.0, extra_frac=0.0, blur=1.0, sidelobes_db=None,
                 damage_generator=None, own_scenario_frac=0.0, annotation_db=None, tail=0.0,
                 confusers=1.0, **kw):
        super().__init__(*args, **kw)
        self._setup(unrender_frac, extra_frac, blur, sidelobes_db, damage_generator,
                    own_scenario_frac, annotation_db, tail, confusers)
        if confusers != 1.0:
            self.sc = dataclasses.replace(self.sc, structure_coverage_scale=confusers,
                                          tree_density_scale=confusers, pole_density_scale=confusers)
        self._parts, self._v_ego = [], 0.0

    def _static_scene(self, road, half_width, ymax):
        out = super()._static_scene(road, half_width, ymax)
        self._n_static = len(out[0])
        return out

    def _car_returns(self, v, px, py):
        v["moving"] = v.setdefault("moving0", v["moving"])   # restored each frame (see below)
        visible, rcs = super()._car_returns(v, px, py)
        if v["moving"] and self.annotation_db is not None and bool(visible.any()):
            spec, sc = self.spec, self.sc
            x, y = px[visible], py[visible]
            r, a = torch.sqrt(x ** 2 + y ** 2), torch.atan2(x, y)
            snr = (_scene_power_db(rcs, r, spec) + _elevation_gain_db(r, spec, sc)
                   + 10 * sc.azimuth_pattern_power * torch.log10(torch.cos(a).abs().clamp(min=1e-3))
                   + sc.snr_offset_db)
            if float(snr.max()) < self.annotation_db:
                v["moving"] = False      # gen_sequence labels moving vehicles only: no label this frame
                self._parts.append((int(visible.sum()), True))
                return visible, rcs
        if v["moving"] and self._coin(self.unrender_frac):
            rcs = rcs + GONE_DB
        self._parts.append((int(visible.sum()), v["moving"]))
        return visible, rcs

    def _vehicles(self, road, half_width, v_ego):
        if self.scenes is not None and self._own_scenario:
            keep, self.scenes = self.scenes, None
            try:
                out = super()._vehicles(road, half_width, v_ego)
            finally:
                self.scenes = keep
        else:
            out = super()._vehicles(road, half_width, v_ego)
        self._v_ego = v_ego
        scenes = [s for s in (self.scenes or []) if s]

        def draw(k):
            if scenes:
                keep, self.scenes = self.scenes, [scenes[int(torch.randint(len(scenes), (), generator=self.g))]]
                try:
                    v = super(RadialDamage, self)._vehicles(road, half_width, v_ego)
                finally:
                    self.scenes = keep
                return v[int(torch.randint(len(v), (), generator=self.g))]
            spread = self.sc.traffic_speed_spread_mps + self.sc.traffic_speed_spread_frac * v_ego
            return self._car(float(self._u(-0.9, 0.9)) * half_width, float(self._u(8.0, self.spec.max_range_m - 10)),
                             self.sc.car_rcs_dbsm, vy=max(v_ego + float(self._n(0.0, spread)), 0.1))
        self._extra = [self._damage_draws(draw, k) for k in range(self._n_extra())]
        return out

    def _render(self, rng_m, az, vr, power_db, looks_phase):
        base = super()._render
        target = torch.cat([torch.zeros(self._n_static, dtype=torch.bool)]
                           + [torch.full((n,), m, dtype=torch.bool) for n, m in self._parts])
        assert len(target) == len(rng_m), "scatterers out of step with their vehicles"
        self._parts = []
        if not self._split_targets():
            power = base(rng_m, az, vr, power_db, None)
        else:                       # targets rendered on their own to blur them
            power = (base(rng_m[~target], az[~target], vr[~target], power_db[~target], None)
                     + self._blurred(base(rng_m[target], az[target], vr[target], power_db[target], None)))
        spec, sc = self.spec, self.sc
        t = self._frame * spec.frame_interval_s
        for v in self._extra:
            px, py = v["x"], v["y"] + v["vy"] * t - self._v_ego * t
            visible, rcs = SceneSimulator._car_returns(self, v, px, py)
            if not bool(visible.any()):
                continue
            rcs = rcs + self.target_offset_db
            x, y = px[visible], py[visible]
            r, a = torch.sqrt(x ** 2 + y ** 2), torch.atan2(x, y)
            p = (_scene_power_db(rcs, r, spec) + _elevation_gain_db(r, spec, sc)
                 + 10 * sc.azimuth_pattern_power * torch.log10(torch.cos(a).abs().clamp(min=1e-3))
                 + sc.snr_offset_db)
            vr_x = (torch.full_like(r, v["vy"]) - self._v_ego) * torch.cos(a)
            power = power + self._blurred(self._damage_draws(base, r, a, vr_x, p, None))
        self._frame += 1
        return power

    def gen_sequence(self):
        self._frame, self._parts = 0, []
        self._own_scenario = self._coin(self.own_scenario_frac)
        return super().gen_sequence()
