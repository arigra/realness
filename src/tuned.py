"""The three simulators tuned to their real data (Ari, 2026-11-02: three
simulators as good as possible; tuning to the real data is allowed here).

Every fix is a switch, so a version can carry any subset and each fix's share
can be measured by leaving it out. Everything a fix uses is measured on the 50
real training sequences of the dataset (never val or test) and stored in
results/tuning/<dataset>.json by scripts/tune_simulators.py:

- processing (CARRADA, RADDet): the FFT windows RADDet's cubes were made with,
  Hann in range and none in Doppler. CARRADA is the same chip and shows the same
  one-bin zero-Doppler ridge, so it gets the same windows.
- range_gain: per range row, the real maps' median level minus the simulator's
  (receiver filters, the far-range dip); added to every map.
- scenes: each sequence starts from the labelled objects of a real training
  frame: their count, range, Doppler (and class; RADIal: position). What is not
  labelled (heading, azimuth for CARRADA/RADDet) is drawn as before.
- label: where the label sits on an object, as both datasets annotate it: the
  centre of the box around the object's returns (CARRADA boxes enclose the RD
  points found on the object, RADDet's the object's cells in the RAD cube).
  Returns count from SNR >= label_snr_db; 0 dB matched the real mean range
  profile around labels best (RADDet: energy centroid 2.57 cells nearer than
  the label vs 2.64 real; 10/20/30 dB and the object's centre were worse), so
  a car's label sits behind the near face that dominates the RD map.
  Both datasets label only what their detector found, so an object whose
  strongest return is under label_min_snr_db gets no label (it is still
  rendered); the threshold is fitted with the contrast offsets.
- clutter_profile: the static scene stronger or weaker per range, so that its
  level over the map (CARRADA/RADDet: the zero-Doppler ridge; RADIal: the 99th
  percentile of each row) follows the real one (zero mean: the overall level
  is the contrast fit's).
- body (RADIal): scatterers over the whole footprint of each vehicle (wheels,
  underbody, cabin), always visible, body_db under the car's RCS in total:
  real RADIal targets trail 2-10 cells behind the near face, the outline model
  does not.
- target_snr_db (CARRADA, RADDet): each object's strongest return is set to an
  SNR drawn once per object from N(mean, sd) dB, instead of following the link
  budget. Real labelled objects are a selected, narrow population (the
  annotation kept what its detector found): the link budget spread simulated
  strengths far wider (target over floor quartiles 1.8 / 4.5 / 7.6 vs real
  4.5 / 5.6 / 7.5 on CARRADA) and labelled objects buried in noise.
- target_offset_db, static_offset_db: moving objects and the static scene
  weaker or stronger, fitted so that the quartiles of target over floor and
  over strong clutter (src/measures.py) match the real training sequences'.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.carrada_sim import CarradaSimulator, _power_db
from src.radar import window
from src.raddet_sim import MODEL, SIZE, SPEED, RaddetSimulator
from src.radial_sim import SceneSimulator

TUNING = Path("results/tuning")
CARRADA_ROOT = Path("/truenas/home/arigra/permuter/ariGranevich/radar data/Carrada")
RADIAL_LABELS = Path("/truenas/home/arigra/permuter/ariGranevich/radar data/RADial/RADIal/labels.csv")


def table(dataset):
    """The measured tables of a dataset (scenes, range gain, fitted offsets)."""
    return json.loads((TUNING / f"{dataset}.json").read_text())


# ------------------------------------------------------------ real scenes

def real_scenes(dataset, meta):
    """The labelled objects of every frame of the real training sequences.
    meta: the cache's per-sequence meta (recording, samples). One scene per frame:
    CARRADA [{range_bin, doppler_bin, cls}], RADDet [{range_bin, doppler_bin, name}],
    RADIal [{x, y, doppler_bin}] (radar coordinates, m)."""
    scenes = []
    if dataset == "carrada":
        ann = json.loads((CARRADA_ROOT / "annotations_frame_oriented.json").read_text())
        for m in meta:
            for f in m["samples"]:
                objs = []
                for a in ann.get(m["recording"], {}).get(f, {}).values():
                    (r0, d0), (r1, d1) = a["range_doppler"]["box"]
                    objs.append({"range_bin": (r0 + r1) / 2, "doppler_bin": (d0 + d1) / 2,
                                 "cls": int(a["range_doppler"]["label"])})
                scenes.append(objs)
    elif dataset == "raddet":
        from src import raddet
        part = {o: {f: p for p, f in raddet.frames(o)} for o in ("train", "test")}
        for m in meta:
            for o, f in m["samples"]:          # (official split, frame id)
                scenes.append([{"range_bin": lab["range_bin"], "doppler_bin": lab["doppler_bin"], "name": lab["cls"]}
                               for lab in raddet.labels(o, part[o][f], f)])
    else:
        rows = {}
        for r in csv.DictReader(open(RADIAL_LABELS)):
            if float(r["radar_R_m"]) >= 0:
                rows.setdefault(int(r["numSample"]), []).append(
                    {"x": float(r["radar_X_m"]), "y": float(r["radar_Y_m"]),
                     "doppler_bin": float(r["radar_D_mps"])})
        for m in meta:
            scenes += [rows.get(int(s), []) for s in m["samples"]]
    return scenes


def _at_range(profile_db, range_m, range_res_m):
    """A per-row dB profile read at these ranges (m)."""
    p = torch.as_tensor(profile_db).float()
    i = (range_m / range_res_m).clamp(0, len(p) - 1)
    lo = i.floor().long().clamp(max=len(p) - 2)
    w = i - lo
    return p[lo] * (1 - w) + p[lo + 1] * w


def row_profile(pool):
    """Median level per range row (dB, relative to the maps' median); pool of src.data items."""
    x = torch.stack([it["x"].float() for it in pool]).flatten(0, 1)
    row = x.median(dim=2).values.median(dim=0).values
    return row - x.median()


# ------------------------------------------------------- CARRADA and RADDet

class _TunedTrack:
    """The switches for the CARRADA-family simulators (mixed in before the
    simulator class, so its overrides come first)."""

    def __init__(self, *args, processing=False, range_gain_db=None, scenes=None, label=None,
                 target_offset_db=0.0, static_offset_db=0.0, label_snr_db=0.0, label_min_snr_db=None,
                 clutter_profile_db=None, target_snr_db=None, **kw):
        super().__init__(*args, **kw)
        self.target_snr_db = target_snr_db
        if processing:
            self.win_r = window(self.spec.n_range, "hann").to(self.device)
            self.win_d = window(self.spec.n_doppler, "none").to(self.device)
        self.range_gain_db = None if range_gain_db is None else torch.as_tensor(range_gain_db).float()
        self.scenes, self.label_kind, self.label_snr_db = scenes, label, label_snr_db
        self.label_min_snr_db, self.clutter_profile_db = label_min_snr_db, clutter_profile_db
        self.target_offset_db, self.static_offset_db = target_offset_db, static_offset_db

    # strengths
    def _static(self):
        x, y, rcs = super()._static()
        if self.clutter_profile_db is not None:
            rcs = rcs + _at_range(self.clutter_profile_db, torch.sqrt(x ** 2 + y ** 2), self.spec.range_res_m)
        return x, y, rcs + self.static_offset_db

    def _returns(self, o, t):
        ret = super()._returns(o, t)
        if ret is None or (not self.target_offset_db and self.target_snr_db is None):
            return ret
        px, py, v_app, rcs = ret
        if self.target_snr_db is not None:
            if "snr_db" not in o:
                o["snr_db"] = float(self._n(*self.target_snr_db))
            rcs = rcs + (o["snr_db"] - float(self._snr_db(px, py, rcs).max()))
        return px, py, v_app, rcs + self.target_offset_db

    # scenes
    def _speeds(self, lab):
        return self.sc.speeds_mps[lab["cls"] - 1]

    def _from_label(self, lab, oid):
        """An object at the label's range whose velocity towards the radar is the
        label's; azimuth, speed (at least the radial one) and side drawn."""
        spec = self.spec
        r0 = lab["range_bin"] * spec.range_res_m
        v_r = (lab["doppler_bin"] - spec.zero_doppler) * spec.velocity_res_mps
        speed = max(float(self._u(*self._speeds(lab))), abs(v_r) * float(self._u(1.0, 1.3)), 0.05)
        az0 = math.radians(float(self._u(-60.0, 60.0)))
        x0, y0 = r0 * math.sin(az0), r0 * math.cos(az0)
        phi = math.acos(max(-1.0, min(1.0, v_r / speed)))     # velocity vs the direction to the radar
        heading = math.atan2(x0, y0) + math.pi + (phi if self._u(0, 1) < 0.5 else -phi)
        o = {"cls": lab["cls"], "id": oid, "x0": x0, "y0": y0, "vx": speed * math.sin(heading),
             "vy": speed * math.cos(heading), "speed": speed, "phase": float(self._u(0, 2 * math.pi))}
        if lab["cls"] == 3:
            o["car"] = self._car_outline(heading)
        return o

    def _objects(self):
        if self.scenes is None:
            return super()._objects()
        scene = self.scenes[int(torch.randint(len(self.scenes), (), generator=self.g))]
        return [self._from_label(lab, i) for i, lab in enumerate(scene[:8])]

    # labels
    def _snr_db(self, px, py, rcs):
        """Per scatterer: SNR in its peak cell (dB over one cell's noise), as _render weighs it."""
        spec, sc = self.spec, self.sc
        rng, az = torch.sqrt(px ** 2 + py ** 2), torch.atan2(px, py)
        dep = torch.rad2deg(torch.atan2(torch.tensor(sc.radar_height_m), rng))
        return (_power_db(rcs, rng, spec) - 24.0 * (dep / sc.elevation_beam_deg) ** 2
                + 10 * sc.azimuth_pattern_power * torch.log10(torch.cos(az).abs().clamp(min=1e-3)))

    def _label(self, o, t, px, py, v_app, rcs):
        if self.label_min_snr_db is not None and float(self._snr_db(px, py, rcs).max()) < self.label_min_snr_db:
            return None                                  # not detectable: not annotated
        if self.label_kind == "centre":
            bx, by = o["x0"] + o["vx"] * t, o["y0"] + o["vy"] * t
            return math.hypot(bx, by), -(o["vx"] * bx + o["vy"] * by) / math.hypot(bx, by)
        if self.label_kind == "box":
            snr = self._snr_db(px, py, rcs)
            keep = snr >= self.label_snr_db
            if not bool(keep.any()):
                keep = snr == snr.max()
            r, v = torch.sqrt(px ** 2 + py ** 2)[keep], v_app[keep]
            return float(r.min() + r.max()) / 2, float(v.min() + v.max()) / 2
        return super()._label(o, t, px, py, v_app, rcs)

    def gen_sequence(self):
        o = super().gen_sequence()
        if self.range_gain_db is not None:
            o["x"] = o["x"] + self.range_gain_db[None, :, None]
        return o


class TunedCarrada(_TunedTrack, CarradaSimulator):
    pass


class TunedRaddet(_TunedTrack, RaddetSimulator):
    def _speeds(self, lab):
        return SPEED[lab["name"]]

    def _from_label(self, lab, oid):
        o = super()._from_label({**lab, "cls": MODEL[lab["name"]]}, oid)
        o["name"] = lab["name"]
        if lab["name"] in SIZE:
            length, width, o["rcs_dbsm"] = SIZE[lab["name"]]
            heading = math.atan2(o["vx"], o["vy"])
            o["car"] = self._car_outline(heading, length, width)
        return o


# ------------------------------------------------------------------ RADIal

class TunedRadial(SceneSimulator):
    """RADIal switches: range_gain, scenes (labelled vehicles placed where a real
    frame's are: position from radar_X/Y, the nearest face at the label;
    velocity so that the relative radial velocity is the label's Doppler),
    target_offset_db (moving vehicles), static_offset_db (everything else)."""

    N_BODY = 16

    def __init__(self, *args, range_gain_db=None, scenes=None, target_offset_db=0.0,
                 static_offset_db=0.0, body_db=None, clutter_profile_db=None, **kw):
        super().__init__(*args, **kw)
        self.range_gain_db = None if range_gain_db is None else torch.as_tensor(range_gain_db).float()
        self.scenes, self.body_db, self.clutter_profile_db = scenes, body_db, clutter_profile_db
        self.target_offset_db, self.static_offset_db = target_offset_db, static_offset_db

    def _static_scene(self, road, half_width, ymax):
        x, y, rcs, surf, parked = super()._static_scene(road, half_width, ymax)
        res = self.spec.range_res_m
        if self.clutter_profile_db is not None:      # at the scene's ranges in the first frame
            rcs = rcs + _at_range(self.clutter_profile_db, torch.sqrt(x ** 2 + y ** 2), res)
        for p in parked:
            p["rcs_total"] += self.static_offset_db
            if self.clutter_profile_db is not None:
                p["rcs_total"] += float(_at_range(self.clutter_profile_db, torch.tensor(math.hypot(p["cx"], p["cy"])), res))
        return x, y, rcs + self.static_offset_db, surf, parked

    def _car(self, x, y, rcs_dbsm, vy, kind=0):
        v = super()._car(x, y, rcs_dbsm, vy, kind)
        if self.body_db is None:
            return v
        length, width = [(4.5, 1.8), (10.0, 2.5), (2.0, 0.8)][kind]
        n = self.N_BODY
        bx, by = self._u(-width / 2, width / 2, (n,)), self._u(-length / 2, length / 2, (n,))
        v["body"] = torch.cat([torch.zeros(len(v["x"]), dtype=torch.bool), torch.ones(n, dtype=torch.bool)])
        v["x"], v["y"] = torch.cat([v["x"], x + bx]), torch.cat([v["y"], y + by])
        v["nx"], v["ny"] = torch.cat([v["nx"], torch.zeros(n)]), torch.cat([v["ny"], torch.zeros(n)])
        v["scint"] = torch.cat([v["scint"], self._n(0, 3.0, (n,))])
        return v

    def _car_returns(self, v, px, py):
        body = v.get("body")
        if body is None:
            visible, rcs = super()._car_returns(v, px, py)
        else:                       # the outline as before, the body always seen
            shell = {**v, "nx": v["nx"][~body], "ny": v["ny"][~body], "scint": v["scint"][~body]}
            vis_s, rcs_s = super()._car_returns(shell, px[~body], py[~body])
            visible = body.clone()
            visible[~body] = vis_s
            full = torch.zeros(len(body))
            full[torch.nonzero(~body).squeeze(1)[vis_s]] = rcs_s
            full[body] = v["rcs_total"] + self.body_db - 10 * math.log10(self.N_BODY) + v["scint"][body]
            rcs = full[visible]
        return visible, rcs + (self.target_offset_db if v["moving"] else 0.0)

    def _vehicles(self, road, half_width, v_ego):
        if self.scenes is None:
            return super()._vehicles(road, half_width, v_ego)
        spec = self.spec
        scene = self.scenes[int(torch.randint(len(self.scenes), (), generator=self.g))]
        out = []
        for lab in scene:
            n = spec.n_doppler
            v_rel = ((lab["doppler_bin"] + n / 2) % n - n / 2) * spec.velocity_res_mps
            az = math.atan2(lab["x"], lab["y"])
            vy = v_ego + v_rel / max(math.cos(az), 0.2)
            vy = vy if abs(vy) > 1e-3 else 1e-3                 # a labelled vehicle moves
            out.append(self._car(lab["x"], lab["y"] + self.sc.car_length_m / 2, self.sc.car_rcs_dbsm, vy=vy))
        return out

    def gen_sequence(self):
        o = super().gen_sequence()
        if self.range_gain_db is not None:
            o["x"] = o["x"] + self.range_gain_db[None, :, None]
        return o


SIMULATORS = {"carrada": TunedCarrada, "raddet": TunedRaddet, "radial": TunedRadial}


def simulator(dataset, switches, generator, device=None, tables=None, mixin=None, target_db=0.0,
              static_db=0.0, **damage):
    """A tuned simulator with the given switches on; tables = table(dataset) (read if None).
    switches: subset of {processing, range_gain, scenes, label, contrast, body, clutter_profile}.
    mixin: a damage mixin of src/damage.py, given **damage; target_db / static_db move the
    targets / static scene further than the tuned offsets."""
    tb = table(dataset) if tables is None else tables
    cls = SIMULATORS[dataset] if mixin is None else type(mixin.__name__ + dataset, (mixin, SIMULATORS[dataset]), {})
    kw = {"generator": generator, **damage}
    if device is not None:
        kw["device"] = device
    if "processing" in switches and dataset != "radial":
        kw["processing"] = True
    if "range_gain" in switches:
        kw["range_gain_db"] = tb["range_gain_db"]
    if "scenes" in switches:
        kw["scenes"] = tb["scenes"]
    if "label" in switches and dataset != "radial":
        kw["label"] = "box"
        kw["label_min_snr_db"] = tb.get("label_min_snr_db")
    if "body" in switches and dataset == "radial":
        kw["body_db"] = tb["body_db"]
    if "clutter_profile" in switches:
        kw["clutter_profile_db"] = tb["clutter_profile_db"]
    t, st = (tb["contrast"]["target_db"], tb["contrast"]["static_db"]) if "contrast" in switches else (0.0, 0.0)
    if "contrast" in switches and tb.get("target_snr_db") is not None:
        kw["target_snr_db"] = tuple(tb["target_snr_db"])
    kw["target_offset_db"], kw["static_offset_db"] = t + target_db, st + static_db
    return cls(**kw)
