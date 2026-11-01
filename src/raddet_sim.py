"""RADDet-style range-Doppler scenes: the CARRADA simulator with RADDet's radar
and a street scene. Nothing is fitted to the recordings.

- Radar: the same chip as CARRADA (TI AWR1843, 2 Tx x 4 Rx), static on a
  tripod; grid from the RADDet README: 256 range bins of 0.195 m (row 0 near,
  as src/raddet.py flips the real maps), 64 Doppler bins of 0.42 m/s, zero at
  bin 32, approaching objects above it. Frames 0.15 s apart (measured from
  label motion; it only sets how far objects move between frames).
- Moving objects: RADDet's 6 classes in the proportions of its published
  counts (README: car 17,004, person 6,506, truck 3,787, bicycle 934, bus 217,
  motorcycle 88; 2.8 objects per frame on average, at most 7 cars and 7 people
  in a frame), 3-46 m (its published range histograms). Person, bicycle and
  car are CARRADA's models; a motorcycle is the cyclist model, a truck and a
  bus are longer car outlines.
- Street (the dataset's scenes are sidewalks and traffic lights): ground, poles
  and signs, parked cars and a building facade on each side.
Assumptions not in any source: tripod height 1.2 m (photo in the README),
speeds, truck and bus 20 dBsm and sizes, street layout and clutter strengths.
"""
import dataclasses
import math

import torch

from src.carrada_sim import CarradaSimulator, CarradaSpec, TrackScenario


@dataclasses.dataclass(frozen=True)
class RaddetSpec(CarradaSpec):
    range_res_m: float = 0.1953125
    velocity_res_mps: float = 0.4197
    v_max_mps: float = 32 * 0.4197
    frame_interval_s: float = 0.15


CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")
COUNTS = (6506, 934, 17004, 88, 217, 3787)                # README, whole dataset
MODEL = {"person": 1, "bicycle": 2, "motorcycle": 2, "car": 3, "bus": 3, "truck": 3}
SPEED = {"person": (0.5, 2.0), "bicycle": (3.0, 7.0), "motorcycle": (3.0, 13.0),
         "car": (2.0, 13.0), "bus": (2.0, 11.0), "truck": (2.0, 11.0)}
SIZE = {"car": (4.5, 1.8, 10.0), "truck": (8.0, 2.5, 20.0), "bus": (12.0, 2.5, 20.0)}   # m, m, dBsm


@dataclasses.dataclass(frozen=True)
class StreetScenario(TrackScenario):
    radar_height_m: float = 1.2
    objects_mean: float = 28536 / 10158                    # README totals / frames
    poles_mean: float = 6.0
    parked_mean: float = 3.0
    facade_offset_m: tuple = (6.0, 15.0)                   # each side of the radar
    facade_rcs_dbsm: tuple = (-5.0, 6.0)                   # per scatterer, 4 per metre


class RaddetSimulator(CarradaSimulator):
    def __init__(self, spec=RaddetSpec(), scenario=StreetScenario(), **kw):
        super().__init__(spec, scenario, **kw)

    def _static(self):
        """Ground and poles (the CARRADA scene), plus parked cars and two facades."""
        sc, rmax = self.sc, self.spec.max_range_m
        gx, gy, grcs = super()._static()
        xs, ys, rs = [gx], [gy], [grcs]
        for _ in range(int(torch.poisson(torch.tensor(sc.parked_mean), generator=self.g))):
            car = self._car_outline(0.0)
            x0, y0 = float(self._u(-12.0, 12.0)), float(self._u(3.0, rmax - 3))
            w = torch.ones_like(car["ox"])
            xs.append(x0 + car["ox"]); ys.append(y0 + car["oy"])
            rs.append(SIZE["car"][2] + 10 * torch.log10(w / w.sum()) + car["scint"])
        for side in (-1.0, 1.0):
            n = int(4 * rmax)
            xs.append(torch.full((n,), side * float(self._u(*sc.facade_offset_m))))
            ys.append(torch.linspace(1.0, rmax, n))
            rs.append(self._n(*sc.facade_rcs_dbsm, (n,)))
        return torch.cat(xs), torch.cat(ys), torch.cat(rs)

    def _objects(self):
        n = min(int(torch.poisson(torch.tensor(self.sc.objects_mean), generator=self.g)), 8)
        p = torch.tensor(COUNTS, dtype=torch.float)
        names = [CLASSES[int(torch.multinomial(p, 1, generator=self.g))] for _ in range(n)]
        return [self._street_object(name, i) for i, name in enumerate(names)]

    def _street_object(self, name, oid):
        """A CARRADA object of the class's model, with RADDet's speeds, ranges and sizes."""
        speed = float(self._u(*SPEED[name]))
        r0, az0 = float(self._u(3.0, 46.0)), math.radians(float(self._u(-60.0, 60.0)))
        x0, y0 = r0 * math.sin(az0), r0 * math.cos(az0)
        kind = int(torch.randint(3, (), generator=self.g))       # approach, recede, cross
        los = math.atan2(x0, y0)
        heading = {0: los + math.pi, 1: los, 2: los + math.pi / 2 * (1 if self._u(0, 1) < 0.5 else -1)}[kind]
        heading += math.radians(float(self._n(0.0, 10.0)))
        o = {"cls": MODEL[name], "name": name, "id": oid, "x0": x0, "y0": y0,
             "vx": speed * math.sin(heading), "vy": speed * math.cos(heading), "speed": speed,
             "phase": float(self._u(0, 2 * math.pi))}
        if name in SIZE:
            length, width, o["rcs_dbsm"] = SIZE[name]
            o["car"] = self._car_outline(heading, length, width)
        return o
