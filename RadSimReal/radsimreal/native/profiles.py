"""Declared native-array conventions; reference defaults require local validation.

Sampling spacings are not resolving power. Array shape never determines a
real-valued file's signal representation. All internal tensors are R,D,A and
all internal velocities are positive receding, in m/s.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
import json
import numpy as np


@dataclass
class NativeProfile:
    name: str
    shape_native: tuple = (256, 256, 64)
    representation: str | None = None
    range_step: float = 0.1953125
    range_min: float = 0.0
    velocity_step: float = 0.41968030701528203
    velocity_min_native: float | None = None
    range_reversed: bool = True
    doppler_reversed: bool = False
    doppler_positive: str = "receding"
    azimuth_reversed: bool = False
    azimuth_coordinate: str = "sin"
    angular_min: float | None = None
    angular_step: float | None = None
    cartesian_affine: dict | None = None
    cartesian_mapping_validated: bool = False
    car_labels: tuple = ("car",)
    annotation_range_reverse_origin: str = "R-1"
    annotation_angle_step: float | None = None
    evidence: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)

    def __post_init__(self):
        self.name = self.name.lower()
        if self.name not in ("raddet", "carrada"):
            raise ValueError("native dataset must be raddet or carrada")
        self.shape_native = tuple(self.shape_native)
        if len(self.shape_native) != 3 or any(type(n) is not int or n < 2 for n in self.shape_native):
            raise ValueError("shape_native must contain three integers >=2 in R,A,D order")
        if self.representation not in (None, "complex", "power", "amplitude", "db"):
            raise ValueError("representation must be complex, power, amplitude or db")
        if not np.isfinite([self.range_step, self.range_min, self.velocity_step]).all() or self.range_step <= 0 or self.velocity_step <= 0 or self.range_min < 0:
            raise ValueError("range/velocity spacing must be positive and range_min nonnegative")
        if self.velocity_min_native is None:
            self.velocity_min_native = -(self.shape_native[2] // 2) * self.velocity_step
        if not np.isfinite(self.velocity_min_native):
            raise ValueError("velocity_min_native must be finite")
        if self.doppler_positive not in ("approaching", "receding"):
            raise ValueError("doppler_positive must be approaching or receding")
        for key in ("range_reversed", "doppler_reversed", "azimuth_reversed", "cartesian_mapping_validated"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be a boolean")
        if self.azimuth_coordinate not in ("sin", "angle"):
            raise ValueError("azimuth_coordinate must be sin or angle")
        a = self.shape_native[1]
        if self.angular_min is None or self.angular_step is None:
            if self.name == "raddet" and self.azimuth_coordinate == "sin":
                ratio = 76.8 / 77.0
                default_min, default_step = -ratio, 2 * ratio / a
            else:
                default_min, default_step = -np.pi / 2, np.pi / a
            if self.angular_min is None:
                self.angular_min = default_min
            if self.angular_step is None:
                self.angular_step = default_step
        if not np.isfinite([self.angular_min, self.angular_step]).all() or self.angular_step <= 0:
            raise ValueError("Angular coordinate spacing must be positive and finite")
        last = self.angular_min + (a - 1) * self.angular_step
        bound = 1.0 if self.azimuth_coordinate == "sin" else np.pi / 2
        if self.angular_min < -bound - 1e-12 or last > bound + 1e-12:
            raise ValueError("Angular coordinate is outside the forward-hemisphere domain")
        self.car_labels = tuple(self.car_labels)
        if self.annotation_range_reverse_origin not in ("R", "R-1"):
            raise ValueError("annotation_range_reverse_origin must be R or R-1")
        if self.annotation_angle_step is not None and (not np.isfinite(self.annotation_angle_step) or self.annotation_angle_step <= 0):
            raise ValueError("annotation_angle_step must be positive")
        if self.cartesian_affine is not None:
            required = {"x_from_col_scale", "x_from_col_offset", "y_from_row_scale", "y_from_row_offset"}
            if set(self.cartesian_affine) != required or not np.isfinite(list(self.cartesian_affine.values())).all():
                raise ValueError("cartesian_affine requires exactly four finite scale/offset fields")
            if self.cartesian_affine["x_from_col_scale"] == 0 or self.cartesian_affine["y_from_row_scale"] == 0:
                raise ValueError("Cartesian affine scales must be nonzero")

    @property
    def shape(self):
        r, a, d = self.shape_native
        return r, d, a

    @property
    def angular_grid(self):
        return self.angular_min + self.angular_step * np.arange(self.shape_native[1])

    @property
    def azimuths_rad(self):
        return np.arcsin(np.clip(self.angular_grid, -1, 1)) if self.azimuth_coordinate == "sin" else self.angular_grid

    @property
    def ranges(self):
        return self.range_min + self.range_step * np.arange(self.shape_native[0])

    @property
    def velocities(self):
        values = self.velocity_min_native + self.velocity_step * np.arange(self.shape_native[2])
        if self.doppler_positive == "approaching":
            values = -values
        return np.sort(values)

    @property
    def canonical_doppler_flip(self):
        return bool(self.doppler_reversed) ^ (self.doppler_positive == "approaching")

    def radar_config(self):
        from ..config import RadarConfig
        return RadarConfig(range_bins=self.shape_native[0], doppler_bins=self.shape_native[2],
                           azimuth_bins=self.shape_native[1], range_min=self.range_min,
                           range_step=self.range_step, velocity_min=float(self.velocities[0]),
                           velocity_step=self.velocity_step,
                           azimuth_min_deg=float(np.rad2deg(self.azimuths_rad[0])),
                           azimuth_max_deg=float(np.rad2deg(self.azimuths_rad[-1])),
                           azimuth_coordinate=self.azimuth_coordinate,
                           representation="complex" if self.representation == "complex" else "intensity")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("native profile must be a mapping")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown native profile fields: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            import yaml
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def reference_profile(dataset, shape_native=(256, 256, 64), representation=None):
    """Source-derived defaults, not a claim of validation on a local release.

    RADDet range reversal follows its inference.py -> toCartesianMask path.
    Its Doppler physical sign is an explicit unresolved reference convention.
    CARRADA annotation sign is positive approaching. Its tensor representation
    cannot be recovered from a folder name; real arrays require a declaration.
    """
    dataset = dataset.lower()
    if dataset == "raddet":
        return NativeProfile(name=dataset, shape_native=shape_native, representation=representation,
            evidence=["RADDet util/helper.py toCartesianMask, inference.py and config.json; reference 2026-11-10"],
            unresolved=["Confirm range/angle overlays on the local release", "Confirm physical Doppler sign",
                        "No Cartesian metre affine is enabled until explicitly supplied", "Sensor PSF/noise are uncalibrated"])
    if dataset == "carrada":
        return NativeProfile(name=dataset, shape_native=shape_native, representation=representation,
            doppler_positive="approaching", azimuth_coordinate="angle", car_labels=("car", 3),
            annotation_range_reverse_origin="R", annotation_angle_step=np.pi / 256,
            evidence=["CARRADA transform_annotations.py and paper; source-derived reference, 2026-11-10"],
            unresolved=["Confirm full-tensor representation, range order and angular coordinate against the local RAD release",
                        "Native radar annotation regions are not physical oriented car footprints", "Sensor PSF/noise are uncalibrated"])
    raise ValueError("dataset must be raddet or carrada")


def raddet_reference_cartesian_affine(profile):
    """Explicit helper-derived pixel-to-metre affine; validate with a real overlay.

    helper.toCartesianMask uses new_row=R-round(forward/dr)-1 and
    new_col=round((lateral+50)/dr)-1. This inverse is continuous; original
    rounding is not undone. Enabling it is a declared choice, never automatic.
    """
    if profile.name != "raddet":
        raise ValueError("Reference Cartesian affine only applies to RADDet")
    dr = profile.range_step
    return {"x_from_col_scale": dr, "x_from_col_offset": dr - 50.0,
            "y_from_row_scale": -dr, "y_from_row_offset": (profile.shape_native[0] - 1) * dr}
