"""Validated configuration. Demo defaults are illustrative, not TI calibration."""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import json
import numpy as np


@dataclass
class RadarConfig:
    range_bins: int = 96
    doppler_bins: int = 16
    azimuth_bins: int = 96
    range_min: float = 0.0
    range_step: float = 0.5
    velocity_min: float = -16.0
    velocity_step: float = 2.0
    azimuth_min_deg: float = -60.0
    azimuth_max_deg: float = 60.0
    azimuth_coordinate: str = "sin"
    noise_std: float = 0.03
    psf_shape: tuple = (9, 5, 21)
    psf_width_bins: tuple = (1.3, 1.3, 3.0)
    psf_energy: float = 0.99
    representation: str = "intensity"
    scattering_gain: float = 50.0
    reference_range: float = 20.0
    psf_path: str | None = None
    noise_model: str = "gaussian"
    noise_mean_real: float = 0.0
    noise_mean_imag: float = 0.0
    noise_covariance: tuple | None = None
    noise_bank_path: str | None = None
    noise_spectrum_path: str | None = None
    doppler_wrap: bool = False
    sensor_height: float = 0.0
    frequency_hz: float = 77e9
    polarization: str = "TE"
    phase_mode: str = "random"

    def __post_init__(self):
        for key in ("range_bins", "doppler_bins", "azimuth_bins"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 2:
                raise ValueError(f"{key} must be an integer >= 2")
        for key in ("range_step", "velocity_step", "reference_range"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not -90 <= self.azimuth_min_deg < self.azimuth_max_deg <= 90:
            raise ValueError("azimuth limits must increase within [-90, 90] degrees")
        if self.range_min < 0 or not np.isfinite(self.range_min):
            raise ValueError("range_min must be finite and nonnegative")
        if self.azimuth_coordinate not in ("sin", "angle"):
            raise ValueError("azimuth_coordinate must be sin or angle")
        if self.representation not in ("intensity", "complex"):
            raise ValueError("representation must be intensity or complex")
        if not np.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        if self.noise_model not in ("gaussian", "empirical", "coloured_gaussian", "none"):
            raise ValueError("noise_model must be gaussian, empirical, coloured_gaussian or none")
        if not np.isfinite([self.sensor_height, self.frequency_hz]).all() or self.sensor_height < 0 or self.frequency_hz <= 0:
            raise ValueError("sensor_height must be nonnegative and frequency_hz positive, both finite")
        if self.polarization not in ("TE", "TM") or self.phase_mode not in ("random", "path"):
            raise ValueError("polarization must be TE/TM and phase_mode random/path")
        if not np.isfinite([self.noise_mean_real, self.noise_mean_imag]).all():
            raise ValueError("noise means must be finite")
        if self.representation == "intensity" and self.noise_mean_imag != 0:
            raise ValueError("real intensity response cannot have an imaginary noise mean")
        if self.noise_covariance is not None:
            cov = np.asarray(self.noise_covariance, dtype=float)
            if cov.shape != (2, 2) or not np.isfinite(cov).all() or not np.allclose(cov, cov.T) or np.linalg.eigvalsh(cov).min() < -1e-12:
                raise ValueError("noise_covariance must be a finite symmetric PSD 2x2 matrix")
            self.noise_covariance = tuple(tuple(float(v) for v in row) for row in cov)
        if not np.isfinite(self.scattering_gain) or self.scattering_gain < 0:
            raise ValueError("scattering_gain must be finite and nonnegative")
        if not np.isfinite(self.velocity_min):
            raise ValueError("velocity_min must be finite")
        self.psf_shape = tuple(self.psf_shape)
        self.psf_width_bins = tuple(self.psf_width_bins)
        if len(self.psf_shape) != 3 or any(not isinstance(n, int) or n < 1 or n % 2 != 1 for n in self.psf_shape):
            raise ValueError("psf_shape requires three positive odd integers")
        if len(self.psf_width_bins) != 3 or any(not np.isfinite(w) or w <= 0 for w in self.psf_width_bins):
            raise ValueError("psf_width_bins requires three positive finite widths")
        if not 0 < self.psf_energy <= 1:
            raise ValueError("psf_energy must be in (0, 1]")

    @property
    def shape(self):
        return self.range_bins, self.doppler_bins, self.azimuth_bins

    @property
    def ranges(self):
        return self.range_min + self.range_step * np.arange(self.range_bins)

    @property
    def velocities(self):
        return self.velocity_min + self.velocity_step * np.arange(self.doppler_bins)

    @property
    def angular_grid(self):
        limits = np.deg2rad([self.azimuth_min_deg, self.azimuth_max_deg])
        if self.azimuth_coordinate == "sin":
            limits = np.sin(limits)
        return np.linspace(*limits, self.azimuth_bins)

    @property
    def angular_step(self):
        return float(self.angular_grid[1] - self.angular_grid[0])

    @property
    def azimuths_rad(self):
        return np.arcsin(self.angular_grid) if self.azimuth_coordinate == "sin" else self.angular_grid


@dataclass
class SceneConfig:
    min_vehicles: int = 1
    max_vehicles: int = 5
    min_range: float = 7.0
    max_range: float = 43.0
    lateral_limit: float = 22.0
    vehicle_length: tuple = (3.6, 5.2)
    vehicle_width: tuple = (1.6, 2.2)
    speed_range: tuple = (-8.0, 8.0)
    points_per_vehicle: int = 64
    clutter_points: int = 30
    ego_velocity: tuple = (0.0, 0.0)
    surface_roughness: float = 0.3

    def __post_init__(self):
        for key in ("min_vehicles", "max_vehicles", "points_per_vehicle", "clutter_points"):
            if isinstance(getattr(self, key), bool) or not isinstance(getattr(self, key), int):
                raise ValueError(f"{key} must be an integer")
        if not 0 <= self.min_vehicles <= self.max_vehicles:
            raise ValueError("vehicle counts must be nonnegative ordered integers")
        if not np.isfinite([self.min_range, self.max_range, self.lateral_limit]).all() or not 0 < self.min_range < self.max_range or self.lateral_limit <= 0:
            raise ValueError("invalid scene extent")
        for key in ("vehicle_length", "vehicle_width", "speed_range"):
            vals = tuple(getattr(self, key))
            if len(vals) != 2 or not np.isfinite(vals).all() or vals[0] > vals[1]:
                raise ValueError(f"{key} requires two finite ordered endpoints")
            if key != "speed_range" and vals[0] <= 0:
                raise ValueError("vehicle dimensions must be positive")
            setattr(self, key, vals)
        if self.points_per_vehicle < 4 or self.clutter_points < 0:
            raise ValueError("need >=4 vehicle points and nonnegative clutter points")
        self.ego_velocity = tuple(self.ego_velocity)
        if len(self.ego_velocity) != 2 or not np.isfinite(self.ego_velocity).all():
            raise ValueError("ego_velocity requires finite lateral/forward components")
        if not 0 <= self.surface_roughness <= 1:
            raise ValueError("surface_roughness must be in [0, 1]")


@dataclass
class SimulationConfig:
    radar: RadarConfig = field(default_factory=RadarConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    seed: int = 2024

    def to_dict(self):
        return asdict(self)


def _strict(cls, data):
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} configuration must be a mapping")
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**data)


def load_config(path=None):
    if path is None:
        return SimulationConfig()
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) - {"radar", "scene", "seed"}:
        raise ValueError("Configuration requires only radar, scene and seed sections")
    radar = _strict(RadarConfig, data.get("radar", {}))
    for key in ("psf_path", "noise_bank_path", "noise_spectrum_path"):
        value = getattr(radar, key)
        if value and not Path(value).is_absolute():
            setattr(radar, key, str((path.parent / value).resolve()))
    return SimulationConfig(radar, _strict(SceneConfig, data.get("scene", {})), int(data.get("seed", 2024)))


def save_config(config, path):
    Path(path).write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

