"""Approximate single-bounce RF surface points and external-engine interchange.

Section 3 leaves RF materials/scattering formulas unspecified, referring to a
radar textbook. This file makes its replacement explicit: visible box faces,
line-of-sight occlusion, diffuse/specular angular mixture, integrated patch-area
weights, exponential power speckle and two-way r^-4 propagation. These arbitrary
units require calibration and are not the authors' undisclosed ray tracer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import numpy as np


@dataclass
class ReflectionPoints:
    xyz: np.ndarray
    velocity: np.ndarray
    coefficients: np.ndarray
    object_ids: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.xyz = np.asarray(self.xyz, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        self.coefficients = np.asarray(self.coefficients)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or self.velocity.shape != self.xyz.shape:
            raise ValueError("xyz and velocity must have shape (N,3)")
        if self.coefficients.shape != (len(self.xyz),):
            raise ValueError("coefficients must have shape (N,)")
        if not all(np.isfinite(a).all() for a in (self.xyz, self.velocity, self.coefficients)):
            raise ValueError("reflection arrays must contain only finite values")
        if self.object_ids is None:
            self.object_ids = np.full(len(self.xyz), -1, dtype=np.int64)
        else:
            self.object_ids = np.asarray(self.object_ids, dtype=np.int64)
        if self.object_ids.shape != (len(self.xyz),):
            raise ValueError("object_ids must have shape (N,)")

    def __len__(self):
        return len(self.coefficients)


def _occluded(point, vehicle):
    """Ray-segment / oriented rectangle intersection before point (BEV prism)."""
    c, s = np.cos(vehicle.yaw), np.sin(vehicle.yaw)
    world_to_local = np.asarray([[c, -s], [s, c]])
    origin = world_to_local @ -np.asarray([vehicle.x, vehicle.y])
    direction = world_to_local @ point[:2]
    half = np.asarray([vehicle.width / 2, vehicle.length / 2])
    t_min, t_max = 0.0, 1.0 - 1e-5
    for axis in range(2):
        if abs(direction[axis]) < 1e-12:
            if abs(origin[axis]) > half[axis]:
                return False
        else:
            a = (-half[axis] - origin[axis]) / direction[axis]
            b = (half[axis] - origin[axis]) / direction[axis]
            t_min = max(t_min, min(a, b))
            t_max = min(t_max, max(a, b))
            if t_min > t_max:
                return False
    return t_max >= max(0.0, t_min)


def generate_reflections(scene, radar_config, scene_config, rng):
    """Generate visible-face samples with integrated area, r^-4 and speckle.

    Object velocities are world-frame in Scene; output velocities are relative
    to radar ego motion. Field mode converts each patch power into sqrt(power)
    with an independent random phase. That is a rough-surface approximation,
    not phase-accurate multipath or material electromagnetics.
    """
    count = int(scene_config.points_per_vehicle)
    if count < 1 or scene_config.clutter_points < 0:
        raise ValueError("positive points_per_vehicle and nonnegative clutter_points required")
    roughness = float(scene_config.surface_roughness)
    if not 0 <= roughness <= 1:
        raise ValueError("surface_roughness must be in [0,1]")
    if radar_config.scattering_gain < 0 or radar_config.reference_range <= 0:
        raise ValueError("nonnegative scattering_gain and positive reference_range required")
    positions, velocities, powers, object_ids = [], [], [], []
    ego = np.asarray([*scene.ego_velocity, 0.0])
    hidden = 0
    for obj in scene.objects:
        corners = obj.corners()
        edges = []
        for start, end in zip(corners, np.roll(corners, -1, axis=0)):
            midpoint = (start + end) / 2
            normal = midpoint - [obj.x, obj.y]
            normal /= np.linalg.norm(normal)
            facing = float(np.dot(normal, -midpoint) / np.linalg.norm(midpoint))
            if facing > 0:
                edges.append((start, end, normal, np.linalg.norm(end - start)))
        if not edges:
            continue
        lengths = np.asarray([edge[3] for edge in edges])
        edge_ids = rng.choice(len(edges), size=count, p=lengths / lengths.sum())
        patch_area = float(lengths.sum() * 1.4 / count)
        for edge_id in edge_ids:
            start, end, normal, _ = edges[edge_id]
            xy = start + float(rng.random()) * (end - start)
            point = np.asarray([xy[0], xy[1], float(rng.uniform(0.2, 1.6))])
            if any(other.object_id != obj.object_id and _occluded(point, other) for other in scene.objects):
                hidden += 1
                continue
            distance = float(np.linalg.norm(point))
            incidence = max(0.0, float(np.dot(normal, -xy) / np.linalg.norm(xy)))
            angular_response = roughness * incidence + (1 - roughness) * incidence ** 8
            # Relative monostatic power law; no claim of calibrated absolute RCS.
            power = (radar_config.scattering_gain * patch_area * angular_response
                     * (radar_config.reference_range / max(distance, 1.0)) ** 4
                     * float(rng.exponential()))
            positions.append(point)
            velocities.append(np.asarray([obj.vx, obj.vy, 0.0]) - ego)
            powers.append(power)
            object_ids.append(obj.object_id)
    for _ in range(int(scene_config.clutter_points)):
        distance = float(rng.uniform(max(scene_config.min_range, 1.0), min(scene_config.max_range, radar_config.ranges[-1])))
        angle = float(rng.uniform(np.deg2rad(radar_config.azimuth_min_deg), np.deg2rad(radar_config.azimuth_max_deg)))
        point = np.asarray([distance * np.sin(angle), distance * np.cos(angle), 0.0])
        if any(_occluded(point, obj) for obj in scene.objects):
            hidden += 1
            continue
        positions.append(point)
        velocities.append(-ego)
        powers.append(float(rng.exponential(0.01 * radar_config.scattering_gain))
                      * (radar_config.reference_range / max(distance, 1.0)) ** 4)
        object_ids.append(-1)
    powers = np.asarray(powers, dtype=float)
    coefficients = powers
    if radar_config.representation in ("field", "complex"):
        coefficients = np.sqrt(powers) * np.exp(2j * np.pi * rng.random(len(powers)))
    return ReflectionPoints(np.asarray(positions).reshape(-1, 3), np.asarray(velocities).reshape(-1, 3),
                            coefficients, np.asarray(object_ids), metadata={
        "source": "approximate visible-box-face single-bounce model",
        "propagation": "relative power proportional to reference_range**4 / range**4",
        "scattering": "patch-area-weighted diffuse/specular mixture with exponential power speckle",
        "clutter": "optional sparse static ground point approximation; not calibrated urban clutter",
        "occluded_samples": hidden, "velocity_frame": "radar-relative",
        "missing_physics": ["CARLA meshes", "material electromagnetic calibration", "multipath", "diffraction", "weather"],
        "calibrated": False,
    })


def save_reflections(path, reflections):
    np.savez_compressed(path, xyz=reflections.xyz, velocity=reflections.velocity,
                        coefficients=reflections.coefficients, object_ids=reflections.object_ids,
                        metadata=json.dumps(reflections.metadata))


def load_reflections(path):
    """Load external RF-ray-tracer / calibrated LIDAR-surface coefficients.

    Keys: xyz or positions (N,3) metres; velocity or velocities (N,3) m/s already
    relative to radar; coefficients (N,) in the SAME linear response convention
    as the PSF; optional object_ids. Coefficients must ALREADY include path loss,
    material response and, for field mode, phase. Raw LIDAR intensity is not RF
    reflectivity and must never be passed as a substitute without calibration.
    """
    with np.load(path, allow_pickle=False) as archive:
        xyz = archive["xyz"] if "xyz" in archive else archive["positions"]
        velocity = archive["velocity"] if "velocity" in archive else archive["velocities"]
        ids = archive["object_ids"] if "object_ids" in archive else None
        metadata = json.loads(str(archive["metadata"])) if "metadata" in archive else {}
        metadata.update({"source": "external reflection point file", "velocity_frame": "radar-relative",
                         "coefficient_requirement": "already includes material response and propagation"})
        return ReflectionPoints(xyz, velocity, archive["coefficients"], ids, metadata)
