"""Standalone procedural scenes: an explicit substitute for CARLA in Fig. 2(a).

Vehicles are rigid extruded rectangles. These geometries and their distribution
are illustrative and do not recreate the authors' undisclosed CARLA assets,
routes, materials, scene probabilities, or RF ray-tracing implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import numpy as np


@dataclass
class Vehicle:
    x: float
    y: float
    length: float
    width: float
    yaw: float
    vx: float
    vy: float
    label: str = "car"
    object_id: int = 0

    def corners(self):
        """Return four BEV corners, yaw measured clockwise from forward."""
        local = np.asarray([[-self.width / 2, -self.length / 2],
                            [self.width / 2, -self.length / 2],
                            [self.width / 2, self.length / 2],
                            [-self.width / 2, self.length / 2]])
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rotation = np.asarray([[c, s], [-s, c]])
        return local @ rotation.T + [self.x, self.y]


@dataclass
class Scene:
    objects: list[Vehicle]
    ego_velocity: tuple[float, float] = (0.0, 0.0)
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {"objects": [asdict(obj) for obj in self.objects],
                "ego_velocity": list(self.ego_velocity), "metadata": self.metadata}

    @classmethod
    def from_dict(cls, payload):
        return cls(objects=[Vehicle(**obj) for obj in payload["objects"]],
                   ego_velocity=tuple(payload.get("ego_velocity", (0.0, 0.0))),
                   metadata=payload.get("metadata", {}))


def sample_scene(radar_config, scene_config, rng):
    """Sample non-overlapping vehicle boxes with randomized heading and motion.

    Separation uses circumscribed circles (conservative), with bounded retries.
    Empty scenes are possible when configured min_vehicles=0. A crowded domain
    can produce fewer than max_vehicles; actual count is recorded.
    """
    if scene_config.min_vehicles < 0 or scene_config.max_vehicles < scene_config.min_vehicles:
        raise ValueError("invalid vehicle-count interval")
    requested = int(rng.integers(scene_config.min_vehicles, scene_config.max_vehicles + 1))
    min_range = max(float(scene_config.min_range), radar_config.range_min + 3.5)
    max_range = min(float(scene_config.max_range), radar_config.ranges[-1] - 3.5)
    if min_range >= max_range:
        raise ValueError("scene radial interval does not fit inside radar range with box margin")
    objects = []
    if not radar_config.azimuth_min_deg < 0 < radar_config.azimuth_max_deg:
        raise ValueError("procedural scene sampler expects azimuth domain spanning zero")
    angular_limit = min(abs(radar_config.azimuth_min_deg), abs(radar_config.azimuth_max_deg))
    for _ in range(max(100, requested * 100)):
        if len(objects) >= requested:
            break
        distance = float(rng.uniform(min_range, max_range))
        lateral_max = min(scene_config.lateral_limit, distance * np.sin(np.deg2rad(angular_limit * 0.85)))
        x = float(rng.uniform(-lateral_max, lateral_max))
        y = float(np.sqrt(max(distance ** 2 - x ** 2, 0)))
        length = float(rng.uniform(*scene_config.vehicle_length))
        width = float(rng.uniform(*scene_config.vehicle_width))
        radius = np.hypot(length, width) / 2
        if any(np.hypot(x - obj.x, y - obj.y) < radius + np.hypot(obj.length, obj.width) / 2 + 0.5
               for obj in objects):
            continue
        yaw = float(rng.uniform(-np.pi, np.pi))
        speed = float(rng.uniform(*scene_config.speed_range))
        if rng.random() < 0.25:
            speed = 0.0
        objects.append(Vehicle(x, y, length, width, yaw, speed * np.sin(yaw), speed * np.cos(yaw),
                               object_id=len(objects)))
    if len(objects) < scene_config.min_vehicles:
        raise RuntimeError("could not place minimum vehicle count; enlarge domain or lower density")
    return Scene(objects, tuple(scene_config.ego_velocity), {
        "source": "procedural extruded rectangles; approximation to CARLA environment stage",
        "requested_vehicle_count": requested, "actual_vehicle_count": len(objects),
        "units": "metres, seconds, radians", "coordinates": "x right, y forward, z up",
    })
