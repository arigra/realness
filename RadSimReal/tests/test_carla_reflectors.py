"""CARLA capture -> reflection conversion on a synthetic capture (no CARLA server needed)."""
import numpy as np
from radsimreal.config import RadarConfig
from radsimreal.bev.geometry import BEVGrid
from radsimreal.radar.coordinates import points_to_indices
from radsimreal.simulation.carla_reflectors import capture_to_reflections, vehicle_objects, TAG_MATERIAL


def _capture():
    # a car (tag 14, actor 7) at 10 m ahead, a building (tag 3) at 30 m, sky (tag 11) ignored
    xyz = np.array([[0.0, 10.0, 0.0], [0.5, 10.2, 0.3], [5.0, 30.0, 2.0], [0.0, 40.0, 5.0]], np.float32)
    return {"xyz": xyz, "cos_inc": np.array([1.0, 0.9, 0.8, 1.0], np.float32),
            "tag": np.array([14, 14, 3, 11], np.uint8), "object_idx": np.array([7, 7, 3, 0]),
            "velocity": np.array([[0, -5, 0], [0, -5, 0], [0, 0, 0], [0, 0, 0]], np.float32)}


def test_conversion_physics_and_labels():
    radar = RadarConfig(range_bins=256, doppler_bins=64, azimuth_bins=256, range_step=0.1953125,
                        velocity_min=-13.01, velocity_step=0.41968, azimuth_coordinate="sin",
                        representation="complex", scattering_gain=1.0, reference_range=20.0)
    refl = capture_to_reflections(_capture(), 1e-5, radar, np.random.default_rng(0))
    assert len(refl) == 3 and TAG_MATERIAL[11] is None            # sky dropped
    assert np.all(refl.object_ids[:2] == 7)
    p = np.abs(refl.coefficients) ** 2
    assert p[0] > p[2]                                             # metal at 10 m >> concrete at 30 m
    idx, valid = points_to_indices(refl.xyz, refl.velocity, radar)
    assert valid.all()
    assert abs(idx[0, 0] - 10 / 0.1953125) < 1e-6                  # range bin
    assert abs((radar.velocity_min + idx[0, 1] * radar.velocity_step) - (-5.0)) < 1e-3  # approaching = negative internal
    grid = BEVGrid(x_min=-50, x_max=50, y_min=0, y_max=50, height=256, width=512)
    meta = {"vehicles": [
        {"actor_id": 7, "type_id": "vehicle.x", "label_class": "car", "footprint_xyxy": [-1, 9, 1, 13], "velocity_sensor": [0, -5, 0]},
        {"actor_id": 8, "type_id": "vehicle.y", "label_class": "car", "footprint_xyxy": [-1, 20, 1, 24], "velocity_sensor": [0, 0, 0]}]}
    objects, excluded = vehicle_objects(meta, refl, radar, grid, min_points=2)
    assert [o["object_id"] for o in objects] == [7]                # actor 8 has no returns -> unlabelled
    meta["vehicles"][0]["footprint_xyxy"] = [-51, 9, -49, 13]
    objects, excluded = vehicle_objects(meta, refl, radar, grid, min_points=2)
    assert not objects and excluded[0]["actor_id"] == 7            # never clipped
