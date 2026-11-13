"""Optional read-only CARLA snapshot bridge.

CARLA's public actor API exposes bounding boxes, not the original RF material
meshes used by the RadSimReal authors. This bridge exports those boxes as
triangle geometry and labels that limitation. Use OBJ/NPZ imports for real CAD
surfaces. Importing this module never imports CARLA or connects to a server.
"""
from __future__ import annotations

from fnmatch import fnmatch
import importlib
import json
from pathlib import Path
import numpy as np

from .materials import SurfaceMaterial
from .mesh import MeshScene, save_mesh


def _triangulate_box(vertices):
    """Convex hull avoids depending on a CARLA release's vertex enumeration."""
    from scipy.spatial import ConvexHull
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (8, 3) or not np.isfinite(vertices).all():
        raise ValueError("CARLA bounding box needs eight finite world vertices")
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    triangle = vertices[faces]
    normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    outward = triangle.mean(axis=1) - vertices.mean(axis=0)
    reverse = np.einsum("ij,ij->i", normal, outward) < 0
    faces[reverse] = faces[reverse][:, [0, 2, 1]]
    return faces


def boxes_to_mesh(boxes, *, sensor_position=(0, 0, 0), sensor_rotation=None,
                  sensor_velocity=(0, 0, 0), material=None, metadata=None):
    """Convert explicit world-box records to MeshScene, useful for offline export.

    Each record supplies vertices (8,3), object_id, velocity (3,), optional
    type_id. This is a bounding-box bridge, not a claim of CARLA CAD recovery.
    """
    vertices, faces, velocities, object_ids, actors = [], [], [], [], []
    for record in boxes:
        box = np.asarray(record["vertices"], dtype=float)
        triangles = _triangulate_box(box)
        velocity = np.asarray(record.get("velocity", [0, 0, 0]), dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("box velocity must be finite (3,)")
        faces.extend(triangles + len(vertices))
        vertices.extend(box)
        velocities.extend([velocity] * len(triangles))
        object_ids.extend([int(record["object_id"])] * len(triangles))
        actors.append({"object_id": int(record["object_id"]), "type_id": record.get("type_id", "unknown"),
                       "geometry": "actor/level bounding box"})
    meta = dict(metadata or {})
    meta.update({"geometry_source": "CARLA public bounding-box snapshot bridge",
                 "mesh_fidelity": "triangulated bounding boxes, not visual/RF CAD surfaces",
                 "actors": actors, "world_modified": False})
    return MeshScene(np.asarray(vertices).reshape(-1, 3), np.asarray(faces, dtype=np.int64).reshape(-1, 3),
                     object_ids=np.asarray(object_ids, dtype=np.int64),
                     face_velocities=np.asarray(velocities).reshape(-1, 3),
                     materials={"carla_uncalibrated": SurfaceMaterial() if material is None else material},
                     sensor_position=sensor_position, sensor_rotation=np.eye(3) if sensor_rotation is None else sensor_rotation,
                     sensor_velocity=sensor_velocity, metadata=meta)


def snapshot_from_carla(host="localhost", port=2000, timeout=5.0, sensor_actor_id=None,
                        actor_patterns=("vehicle.*", "walker.pedestrian.*"),
                        max_distance=100.0, include_static_labels=(), material=None):
    """Connect lazily and export one snapshot without tick/spawn/settings mutation.

    With sensor_actor_id, its local x-forward/y-right CARLA basis is converted
    into this project's x-right/y-forward basis. The selected actor is omitted
    as an occluder; supply separate ego-body geometry if that effect is needed.
    Snapshot actor transforms/velocities are preferred for temporal consistency.
    Static level boxes are optional and depend on the installed CARLA version.
    """
    if not isinstance(port, int) or not 1 <= port <= 65535 or timeout <= 0 or max_distance <= 0:
        raise ValueError("invalid server port, timeout or maximum distance")
    try:
        carla = importlib.import_module("carla")
    except ImportError as error:
        raise RuntimeError("CARLA client is optional. Install the matching client on the CARLA host, or import OBJ/NPZ geometry.") from error
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()
    snapshot = world.get_snapshot()
    all_actors = world.get_actors()
    sensor_position = np.zeros(3)
    sensor_velocity = np.zeros(3)
    # CARLA world coordinates use x forward/y right. This orthogonal basis
    # conversion is allowed to reverse handedness without distorting lengths.
    canonical_to_carla = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    sensor_rotation = canonical_to_carla.copy()
    if sensor_actor_id is not None:
        sensor_actor = world.get_actor(int(sensor_actor_id))
        if sensor_actor is None:
            raise ValueError(f"CARLA actor {sensor_actor_id} does not exist")
        state = snapshot.find(sensor_actor.id)
        transform = state.get_transform() if state is not None else sensor_actor.get_transform()
        matrix = np.asarray(transform.get_matrix(), dtype=float)
        sensor_position = matrix[:3, 3]
        sensor_rotation = matrix[:3, :3] @ canonical_to_carla
        v = state.get_velocity() if state is not None else sensor_actor.get_velocity()
        sensor_velocity = np.array([v.x, v.y, v.z])
    boxes, missing_states = [], 0
    for actor in all_actors:
        if actor.id == sensor_actor_id or not any(fnmatch(actor.type_id, pattern) for pattern in actor_patterns):
            continue
        if not hasattr(actor, "bounding_box"):
            continue
        state = snapshot.find(actor.id)
        if state is None:
            missing_states += 1
        transform = state.get_transform() if state is not None else actor.get_transform()
        location = transform.location
        if np.linalg.norm(np.array([location.x, location.y, location.z]) - sensor_position) > max_distance:
            continue
        velocity = state.get_velocity() if state is not None else actor.get_velocity()
        vertices = actor.bounding_box.get_world_vertices(transform)
        boxes.append({"vertices": [[p.x, p.y, p.z] for p in vertices], "object_id": actor.id,
                      "velocity": [velocity.x, velocity.y, velocity.z], "type_id": actor.type_id})
    static_id = -2
    for name in include_static_labels:
        try:
            label = getattr(carla.CityObjectLabel, name)
        except AttributeError as error:
            raise ValueError(f"CARLA release has no CityObjectLabel.{name}") from error
        for box in world.get_level_bbs(label):
            location = box.location
            if np.linalg.norm(np.array([location.x, location.y, location.z]) - sensor_position) > max_distance:
                continue
            vertices = box.get_world_vertices(carla.Transform())
            boxes.append({"vertices": [[p.x, p.y, p.z] for p in vertices], "object_id": static_id,
                          "velocity": [0, 0, 0], "type_id": f"static.{name}"})
            static_id -= 1
    timestamp = getattr(snapshot, "timestamp", None)
    meta = {"carla_frame": int(snapshot.frame),
            "simulation_elapsed_seconds": None if timestamp is None else float(timestamp.elapsed_seconds),
            "snapshot_state_fallback_count": missing_states,
            "source_axes": "CARLA x-forward/y-right/z-up; converted into canonical radar-local coordinates",
            "sensor_actor_id": sensor_actor_id, "maximum_actor_distance_m": float(max_distance),
            "static_labels": list(include_static_labels),
            "material_source": "user-supplied RF material or uncalibrated default"}
    return boxes_to_mesh(boxes, sensor_position=sensor_position, sensor_rotation=sensor_rotation,
                         sensor_velocity=sensor_velocity, material=material, metadata=meta)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Export one read-only CARLA bounding-box snapshot as RF mesh NPZ")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("--sensor-actor-id", type=int)
    parser.add_argument("--max-distance", type=float, default=100)
    parser.add_argument("--static-label", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    scene = snapshot_from_carla(args.host, args.port, args.timeout, args.sensor_actor_id,
                                 max_distance=args.max_distance, include_static_labels=args.static_label)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_mesh(output, scene)
    output.with_suffix(".json").write_text(json.dumps(scene.metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(scene.faces)} triangles from a read-only snapshot to {output}")


if __name__ == "__main__":
    main()
