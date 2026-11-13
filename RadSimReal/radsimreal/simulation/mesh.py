"""Triangle-mesh single-bounce RF scene backend for RadSimReal Fig. 2(a).

Visibility uses real ray/triangle intersections with a bounding-volume hierarchy.
Materials and rough-surface scattering are explicit, calibratable approximations.
The backend supports supplied CAD geometry without requiring CARLA or a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np

from .materials import LIGHT_SPEED_M_S, RadarLinkBudget, SurfaceMaterial
from .reflections import ReflectionPoints


@dataclass
class MeshScene:
    vertices: np.ndarray
    faces: np.ndarray
    material_ids: np.ndarray | None = None
    object_ids: np.ndarray | None = None
    face_velocities: np.ndarray | None = None
    materials: dict = field(default_factory=lambda: {"default": SurfaceMaterial()})
    sensor_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sensor_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    sensor_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=float)
        face_input = np.asarray(self.faces)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or not np.isfinite(self.vertices).all():
            raise ValueError("vertices must be finite (V,3)")
        if face_input.ndim != 2 or face_input.shape[1] != 3 or not np.issubdtype(face_input.dtype, np.integer):
            raise ValueError("faces must be integer triangle indices (F,3)")
        self.faces = face_input.astype(np.int64)
        if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= len(self.vertices)):
            raise ValueError("triangle vertex index out of bounds")
        self.materials = {str(k): v if isinstance(v, SurfaceMaterial) else SurfaceMaterial(**v)
                          for k, v in self.materials.items()}
        if not self.materials:
            raise ValueError("at least one material must be declared")
        default_id = next(iter(self.materials))
        self.material_ids = np.full(len(self.faces), default_id) if self.material_ids is None else np.asarray(self.material_ids).astype(str)
        self.object_ids = np.full(len(self.faces), -1, dtype=np.int64) if self.object_ids is None else np.asarray(self.object_ids, dtype=np.int64)
        self.face_velocities = np.zeros((len(self.faces), 3)) if self.face_velocities is None else np.asarray(self.face_velocities, dtype=float)
        if self.material_ids.shape != (len(self.faces),) or self.object_ids.shape != (len(self.faces),):
            raise ValueError("material_ids and object_ids require one entry per triangle")
        if set(self.material_ids.tolist()) - set(self.materials):
            raise ValueError("face refers to an undeclared material")
        if self.face_velocities.shape != (len(self.faces), 3) or not np.isfinite(self.face_velocities).all():
            raise ValueError("face_velocities must be finite (F,3)")
        self.sensor_position = np.asarray(self.sensor_position, dtype=float)
        self.sensor_rotation = np.asarray(self.sensor_rotation, dtype=float)
        self.sensor_velocity = np.asarray(self.sensor_velocity, dtype=float)
        if self.sensor_position.shape != (3,) or self.sensor_velocity.shape != (3,) or self.sensor_rotation.shape != (3, 3):
            raise ValueError("sensor pose requires position (3,), orthogonal rotation (3,3), velocity (3,)")
        if not all(np.isfinite(a).all() for a in (self.sensor_position, self.sensor_rotation, self.sensor_velocity)):
            raise ValueError("sensor pose must be finite")
        # Reflections allow handedness changes when converting CARLA's axes.
        if not np.allclose(self.sensor_rotation.T @ self.sensor_rotation, np.eye(3), atol=1e-8):
            raise ValueError("sensor_rotation must be orthogonal; columns map radar-local axes into world")

    @property
    def triangles(self):
        return self.vertices[self.faces]

    @property
    def normals_and_areas(self):
        triangles = self.triangles
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        length = np.linalg.norm(cross, axis=1)
        normals = cross / np.maximum(length[:, None], np.finfo(float).tiny)
        return normals, length / 2


def _triangulate_polygon(vertices, indices):
    """Ear-clip a simple planar OBJ polygon, retaining its declared winding."""
    if len(indices) == 3:
        return [tuple(indices)]
    points = np.asarray(vertices)[indices]
    normal = np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
    if np.linalg.norm(normal) < 1e-14:
        raise ValueError("OBJ polygon has zero area")
    unit = normal / np.linalg.norm(normal)
    span = max(float(np.ptp(points, axis=0).max()), 1.0)
    if np.max(np.abs((points - points[0]) @ unit)) > 1e-7 * span:
        raise ValueError("OBJ polygons must be planar; triangulate nonplanar faces externally")
    projected = np.delete(points, int(np.argmax(np.abs(normal))), axis=1)
    signed_area = np.sum(projected[:, 0] * np.roll(projected[:, 1], -1) - projected[:, 1] * np.roll(projected[:, 0], -1))
    orientation = 1 if signed_area > 0 else -1
    remaining = list(range(len(indices)))
    triangles = []

    def cross2(a, b):
        return float(a[0] * b[1] - a[1] * b[0])

    while len(remaining) > 3:
        clipped = False
        for k, current in enumerate(remaining):
            before, after = remaining[k - 1], remaining[(k + 1) % len(remaining)]
            a, b, c = projected[[before, current, after]]
            if orientation * cross2(b - a, c - b) <= 1e-14:
                continue
            inside = False
            for other in remaining:
                if other in (before, current, after):
                    continue
                p = projected[other]
                tests = [orientation * cross2(b - a, p - a), orientation * cross2(c - b, p - b),
                         orientation * cross2(a - c, p - c)]
                if min(tests) >= -1e-14:
                    inside = True
                    break
            if not inside:
                triangles.append((indices[before], indices[current], indices[after]))
                remaining.pop(k)
                clipped = True
                break
        if not clipped:
            raise ValueError("OBJ polygon is degenerate or self-intersecting; triangulate it externally")
    triangles.append(tuple(indices[i] for i in remaining))
    return triangles


def import_obj(path, materials=None, default_material=None, sensor_position=(0, 0, 0),
               sensor_rotation=None, sensor_velocity=(0, 0, 0)):
    """Read OBJ vertices/faces/usemtl/groups, including negative/slashed indices.

    Face winding controls one-sided normals. MTL optical colors are deliberately
    not converted to RF material coefficients. Supply an explicit material map;
    unknown material names receive the declared uncalibrated default.
    """
    path = Path(path)
    default_material = SurfaceMaterial() if default_material is None else default_material
    material_map = dict(materials or {"default": default_material})
    current_material = next(iter(material_map))
    vertices, faces, mat_ids, object_ids = [], [], [], []
    object_names, current_object = {"default": 0}, 0
    ignored_mtl = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
        tokens = raw_line.split("#", 1)[0].split()
        if not tokens:
            continue
        tag = tokens[0]
        try:
            if tag == "v":
                if len(tokens) < 4:
                    raise ValueError("vertex needs three coordinates")
                vertices.append([float(v) for v in tokens[1:4]])
            elif tag == "f":
                if not 3 <= len(tokens) - 1 <= 4096:
                    raise ValueError("face needs 3..4096 vertex indices")
                ids = []
                for token in tokens[1:]:
                    raw = int(token.split("/")[0])
                    if raw == 0:
                        raise ValueError("OBJ indices are 1-based or negative")
                    index = raw - 1 if raw > 0 else len(vertices) + raw
                    if not 0 <= index < len(vertices):
                        raise ValueError("face index outside previously declared vertices")
                    ids.append(index)
                triangles = _triangulate_polygon(vertices, ids)
                faces.extend(triangles)
                mat_ids.extend([current_material] * len(triangles))
                object_ids.extend([current_object] * len(triangles))
            elif tag == "usemtl":
                current_material = " ".join(tokens[1:])
                if not current_material:
                    raise ValueError("usemtl requires a material name")
                material_map.setdefault(current_material, default_material)
            elif tag in ("o", "g"):
                name = " ".join(tokens[1:]) or "default"
                current_object = object_names.setdefault(name, len(object_names))
            elif tag == "mtllib":
                ignored_mtl.extend(tokens[1:])
        except (ValueError, IndexError) as error:
            raise ValueError(f"{path.name}:{line_no}: {error}") from error
    return MeshScene(np.asarray(vertices, dtype=float).reshape(-1, 3), np.asarray(faces, dtype=np.int64).reshape(-1, 3),
                     np.asarray(mat_ids), np.asarray(object_ids, dtype=np.int64), materials=material_map,
                     sensor_position=sensor_position, sensor_rotation=np.eye(3) if sensor_rotation is None else sensor_rotation,
                     sensor_velocity=sensor_velocity, metadata={
                         "geometry_source": "OBJ triangle geometry", "source_file": path.name,
                         "object_names": object_names, "ignored_optical_mtl_files": ignored_mtl,
                         "material_status": "explicit RF map or uncalibrated default; optical MTL is not RF calibration"})


def save_mesh(path, scene):
    metadata = {"materials": {k: v.to_dict() for k, v in scene.materials.items()}, "scene": scene.metadata}
    np.savez_compressed(path, vertices=scene.vertices, faces=scene.faces, material_ids=scene.material_ids,
                        object_ids=scene.object_ids, face_velocities=scene.face_velocities,
                        sensor_position=scene.sensor_position, sensor_rotation=scene.sensor_rotation,
                        sensor_velocity=scene.sensor_velocity, metadata=json.dumps(metadata))


def load_mesh(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"])) if "metadata" in archive else {}
        fields = {name: archive[name] for name in ("vertices", "faces", "material_ids", "object_ids", "face_velocities",
                                                  "sensor_position", "sensor_rotation", "sensor_velocity") if name in archive}
        return MeshScene(**fields, materials=metadata.get("materials", {"default": SurfaceMaterial()}),
                         metadata=metadata.get("scene", {}))


def export_obj(path, scene):
    """Export geometry/material names; dynamic metadata is preserved only by NPZ."""
    lines = ["# RadSimReal mesh; coordinates in metres, see NPZ for RF metadata"]
    lines.extend("v " + " ".join(f"{v:.12g}" for v in point) for point in scene.vertices)
    last_material, last_object = None, None
    for face, material, object_id in zip(scene.faces, scene.material_ids, scene.object_ids):
        if object_id != last_object:
            lines.append(f"o object_{object_id}")
            last_object = object_id
        if material != last_material:
            lines.append(f"usemtl {material}")
            last_material = material
        lines.append("f " + " ".join(str(int(v) + 1) for v in face))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _box_faces(vertices):
    """Triangulate a convex cuboid from ordered bottom/top four-vertex rings."""
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7)]
    for i in range(4):
        j = (i + 1) % 4
        faces.extend([(i, j, j + 4), (i, j + 4, i + 4)])
    faces = np.asarray(faces, dtype=np.int64)
    # Correct outward orientation independently of caller's ring handedness.
    triangles = vertices[faces]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    outward = triangles.mean(axis=1) - vertices.mean(axis=0)
    flip = np.einsum("ij,ij->i", normal, outward) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces


def mesh_from_scene(scene, height=1.6, material=None, sensor_position=(0, 0, 0)):
    """Convert procedural vehicles to genuine closed triangle meshes (still box geometry)."""
    if not np.isfinite(height) or height <= 0:
        raise ValueError("height must be finite and positive")
    vertices, faces, object_ids, velocities = [], [], [], []
    for obj in scene.objects:
        corners = np.asarray(obj.corners())
        box = np.concatenate([np.column_stack([corners, np.zeros(4)]),
                              np.column_stack([corners, np.full(4, height)])])
        triangle_ids = _box_faces(box)
        faces.extend(triangle_ids + len(vertices))
        vertices.extend(box)
        object_ids.extend([obj.object_id] * len(triangle_ids))
        velocities.extend([[obj.vx, obj.vy, 0.0]] * len(triangle_ids))
    return MeshScene(np.asarray(vertices).reshape(-1, 3), np.asarray(faces, dtype=np.int64).reshape(-1, 3),
                     object_ids=np.asarray(object_ids, dtype=np.int64),
                     face_velocities=np.asarray(velocities).reshape(-1, 3),
                     materials={"vehicle": SurfaceMaterial() if material is None else material},
                     sensor_position=sensor_position, sensor_velocity=[*scene.ego_velocity, 0.0],
                     metadata={"geometry_source": "closed triangle extrusion of procedural vehicle footprints",
                               "scene": scene.to_dict(), "mesh_fidelity": "box approximation, not recovered CARLA CAD assets"})


def ground_plane(x_min=-50.0, x_max=50.0, y_min=0.0, y_max=60.0, z=-0.5,
                 material=None, object_id=-1):
    """Explicit static ground geometry; defaults are illustrative RF properties."""
    if not np.isfinite([x_min, x_max, y_min, y_max, z]).all() or x_max <= x_min or y_max <= y_min:
        raise ValueError("ground plane requires finite increasing extents")
    vertices = np.asarray([[x_min, y_min, z], [x_max, y_min, z],
                           [x_max, y_max, z], [x_min, y_max, z]])
    return MeshScene(vertices, np.array([[0, 1, 2], [0, 2, 3]]), object_ids=np.full(2, object_id),
                     materials={"ground": SurfaceMaterial(name="uncalibrated_ground", roughness_m=.005,
                                                           diffuse_fraction=.9) if material is None else material},
                     metadata={"geometry_source": "user-configured static ground plane", "calibrated": False})


def merge_meshes(scenes, preserve_object_ids=True):
    """Combine world-frame meshes and preserve the first scene's sensor pose.

    Materials with conflicting names are renamed without losing their RF
    properties. Object IDs are retained by default for native labels; callers
    can request unique IDs and inspect the recorded mapping when needed.
    """
    scenes = list(scenes)
    if not scenes:
        return MeshScene(np.empty((0, 3)), np.empty((0, 3), dtype=np.int64))
    vertices, faces, material_ids, object_ids, velocities = [], [], [], [], []
    materials, mappings, next_object = {}, [], 0
    vertex_count = 0
    for scene_index, scene in enumerate(scenes):
        material_map = {}
        for name, material in scene.materials.items():
            target = name
            while target in materials and materials[target].to_dict() != material.to_dict():
                target = f"scene{scene_index}:{target}"
            materials[target] = material
            material_map[name] = target
        ids = scene.object_ids.copy()
        object_map = {}
        if not preserve_object_ids:
            for source_id in np.unique(ids):
                object_map[str(int(source_id))] = next_object
                ids[scene.object_ids == source_id] = next_object
                next_object += 1
        vertices.append(scene.vertices)
        faces.append(scene.faces + vertex_count)
        material_ids.extend(material_map[str(name)] for name in scene.material_ids)
        object_ids.append(ids)
        velocities.append(scene.face_velocities)
        mappings.append({"scene_index": scene_index, "materials": material_map,
                         "object_ids": "preserved" if preserve_object_ids else object_map,
                         "source_metadata": scene.metadata})
        vertex_count += len(scene.vertices)
    first = scenes[0]
    return MeshScene(np.concatenate(vertices), np.concatenate(faces), np.asarray(material_ids),
                     np.concatenate(object_ids), np.concatenate(velocities), materials,
                     first.sensor_position.copy(), first.sensor_rotation.copy(), first.sensor_velocity.copy(),
                     {"geometry_source": "combined world-frame meshes", "sources": mappings,
                      "object_ids_preserved": bool(preserve_object_ids)})


class _BVH:
    """Median-split triangle BVH; bounded leaf pairs avoid N-rays x N-faces memory."""
    def __init__(self, triangles, ids=None, leaf_size=16):
        self.triangles = triangles
        self.ids = np.arange(len(triangles)) if ids is None else ids
        subset = triangles[self.ids]
        self.lower, self.upper = subset.min(axis=(0, 1)), subset.max(axis=(0, 1))
        self.left = self.right = None
        if len(self.ids) > leaf_size:
            centers = subset.mean(axis=1)
            axis = int(np.argmax(np.ptp(centers, axis=0)))
            ordered = self.ids[np.argsort(centers[:, axis], kind="stable")]
            middle = len(ordered) // 2
            self.left = _BVH(triangles, ordered[:middle], leaf_size)
            self.right = _BVH(triangles, ordered[middle:], leaf_size)

    def candidates(self, origins, directions, max_distance, epsilon):
        parallel = np.abs(directions) < 1e-15
        inverse = np.divide(1.0, directions, out=np.zeros_like(directions), where=~parallel)
        lo = (self.lower - origins) * inverse
        hi = (self.upper - origins) * inverse
        entry, exit_ = np.minimum(lo, hi), np.maximum(lo, hi)
        inside = (origins >= self.lower - epsilon) & (origins <= self.upper + epsilon)
        entry = np.where(parallel, np.where(inside, -np.inf, np.inf), entry)
        exit_ = np.where(parallel, np.where(inside, np.inf, -np.inf), exit_)
        return np.maximum(entry.max(axis=1), epsilon) <= np.minimum(exit_.min(axis=1), max_distance)


def _triangle_distances(origins, directions, triangles, epsilon):
    edge1, edge2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    p = np.cross(directions[:, None, :], edge2[None, :, :])
    determinant = np.einsum("tj,rtj->rt", edge1, p)
    determinant_valid = np.abs(determinant) > 1e-13
    reciprocal = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=determinant_valid)
    relative = origins[:, None, :] - triangles[None, :, 0, :]
    u = np.einsum("rtj,rtj->rt", relative, p) * reciprocal
    q = np.cross(relative, edge1[None, :, :])
    v = np.einsum("rj,rtj->rt", directions, q) * reciprocal
    distance = np.einsum("tj,rtj->rt", edge2, q) * reciprocal
    valid = determinant_valid & (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10) & (distance > epsilon)
    return np.where(valid, distance, np.inf)


def ray_triangle_intersections(origins, directions, vertices, faces, max_distance=None,
                               epsilon=1e-7, ray_batch_size=512):
    """Return nearest (distance,face_index) per ray; no hit is (inf,-1).

    Directions are normalized internally. Distances are physical metres. This
    tests both triangle sides for opaque occlusion; shading/backface selection
    is independent. max_distance can be a scalar or one distance per ray.
    """
    origins, directions = np.asarray(origins, dtype=float), np.asarray(directions, dtype=float)
    if origins.ndim == 1:
        origins = np.broadcast_to(origins, directions.shape)
    if origins.ndim != 2 or origins.shape[1] != 3 or directions.shape != origins.shape:
        raise ValueError("origins/directions must broadcast to (N,3)")
    if not np.isfinite(origins).all() or not np.isfinite(directions).all():
        raise ValueError("ray values must be finite")
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths <= 0) or not np.isfinite(epsilon) or epsilon <= 0 or ray_batch_size < 1:
        raise ValueError("ray directions, epsilon and batch size must be positive")
    directions = directions / lengths[:, None]
    triangles = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=np.int64)]
    maximum = np.full(len(directions), np.inf) if max_distance is None else np.broadcast_to(np.asarray(max_distance, dtype=float), (len(directions),)).copy()
    if np.isnan(maximum).any() or np.any(maximum < 0):
        raise ValueError("max_distance must be nonnegative")
    nearest = np.full(len(directions), np.inf)
    hit_ids = np.full(len(directions), -1, dtype=np.int64)
    if len(triangles) == 0:
        return nearest, hit_ids
    bvh = _BVH(triangles)
    for first in range(0, len(directions), ray_batch_size):
        indices = np.arange(first, min(first + ray_batch_size, len(directions)))
        stack = [(bvh, indices)]
        while stack:
            node, ray_ids = stack.pop()
            cutoffs = np.minimum(maximum[ray_ids], nearest[ray_ids])
            keep = node.candidates(origins[ray_ids], directions[ray_ids], cutoffs, epsilon)
            ray_ids = ray_ids[keep]
            if not len(ray_ids):
                continue
            if node.left is not None:
                stack.append((node.right, ray_ids))
                stack.append((node.left, ray_ids))
            else:
                distances = _triangle_distances(origins[ray_ids], directions[ray_ids], triangles[node.ids], epsilon)
                columns = np.argmin(distances, axis=1)
                distance = distances[np.arange(len(ray_ids)), columns]
                better = (distance < nearest[ray_ids]) & (distance <= maximum[ray_ids])
                changed = ray_ids[better]
                nearest[changed] = distance[better]
                hit_ids[changed] = node.ids[columns[better]]
    return nearest, hit_ids


def sample_surface(mesh_scene, rng=None, samples_per_square_meter=4.0, max_samples=20000):
    """Area-importance Monte Carlo quadrature: each sample carries total_area/N.

    Splitting a triangle does not change the represented surface measure. Counts
    are global, so the sample cap also preserves the quadrature area weights.
    """
    rng = np.random.default_rng() if rng is None else rng
    if not np.isfinite(samples_per_square_meter) or samples_per_square_meter <= 0:
        raise ValueError("samples_per_square_meter must be finite positive")
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer")
    normals, areas = mesh_scene.normals_and_areas
    valid = areas > 1e-14
    if not valid.any():
        return np.empty((0, 3)), np.empty(0, dtype=np.int64), np.empty(0)
    total = float(areas[valid].sum())
    count = min(max_samples, max(1, int(np.ceil(total * samples_per_square_meter))))
    probabilities = np.where(valid, areas, 0) / total
    selected = rng.choice(len(areas), size=count, p=probabilities)
    triangles = mesh_scene.triangles[selected]
    u = np.sqrt(rng.random(count))
    v = rng.random(count)
    points = ((1 - u)[:, None] * triangles[:, 0] + (u * (1 - v))[:, None] * triangles[:, 1]
              + (u * v)[:, None] * triangles[:, 2])
    return points, selected, np.full(count, total / count)


def trace_reflections(mesh_scene, radar_config, *, rng=None, samples_per_square_meter=4.0,
                      max_samples=20000, frequency_hz=77e9, phase_mode="random", speckle=False,
                      polarization="TE", min_range=0.1, link_budget=None):
    """Visible mesh patches -> existing ReflectionPoints in radar-local coordinates.

    Intensity/random-field modes use differential-RCS density times patch area.
    Incoherent expected power is independent of mesh tessellation and sample
    count. phase_mode='path' instead uses coherent physical-optics surface-area
    quadrature; this oscillatory integral needs wavelength-resolving sampling.
    Random mode includes two-way path phase plus independent rough-surface phase.

    Without link_budget, powers use the existing relative scattering_gain and
    reference_range scaling. An explicit RadarLinkBudget yields conditional
    received Watts; the material RCS approximation still requires calibration.
    Single-bounce only: no diffraction, transmission, multipath or weather.
    """
    if phase_mode not in ("random", "path"):
        raise ValueError("phase_mode must be random or path")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0 or not np.isfinite(min_range) or min_range <= 0:
        raise ValueError("frequency_hz and min_range must be finite positive")
    if link_budget is not None and not isinstance(link_budget, RadarLinkBudget):
        raise ValueError("link_budget must be RadarLinkBudget or None")
    rng = np.random.default_rng() if rng is None else rng
    points, face_ids, patch_area = sample_surface(mesh_scene, rng, samples_per_square_meter, max_samples)
    normals, _ = mesh_scene.normals_and_areas
    delta = points - mesh_scene.sensor_position
    distance = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(distance[:, None], 1e-30)
    cosine_signed = -np.einsum("ij,ij->i", normals[face_ids], direction)
    two_sided = np.array([mesh_scene.materials[str(i)].two_sided for i in mesh_scene.material_ids[face_ids]])
    cosine = np.where(two_sided, np.abs(cosine_signed), cosine_signed)
    facing = (cosine > 0) & (distance >= min_range)
    face_ids, patch_area, points, direction, distance, cosine = [a[facing] for a in
        (face_ids, patch_area, points, direction, distance, cosine)]
    epsilon = np.maximum(1e-7, distance * 1e-9)
    # Endpoint exclusion prevents sampled surfaces from self-shadowing.
    hit_distance, _ = ray_triangle_intersections(mesh_scene.sensor_position, direction,
        mesh_scene.vertices, mesh_scene.faces, max_distance=np.maximum(distance - epsilon, 0))
    visible = ~np.isfinite(hit_distance)
    face_ids, patch_area, points, direction, distance, cosine = [a[visible] for a in
        (face_ids, patch_area, points, direction, distance, cosine)]
    cosine = np.clip(cosine, 0, 1)
    complex_mode = radar_config.representation in ("field", "complex")
    density = np.zeros(len(face_ids))
    fresnel = np.zeros(len(face_ids), dtype=np.complex128)
    roughness = np.zeros(len(face_ids))
    material_names = mesh_scene.material_ids[face_ids]
    for name in np.unique(material_names):
        selected = material_names == name
        material = mesh_scene.materials[str(name)]
        density[selected] = material.backscatter_coefficient(cosine[selected], frequency_hz, polarization)
        fresnel[selected] = (material.fresnel_reflection(cosine[selected], frequency_hz, polarization)
                             * np.sqrt(material.backscatter_scale * (1 - material.diffuse_fraction)))
        roughness[selected] = material.coherent_roughness_power(cosine[selected], frequency_hz)
    wavelength = LIGHT_SPEED_M_S / frequency_hz
    if complex_mode and phase_mode == "path":
        # Scalar physical-optics PEC-plate limit: sigma = 4*pi*A^2/lambda^2 at normal incidence.
        sigma_field = np.sqrt(4 * np.pi) / wavelength * fresnel * cosine * patch_area * np.sqrt(roughness)
        if link_budget is None:
            coefficients = (np.sqrt(radar_config.scattering_gain) * sigma_field
                            * (radar_config.reference_range / distance) ** 2)
        else:
            factor = np.sqrt(link_budget.power(np.ones(len(distance)), distance, frequency_hz))
            coefficients = sigma_field * factor
        coefficients *= np.exp(-4j * np.pi * distance / wavelength)
        phase_description = "coherent scalar physical-optics area quadrature with two-way propagation phase; diffuse component excluded"
    else:
        sigma = density * patch_area
        if link_budget is None:
            power = radar_config.scattering_gain * sigma * (radar_config.reference_range / distance) ** 4
        else:
            power = link_budget.power(sigma, distance, frequency_hz)
        if speckle:
            power *= rng.exponential(size=len(power))
        coefficients = power
        if complex_mode:
            phase = -4 * np.pi * distance / wavelength + 2 * np.pi * rng.random(len(power))
            coefficients = np.sqrt(power) * np.exp(1j * phase)
        phase_description = "independent rough-surface phase plus two-way propagation phase" if complex_mode else "nonnegative incoherent power weights"
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("Nonfinite mesh return; check geometry/link budget/material scales")
    # A geometrically visible integration point can have exactly zero RF
    # response (zero RCS, matched interface, coherent-only diffuse surface).
    # Removing it leaves the image unchanged and prevents an invented target.
    # Complex equality avoids an unnecessary magnitude overflow for huge units.
    nonzero = coefficients != 0
    zero_response_samples_removed = int(np.count_nonzero(~nonzero))
    points, face_ids, coefficients = points[nonzero], face_ids[nonzero], coefficients[nonzero]
    # World vectors -> canonical radar-local x right, y forward, z up.
    local_points = (points - mesh_scene.sensor_position) @ mesh_scene.sensor_rotation
    local_velocity = (mesh_scene.face_velocities[face_ids] - mesh_scene.sensor_velocity) @ mesh_scene.sensor_rotation
    metadata = {"source": "single-bounce triangle-mesh RF approximation",
                "geometry": mesh_scene.metadata, "total_triangles": int(len(mesh_scene.faces)),
                "sampled_surface_points": int(len(facing)), "front_facing_points": int(facing.sum()),
                "visible_points": int(visible.sum()), "occluded_points": int((~visible).sum()),
                "visible_geometry_points": int(visible.sum()),
                "nonzero_return_count": int(nonzero.sum()),
                "zero_response_samples_removed": zero_response_samples_removed,
                "samples_per_square_meter": float(samples_per_square_meter), "maximum_samples": max_samples,
                "surface_quadrature": "area-importance Monte Carlo; total_area/N per sampled patch",
                "frequency_hz": float(frequency_hz), "polarization": polarization.upper(),
                "phase_model": phase_description, "phase_mode": phase_mode,
                "scattering_model": "Fresnel planar interface plus explicitly approximate rough-surface backscatter",
                "link_budget": None if link_budget is None else link_budget.to_dict(),
                "power_units": "relative units; calibrate sensor/material gain" if link_budget is None else "conditional received Watts with approximate material RCS",
                "velocity_frame": "radar-relative", "coordinates": "x right, y forward, z up",
                "missing_physics": ["multiple reflections", "transmission", "diffraction", "weather", "measured automotive material RCS"],
                "calibrated": False}
    return ReflectionPoints(local_points, local_velocity, coefficients, mesh_scene.object_ids[face_ids], metadata)
