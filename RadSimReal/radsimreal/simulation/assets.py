"""Place supplied vehicle CAD/OBJ assets into dynamic radar scenes.

Geometry, RF material declarations and provenance are preserved. This does not
redistribute or recover the RadSimReal authors' CARLA vehicle assets.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
import numpy as np

from .materials import SurfaceMaterial
from .mesh import MeshScene, import_obj, merge_meshes

COORDINATE_CONTRACT = "x_right_y_forward_z_up"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class VehicleAsset:
    name: str
    mesh: MeshScene
    source_path: Path
    source_sha256: str
    lower: np.ndarray
    upper: np.ndarray

    @property
    def dimensions(self):
        return self.upper - self.lower


@dataclass
class VehicleAssetLibrary:
    assets: list[VehicleAsset]
    fit_dimensions: bool = True
    height: float = 1.6
    dimension_tolerance: float = 1e-5
    coordinate_system: str = COORDINATE_CONTRACT
    manifest_path: Path | None = None
    manifest_sha256: str | None = None

    def __post_init__(self):
        if not self.assets:
            raise ValueError("Vehicle asset library cannot be empty")
        if type(self.fit_dimensions) is not bool:
            raise ValueError("fit_dimensions must be a boolean")
        if not np.isfinite([self.height, self.dimension_tolerance]).all() or min(self.height, self.dimension_tolerance) <= 0:
            raise ValueError("height and dimension_tolerance must be finite positive")
        if self.coordinate_system != COORDINATE_CONTRACT:
            raise ValueError("Vehicle assets must explicitly use x right, y forward, z up; convert other source axes before loading")


def load_vehicle_assets(manifest_path):
    """Load a JSON library and its OBJ templates once, ready for repeated scenes.

    OBJ paths resolve relative to the manifest. Default axes are the declared
    contract x right/y forward/z up. Axis semantics cannot be inferred from a
    vehicle's bounding dimensions. Material overrides contain SurfaceMaterial
    fields keyed by OBJ usemtl name; optical MTL data are not used as RF values.
    """
    path = Path(manifest_path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Vehicle asset manifest must be a JSON mapping")
    allowed = {"assets", "fit_dimensions", "height", "dimension_tolerance", "coordinate_system"}
    if set(data) - allowed:
        raise ValueError(f"Unknown vehicle asset manifest fields: {sorted(set(data) - allowed)}")
    records = data.get("assets")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest needs a nonempty assets list")
    coordinate_system = data.get("coordinate_system", COORDINATE_CONTRACT)
    if coordinate_system != COORDINATE_CONTRACT:
        raise ValueError("Vehicle assets require x_right_y_forward_z_up coordinates; transform them explicitly first")
    assets = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("obj"), str):
            raise ValueError("Each vehicle asset needs an obj filename")
        if set(record) - {"obj", "material_overrides", "name"}:
            raise ValueError(f"Unknown fields in vehicle asset {index}")
        source = (path.parent / record["obj"]).resolve()
        if not source.is_file() or source.suffix.lower() != ".obj":
            raise FileNotFoundError(f"Vehicle asset OBJ file is missing or has the wrong extension: {source}")
        overrides = record.get("material_overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError("material_overrides must map OBJ usemtl names to SurfaceMaterial fields")
        materials = {}
        for name, material_data in overrides.items():
            if not isinstance(material_data, dict):
                raise ValueError(f"Material override {name!r} must be a mapping")
            parameters = dict(material_data)
            parameters.setdefault("name", str(name))
            try:
                materials[str(name)] = SurfaceMaterial(**parameters)
            except TypeError as error:
                raise ValueError(f"Invalid RF material fields for {name!r}: {error}") from error
        mesh = import_obj(source, materials=materials or None)
        if not len(mesh.vertices) or not len(mesh.faces):
            raise ValueError(f"Vehicle asset {source.name} has no triangle geometry")
        lower, upper = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
        dimensions = upper - lower
        if np.any(dimensions <= 1e-9) or not np.isfinite(dimensions).all() or not np.any(mesh.normals_and_areas[1] > 1e-14):
            raise ValueError(f"Vehicle asset {source.name} needs nondegenerate lateral, forward and vertical extents")
        assets.append(VehicleAsset(str(record.get("name", source.stem)), mesh, source, _sha256(source), lower, upper))
    return VehicleAssetLibrary(assets=assets, fit_dimensions=data.get("fit_dimensions", True),
                               height=data.get("height", 1.6),
                               dimension_tolerance=data.get("dimension_tolerance", 1e-5),
                               coordinate_system=coordinate_system, manifest_path=path,
                               manifest_sha256=_sha256(path))


def mesh_from_assets(scene, library, rng=None, sensor_position=(0, 0, 0)):
    """Place actual templates at each Vehicle's pose, preserving target dimensions.

    Bounding extents are fitted before clockwise-from-forward Vehicle rotation.
    Scaling is positive and preserves triangle winding. Each whole vehicle gets
    its original scene object ID and world translational velocity. Source group
    names/material identities remain available in asset/merge provenance.
    """
    if not isinstance(library, VehicleAssetLibrary):
        raise ValueError("library must be returned by load_vehicle_assets")
    rng = np.random.default_rng() if rng is None else rng
    ego = np.asarray([*scene.ego_velocity, 0.0], dtype=float)
    if ego.shape != (3,) or not np.isfinite(ego).all():
        raise ValueError("Scene ego velocity must have two finite lateral/forward components")
    ids = [obj.object_id for obj in scene.objects]
    if any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Vehicle scene object IDs must be unique nonnegative integers")
    placed, placements = [], []
    for obj in scene.objects:
        values = [obj.x, obj.y, obj.width, obj.length, obj.yaw, obj.vx, obj.vy]
        if not np.isfinite(values).all() or min(obj.width, obj.length) <= 0:
            raise ValueError("Vehicle pose/motion must be finite, with positive width and length")
        asset_index = int(rng.integers(len(library.assets)))
        asset = library.assets[asset_index]
        target_dimensions = np.array([obj.width, obj.length, library.height], dtype=float)
        if library.fit_dimensions:
            scale = target_dimensions / asset.dimensions
        else:
            if not np.allclose(asset.dimensions, target_dimensions,
                               rtol=library.dimension_tolerance, atol=library.dimension_tolerance):
                raise ValueError(f"Vehicle {obj.object_id} dimensions disagree with preserved-unit asset {asset.name!r}; enable fit_dimensions or supply matching scene dimensions")
            scale = np.ones(3)
        origin = np.array([(asset.lower[0] + asset.upper[0]) / 2,
                           (asset.lower[1] + asset.upper[1]) / 2, asset.lower[2]])
        local = (asset.mesh.vertices - origin) * scale
        cosine, sine = np.cos(obj.yaw), np.sin(obj.yaw)
        rotation = np.array([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        translation = np.array([obj.x, obj.y, 0.0])
        world_vertices = local @ rotation.T + translation
        if not np.isfinite(world_vertices).all():
            raise ValueError("Vehicle asset placement overflowed; check source units and requested dimensions")
        placement = {"object_id": int(obj.object_id), "asset_index": asset_index,
                     "asset_name": asset.name, "source_obj": asset.source_path.name,
                     "source_sha256": asset.source_sha256,
                     "source_dimensions_m_or_declared_units": asset.dimensions.tolist(),
                     "requested_dimensions_m": target_dimensions.tolist(),
                     "scale_xyz": scale.tolist(), "translation_xyz_m": translation.tolist(),
                     "yaw_clockwise_from_forward_rad": float(obj.yaw),
                     "velocity_world_xyz_m_s": [float(obj.vx), float(obj.vy), 0.0],
                     "template_object_names": asset.mesh.metadata.get("object_names", {}),
                     "template_material_names": sorted(asset.mesh.materials)}
        mesh = MeshScene(world_vertices, asset.mesh.faces.copy(), asset.mesh.material_ids.copy(),
                         np.full(len(asset.mesh.faces), obj.object_id, dtype=np.int64),
                         np.tile([obj.vx, obj.vy, 0.0], (len(asset.mesh.faces), 1)),
                         dict(asset.mesh.materials), sensor_position=np.asarray(sensor_position, dtype=float),
                         sensor_velocity=ego.copy(), metadata={"vehicle_asset_placement": placement})
        placed.append(mesh)
        placements.append(placement)
    if placed:
        result = merge_meshes(placed)
    else:
        result = MeshScene(np.empty((0, 3)), np.empty((0, 3), dtype=np.int64),
                           sensor_position=sensor_position, sensor_velocity=ego)
    result.metadata.update({"geometry_source": "moving user-supplied vehicle OBJ templates",
                            "coordinate_system": library.coordinate_system,
                            "asset_coordinate_semantics": "declared x-right/y-forward/z-up; front/nose cannot be inferred from extents",
                            "asset_manifest": None if library.manifest_path is None else library.manifest_path.name,
                            "asset_manifest_sha256": library.manifest_sha256,
                            "fit_dimensions": library.fit_dimensions, "vehicle_placements": placements,
                            "template_selection": "uniform per vehicle using supplied NumPy generator",
                            "rf_material_status": "user overrides or explicit uncalibrated defaults; optical MTL is not RF calibration",
                            "scene": scene.to_dict(), "calibrated": False})
    return result
