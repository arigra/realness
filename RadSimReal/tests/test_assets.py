"""Supplied vehicle OBJ placement and RF material provenance tests."""
import hashlib
import json
import numpy as np
import pytest

from radsimreal.config import RadarConfig
from radsimreal.simulation.assets import load_vehicle_assets, mesh_from_assets
from radsimreal.simulation.mesh import trace_reflections
from radsimreal.simulation.scene import Scene, Vehicle


VEHICLE_OBJ = """# Synthetic test fixture: lower body plus a sloped glass cabin
v -1 -2 1
v 1 -2 1
v 1 2 1
v -1 2 1
v -1 -2 2
v 1 -2 2
v 1 2 2
v -1 2 2
v -0.7 -0.8 3
v 0.7 -0.8 3
v 0.7 0.8 3
v -0.7 0.8 3
o body
usemtl body
f 1 4 3 2
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
o cabin
usemtl glass
f 5 6 10 9
f 6 7 11 10
f 7 8 12 11
f 8 5 9 12
f 9 10 11 12
"""


def library_file(tmp_path, *, fit=True, height=1.6, count=1):
    assets = []
    for i in range(count):
        path = tmp_path / f"vehicle_{i}.obj"
        path.write_text(VEHICLE_OBJ + f"# variant {i}\n")
        assets.append({"obj": path.name, "name": f"vehicle_variant_{i}",
                       "material_overrides": {"body": {"relative_permittivity_real": 6., "diffuse_fraction": .4},
                                              "glass": {"relative_permittivity_real": 2.25, "roughness_m": .00001}}})
    path = tmp_path / "vehicles.json"
    path.write_text(json.dumps({"assets": assets, "fit_dimensions": fit, "height": height,
                                "coordinate_system": "x_right_y_forward_z_up"}))
    return path


def test_relative_obj_loading_material_names_and_source_hashes(tmp_path):
    manifest = library_file(tmp_path)
    library = load_vehicle_assets(manifest)
    asset = library.assets[0]
    assert asset.source_path == (tmp_path / "vehicle_0.obj").resolve()
    assert asset.source_sha256 == hashlib.sha256(asset.source_path.read_bytes()).hexdigest()
    assert library.manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    np.testing.assert_allclose(asset.dimensions, [2, 4, 2])
    assert {"body", "glass"} <= set(asset.mesh.material_ids)
    assert asset.mesh.materials["glass"].relative_permittivity_real == 2.25
    assert {"body", "cabin"} <= set(asset.mesh.metadata["object_names"])


def test_fitted_template_pose_extents_grounding_and_face_velocity(tmp_path):
    library = load_vehicle_assets(library_file(tmp_path, height=3.))
    vehicle = Vehicle(5., 20., 8., 4., np.pi / 2, 1., 4., object_id=7)
    scene = Scene([vehicle], ego_velocity=(0., 2.))
    mesh = mesh_from_assets(scene, library, np.random.default_rng(3))
    assert set(mesh.object_ids) == {7}
    np.testing.assert_allclose(mesh.face_velocities, np.tile([1, 4, 0], (len(mesh.faces), 1)))
    assert mesh.vertices[:, 2].min() == pytest.approx(0)
    assert mesh.vertices[:, 2].max() == pytest.approx(3)
    corners = vehicle.corners()
    np.testing.assert_allclose(mesh.vertices[:, :2].min(axis=0), corners.min(axis=0))
    np.testing.assert_allclose(mesh.vertices[:, :2].max(axis=0), corners.max(axis=0))
    np.testing.assert_allclose(mesh.metadata["vehicle_placements"][0]["scale_xyz"], [2, 2, 1.5])
    result = trace_reflections(mesh, RadarConfig(), rng=np.random.default_rng(5), samples_per_square_meter=8)
    assert len(result) > 0
    np.testing.assert_allclose(result.velocity, np.tile([1, 2, 0], (len(result), 1)))
    assert set(result.object_ids) == {7}


def test_uniform_scaling_multiplies_area_by_square_and_rotates_normals(tmp_path):
    library = load_vehicle_assets(library_file(tmp_path, height=4.))
    yaw = .37
    scene = Scene([Vehicle(3., 20., 8., 4., yaw, 0., 0., object_id=1)])
    mesh = mesh_from_assets(scene, library, np.random.default_rng(1))
    template = library.assets[0].mesh
    source_normal, source_area = template.normals_and_areas
    target_normal, target_area = mesh.normals_and_areas
    np.testing.assert_allclose(target_area, source_area * 4)
    rotation = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    np.testing.assert_allclose(target_normal, source_normal @ rotation.T, atol=1e-14)


def test_seeded_asset_assignment_does_not_mutate_templates(tmp_path):
    library = load_vehicle_assets(library_file(tmp_path, count=2))
    source = [asset.mesh.vertices.copy() for asset in library.assets]
    scene = Scene([Vehicle(i * 3., 20., 4.5, 2., .1 * i, 1., 0., object_id=i) for i in range(10)])
    first = mesh_from_assets(scene, library, np.random.default_rng(42))
    second = mesh_from_assets(scene, library, np.random.default_rng(42))
    np.testing.assert_array_equal(first.vertices, second.vertices)
    assert first.metadata["vehicle_placements"] == second.metadata["vehicle_placements"]
    assert {p["asset_index"] for p in first.metadata["vehicle_placements"]} == {0, 1}
    for original, asset in zip(source, library.assets):
        np.testing.assert_array_equal(original, asset.mesh.vertices)


def test_preserved_units_require_matching_target_dimensions(tmp_path):
    library = load_vehicle_assets(library_file(tmp_path, fit=False, height=2.))
    matching = Scene([Vehicle(0, 10, 4, 2, 0, 0, 0)])
    mesh = mesh_from_assets(matching, library, np.random.default_rng(1))
    np.testing.assert_allclose(mesh.metadata["vehicle_placements"][0]["scale_xyz"], 1)
    wrong = Scene([Vehicle(0, 10, 5, 2, 0, 0, 0)])
    with pytest.raises(ValueError, match="dimensions disagree"):
        mesh_from_assets(wrong, library, np.random.default_rng(1))


def test_reject_missing_degenerate_or_wrong_axis_assets(tmp_path):
    manifest = tmp_path / "invalid.json"
    manifest.write_text(json.dumps({"assets": [{"obj": "missing.obj"}]}))
    with pytest.raises(FileNotFoundError):
        load_vehicle_assets(manifest)
    manifest.write_text(json.dumps({"assets": [{"obj": "missing.obj"}], "coordinate_system": "x_forward_y_left_z_up"}))
    with pytest.raises(ValueError, match="coordinates"):
        load_vehicle_assets(manifest)
    flat = tmp_path / "flat.obj"
    flat.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    manifest.write_text(json.dumps({"assets": [{"obj": "flat.obj"}]}))
    with pytest.raises(ValueError, match="nondegenerate"):
        load_vehicle_assets(manifest)


def test_invalid_vehicle_pose_ids_and_empty_scene(tmp_path):
    library = load_vehicle_assets(library_file(tmp_path))
    empty = mesh_from_assets(Scene([], ego_velocity=(1, 2)), library, np.random.default_rng(1))
    assert empty.vertices.shape == (0, 3) and empty.faces.shape == (0, 3)
    np.testing.assert_allclose(empty.sensor_velocity, [1, 2, 0])
    invalid = Scene([Vehicle(0, 10, 4, 2, float("nan"), 0, 0)])
    with pytest.raises(ValueError, match="pose"):
        mesh_from_assets(invalid, library)
    duplicate = Scene([Vehicle(0, 10, 4, 2, 0, 0, 0), Vehicle(3, 10, 4, 2, 0, 0, 0)])
    with pytest.raises(ValueError, match="unique"):
        mesh_from_assets(duplicate, library)
