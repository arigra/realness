"""Analytical geometry/physics checks; these are not real-scene calibration."""
import importlib
import sys
from unittest.mock import patch
import numpy as np
import pytest

from radsimreal.config import RadarConfig
from radsimreal.simulation.materials import SurfaceMaterial, RadarLinkBudget, monostatic_received_power, LIGHT_SPEED_M_S
from radsimreal.simulation.mesh import (MeshScene, import_obj, export_obj, save_mesh, load_mesh, mesh_from_scene,
                                      sample_surface, ray_triangle_intersections, trace_reflections,
                                      ground_plane, merge_meshes)
from radsimreal.simulation.scene import Scene, Vehicle


def panel(distance=10., width=2., height=2., object_id=1, material=None):
    vertices = np.array([[-width / 2, distance, -height / 2], [width / 2, distance, -height / 2],
                         [width / 2, distance, height / 2], [-width / 2, distance, height / 2]])
    return MeshScene(vertices, np.array([[0, 1, 2], [0, 2, 3]]), object_ids=np.array([object_id, object_id]),
                     materials={"test": SurfaceMaterial(diffuse_fraction=1) if material is None else material})


def test_fresnel_normal_incidence_and_brewster():
    material = SurfaceMaterial(relative_permittivity_real=4, loss_tangent=0, conductivity_s_per_m=0)
    assert material.fresnel_reflection(1., polarization="TE") == pytest.approx(-1 / 3)
    assert material.fresnel_reflection(1., polarization="TM") == pytest.approx(1 / 3)
    cosine = 1 / np.sqrt(5)
    assert abs(material.fresnel_reflection(cosine, polarization="TM")) < 1e-14
    values = material.fresnel_reflection(np.linspace(0, 1, 101))
    assert np.max(np.abs(values)) <= 1 + 1e-14


def test_material_loss_and_roughness_physical_limits():
    material = SurfaceMaterial(conductivity_s_per_m=1e7, roughness_m=.002)
    assert material.complex_relative_permittivity().imag < 0
    assert abs(material.fresnel_reflection(1.)) > .99
    assert material.coherent_roughness_power(1.) < 1e-12
    response = material.backscatter_coefficient(np.linspace(0, 1, 100))
    assert np.isfinite(response).all() and np.all(response >= 0)
    with pytest.raises(ValueError):
        SurfaceMaterial(diffuse_fraction=1.1)
    with pytest.raises(ValueError):
        material.fresnel_reflection(-.1)
    with pytest.raises(ValueError):
        material.coherent_roughness_power(.5, frequency_hz=0)


def test_link_budget_known_equation_and_inverse_fourth_power():
    power = monostatic_received_power(1, 1, LIGHT_SPEED_M_S)
    assert power == pytest.approx(1 / (4 * np.pi) ** 3)
    assert monostatic_received_power(1, 2, LIGHT_SPEED_M_S) == pytest.approx(power / 16)
    assert RadarLinkBudget(transmit_power_w=2).power(1, 1, LIGHT_SPEED_M_S) == pytest.approx(2 * power)
    with pytest.raises(ValueError):
        monostatic_received_power(1, 0)


def test_mesh_closed_box_area_and_outward_normals():
    scene = Scene([Vehicle(0, 10, 4, 2, .3, 1, 4, object_id=5)], ego_velocity=(0, 2))
    mesh = mesh_from_scene(scene, height=1.6)
    assert mesh.vertices.shape == (8, 3) and mesh.faces.shape == (12, 3)
    normals, areas = mesh.normals_and_areas
    assert areas.sum() == pytest.approx(2 * (4 * 2 + 4 * 1.6 + 2 * 1.6))
    centroid = mesh.vertices.mean(axis=0)
    assert np.all(np.einsum("ij,ij->i", normals, mesh.triangles.mean(axis=1) - centroid) > 0)
    result = trace_reflections(mesh, RadarConfig(), rng=np.random.default_rng(4))
    assert len(result) > 0
    np.testing.assert_allclose(result.velocity, np.tile([1., 2., 0.], (len(result), 1)))
    assert np.all(result.object_ids == 5)


def test_ray_triangle_intersections_hit_miss_bounds_and_backside():
    mesh = panel()
    distance, face = ray_triangle_intersections(np.zeros(3), np.array([[0, 1, 0], [2, 1, 0]]), mesh.vertices, mesh.faces)
    assert distance[0] == pytest.approx(10)
    assert face[0] in (0, 1) and np.isinf(distance[1]) and face[1] == -1
    distance, face = ray_triangle_intersections([0, 20, 0], [[0, -1, 0]], mesh.vertices, mesh.faces)
    assert distance[0] == pytest.approx(10)  # opaque occlusion uses both sides
    limited, _ = ray_triangle_intersections([0, 0, 0], [[0, 1, 0]], mesh.vertices, mesh.faces, max_distance=9)
    assert np.isinf(limited[0])


def test_triangle_bvh_agrees_with_analytic_multiple_panels():
    panels = [panel(d, object_id=i) for i, d in enumerate((5, 10, 15, 20, 25))]
    mesh = MeshScene(np.concatenate([p.vertices for p in panels]),
                     np.concatenate([p.faces + 4 * i for i, p in enumerate(panels)]))
    directions = np.column_stack([np.linspace(-.1, .1, 51), np.ones(51), np.zeros(51)])
    distance, _ = ray_triangle_intersections([0, 0, 0], directions, mesh.vertices, mesh.faces, ray_batch_size=7)
    expected = 5 * np.sqrt(1 + directions[:, 0] ** 2)
    np.testing.assert_allclose(distance, expected, rtol=1e-13)


def test_mesh_occlusion_removes_far_surface():
    front, back = panel(10, width=4, height=4, object_id=1), panel(20, object_id=2)
    mesh = MeshScene(np.concatenate([front.vertices, back.vertices]), np.concatenate([front.faces, back.faces + 4]),
                     object_ids=np.array([1, 1, 2, 2]))
    result = trace_reflections(mesh, RadarConfig(), rng=np.random.default_rng(2), samples_per_square_meter=30)
    assert len(result) > 100
    assert set(result.object_ids.tolist()) == {1}
    assert result.metadata["occluded_points"] > 10


def test_surface_quadrature_area_and_triangulation_invariance():
    a = panel(100)
    # Same square, now split into four triangles around a central vertex.
    b = MeshScene(np.vstack([a.vertices, a.vertices.mean(axis=0)]), np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]),
                  materials=a.materials)
    for scene in (a, b):
        points, ids, weights = sample_surface(scene, np.random.default_rng(5), samples_per_square_meter=100)
        assert len(points) == 400 and len(ids) == 400
        assert weights.sum() == pytest.approx(4)
    cfg = RadarConfig()
    p1 = trace_reflections(a, cfg, rng=np.random.default_rng(4), samples_per_square_meter=100).coefficients.sum()
    p2 = trace_reflections(b, cfg, rng=np.random.default_rng(4), samples_per_square_meter=100).coefficients.sum()
    assert p1 == pytest.approx(p2, rel=5e-5)


def test_relative_mesh_return_obeys_inverse_fourth_power():
    cfg = RadarConfig()
    a = trace_reflections(panel(10, width=1e-4, height=1e-4), cfg, rng=np.random.default_rng(2)).coefficients.sum()
    b = trace_reflections(panel(20, width=1e-4, height=1e-4), cfg, rng=np.random.default_rng(2)).coefficients.sum()
    assert a / b == pytest.approx(16, rel=1e-9)


def test_complex_modes_phase_and_area_quadrature():
    cfg = RadarConfig(representation="complex")
    material = SurfaceMaterial(relative_permittivity_real=4, loss_tangent=0, roughness_m=0, diffuse_fraction=0)
    scene = panel(10, width=1e-5, height=1e-5, material=material)
    one = trace_reflections(scene, cfg, rng=np.random.default_rng(1), phase_mode="path", samples_per_square_meter=1)
    many = trace_reflections(scene, cfg, rng=np.random.default_rng(1), phase_mode="path", samples_per_square_meter=1e12)
    assert len(one) == 1 and 100 <= len(many) <= 101  # binary rounding can move the ceiling by one
    assert np.iscomplexobj(one.coefficients)
    assert one.coefficients.sum() == pytest.approx(many.coefficients.sum(), rel=2e-8)
    distance = np.linalg.norm(one.xyz[0])
    expected_phase = np.exp(-4j * np.pi * distance / (LIGHT_SPEED_M_S / 77e9)) * -1
    assert one.coefficients[0] / abs(one.coefficients[0]) == pytest.approx(expected_phase, rel=1e-10)
    random = trace_reflections(scene, cfg, rng=np.random.default_rng(3), phase_mode="random")
    again = trace_reflections(scene, cfg, rng=np.random.default_rng(3), phase_mode="random")
    np.testing.assert_array_equal(random.coefficients, again.coefficients)


def test_sensor_pose_and_velocity_conversion():
    mesh = panel()
    mesh.sensor_position = np.array([0., 1., 0.])
    mesh.sensor_rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    mesh.sensor_velocity = np.array([1., 2., 0.])
    mesh.face_velocities[:] = [3., 6., 0.]
    result = trace_reflections(mesh, RadarConfig(), rng=np.random.default_rng(3))
    world_points = result.xyz @ mesh.sensor_rotation.T + mesh.sensor_position
    np.testing.assert_allclose(world_points[:, 1], 10)
    np.testing.assert_allclose(result.velocity, np.tile([-4., 2., 0.], (len(result), 1)))


def test_obj_quad_slashes_negative_indices_and_concave_ear_clipping(tmp_path):
    text = "v 0 10 0\nv 2 10 0\nv 2 10 2\nv 1 10 1\nv 0 10 2\nusemtl panel\no car\nf -5/1/1 -4/2/1 -3/3/1 -2/4/1 -1/5/1\n"
    path = tmp_path / "concave.obj"
    path.write_text(text)
    mesh = import_obj(path)
    assert mesh.faces.shape == (3, 3)
    assert mesh.normals_and_areas[1].sum() == pytest.approx(3)
    assert set(mesh.material_ids) == {"panel"}
    with pytest.raises(ValueError):
        MeshScene(mesh.vertices, np.array([[0, 1, 10]]))


def test_mesh_npz_and_obj_roundtrip_preserve_geometry(tmp_path):
    original = mesh_from_scene(Scene([Vehicle(2, 15, 4.5, 2, .1, 1, 2)]))
    save_mesh(tmp_path / "scene.npz", original)
    restored = load_mesh(tmp_path / "scene.npz")
    np.testing.assert_array_equal(restored.faces, original.faces)
    np.testing.assert_array_equal(restored.face_velocities, original.face_velocities)
    np.testing.assert_allclose(restored.vertices, original.vertices)
    export_obj(tmp_path / "scene.obj", original)
    obj = import_obj(tmp_path / "scene.obj", materials=original.materials)
    assert obj.normals_and_areas[1].sum() == pytest.approx(original.normals_and_areas[1].sum())


def test_ground_plane_and_static_dynamic_mesh_merge():
    car = mesh_from_scene(Scene([Vehicle(0, 10, 4, 2, 0, 1, 2, object_id=5)]))
    ground = ground_plane(-10, 10, 0, 30)
    assert ground.normals_and_areas[1].sum() == pytest.approx(600)
    np.testing.assert_allclose(ground.normals_and_areas[0], np.tile([0, 0, 1], (2, 1)))
    combined = merge_meshes([car, ground])
    assert len(combined.faces) == 14 and set(combined.object_ids.tolist()) == {-1, 5}
    assert set(combined.material_ids) == {"vehicle", "ground"}
    remapped = merge_meshes([car, ground], preserve_object_ids=False)
    assert set(remapped.object_ids.tolist()) == {0, 1}


def test_empty_degenerate_and_invalid_meshes():
    cfg = RadarConfig()
    empty = mesh_from_scene(Scene([]))
    assert len(trace_reflections(empty, cfg)) == 0
    degenerate = MeshScene(np.zeros((3, 3)), np.array([[0, 1, 2]]))
    assert len(trace_reflections(degenerate, cfg)) == 0
    with pytest.raises(ValueError):
        MeshScene(np.zeros((3, 3)), np.array([[0., 1., 2.]]))
    with pytest.raises(ValueError):
        MeshScene(np.zeros((3, 3)), np.array([[0, 1, 2]]), sensor_rotation=np.ones((3, 3)))
    with pytest.raises(ValueError):
        trace_reflections(panel(), cfg, phase_mode="unknown")


@pytest.mark.parametrize("representation,phase_mode", [("intensity", "random"), ("complex", "random"), ("complex", "path")])
def test_zero_rcs_surface_has_visible_geometry_but_no_rf_return_points(representation, phase_mode):
    scene = panel(material=SurfaceMaterial(backscatter_scale=0))
    result = trace_reflections(scene, RadarConfig(representation=representation),
                                rng=np.random.default_rng(2), samples_per_square_meter=10,
                                phase_mode=phase_mode)
    assert len(result) == 0
    assert result.xyz.shape == (0, 3) and result.velocity.shape == (0, 3)
    assert result.coefficients.shape == (0,) and result.object_ids.shape == (0,)
    assert result.metadata["visible_geometry_points"] > 0
    assert result.metadata["nonzero_return_count"] == 0
    assert result.metadata["zero_response_samples_removed"] == result.metadata["visible_geometry_points"]


def test_readonly_carla_box_bridge_without_carla_install():
    from radsimreal.simulation.carla_bridge import boxes_to_mesh, snapshot_from_carla
    source = mesh_from_scene(Scene([Vehicle(0, 10, 4, 2, 0, 0, 0)]))
    bridged = boxes_to_mesh([{"vertices": source.vertices, "object_id": 42, "velocity": [1, 2, 0], "type_id": "vehicle.test"}])
    assert bridged.faces.shape == (12, 3)
    assert bridged.metadata["world_modified"] is False
    assert np.all(bridged.object_ids == 42)
    with patch("radsimreal.simulation.carla_bridge.importlib.import_module", side_effect=ImportError):
        with pytest.raises(RuntimeError, match="optional"):
            snapshot_from_carla()
