"""Native on-disk export/import roundtrips on disclosed synthetic fixtures."""
from dataclasses import replace
import json
from pathlib import Path
import pickle
import numpy as np
import pytest

from radsimreal.native.export import canonical_to_native, ExportNativeWriter
from radsimreal.native.profiles import NativeProfile
from radsimreal.native.readers import NativeReader
from radsimreal.radar import simulate
from radsimreal.simulation.scene import Scene, Vehicle
from radsimreal.simulation.reflections import ReflectionPoints


def profile(dataset="raddet", representation="complex", **kwargs):
    defaults = dict(name=dataset, shape_native=(32, 32, 8), representation=representation,
                    range_step=1., velocity_step=2., angular_min=-.9,
                    angular_step=1.8 / 31, azimuth_coordinate="sin",
                    range_reversed=True, doppler_positive="receding",
                    cartesian_affine={"x_from_col_scale": 1., "x_from_col_offset": -32.,
                                      "y_from_row_scale": -1., "y_from_row_offset": 31.})
    if dataset == "carrada":
        defaults.update(doppler_positive="approaching", annotation_range_reverse_origin="R",
                        annotation_angle_step=np.pi / 32, cartesian_affine=None, car_labels=("car", 3))
    defaults.update(kwargs)
    return NativeProfile(**defaults)


def fixture(p):
    scene = Scene([Vehicle(2., 12., 4., 2., .2, 0., 4., object_id=7),
                   Vehicle(-8., 20., 4., 2., 0., 0., 0., object_id=99)])
    xyz = np.array([[1.2, 10.2, .4], [2., 10.4, .6], [2.7, 10.8, .5]])
    velocity = np.tile([0., 4., 0.], (3, 1))
    weights = np.array([1., 2., 1.5])
    if p.representation == "complex":
        weights = np.sqrt(weights) * np.exp(1j * np.array([0., .4, .8]))
    refs = ReflectionPoints(xyz, velocity, weights, np.full(3, 7))
    cfg = p.radar_config()
    cfg.noise_std = 0
    result = simulate(refs, cfg, psf=np.ones((3, 3, 5)), rng=np.random.default_rng(3))
    return scene, refs, result


@pytest.mark.parametrize("representation", ["complex", "power", "amplitude", "db"])
@pytest.mark.parametrize("range_flip,angle_flip,doppler_flip,sign", [
    (False, False, False, "receding"), (True, True, False, "approaching"),
    (True, False, True, "receding"), (False, True, True, "approaching")])
def test_canonical_native_array_roundtrip_all_response_and_axis_conventions(tmp_path, representation, range_flip, angle_flip, doppler_flip, sign):
    p = profile(representation=representation, range_reversed=range_flip,
                azimuth_reversed=angle_flip, doppler_reversed=doppler_flip, doppler_positive=sign)
    rng = np.random.default_rng(12)
    canonical = rng.uniform(.01, 10, p.shape)
    if representation == "complex":
        canonical = np.sqrt(canonical) * np.exp(1j * rng.uniform(-np.pi, np.pi, p.shape))
    native = canonical_to_native(canonical, p)
    assert native.shape == p.shape_native and native.flags.c_contiguous
    directory = tmp_path / "train/RAD/part1"
    directory.mkdir(parents=True)
    np.save(directory / "000000.npy", native)
    recovered = NativeReader(tmp_path, "raddet", p).load_tensor(0)
    np.testing.assert_allclose(recovered, canonical, rtol=1e-6, atol=1e-6)


def test_native_conversion_does_not_invent_phase_and_checks_storage_ranges():
    p = profile()
    with pytest.raises(ValueError, match="phase"):
        canonical_to_native(np.ones(p.shape), p)
    with pytest.raises(ValueError, match="storage range"):
        canonical_to_native(np.full(p.shape, 1e40 + 0j), p)
    with pytest.raises(ValueError, match="float32"):
        canonical_to_native(np.full(p.shape, 1e40), replace(p, representation="power"))
    raw = np.ones(p.shape)
    raw[0, 0, 0] = -1
    clipped = canonical_to_native(raw, replace(p, representation="power", range_reversed=False))
    assert clipped[0, 0, 0] == 0
    floored = canonical_to_native(np.zeros(p.shape), replace(p, representation="db"))
    np.testing.assert_allclose(floored, -120)


def test_raddet_writer_true_cartesian_footprint_and_rf_labels_roundtrip(tmp_path):
    p = profile()
    scene, refs, result = fixture(p)
    writer = ExportNativeWriter(tmp_path / "native", p)
    row = writer.write_frame("1", "train", scene, refs, result)
    manifest = writer.finalize()
    assert manifest["counts"] == {"train": 1, "val": 0, "test": 0}
    assert row["skipped_objects_without_in_domain_returns"] == [99]
    reader = NativeReader(tmp_path / "native", "raddet", p, trusted_labels=True)
    np.testing.assert_allclose(reader.load_tensor(0), result.tensor, rtol=1e-6, atol=1e-6)
    label = reader.load_annotations(0)
    corners = scene.objects[0].corners()
    expected = np.r_[corners.min(axis=0), corners.max(axis=0)]
    np.testing.assert_allclose(label["boxes_bev"][0], expected, atol=2e-6)
    assert label["labels"] == ["car"]
    assert not label["geometry_approximation"]
    with (tmp_path / "native/train/gt/part1/000001.pickle").open("rb") as stream:
        native = pickle.load(stream)
    assert native["boxes"].shape == (1, 6) and native["cart_boxes"].shape == (1, 4)
    assert np.all(native["boxes"][:, 3:] >= 1)
    assert np.all((native["boxes"][:, 2] >= 0) & (native["boxes"][:, 2] < p.shape_native[2]))


def test_raddet_export_requires_explicit_physical_affine(tmp_path):
    p = replace(profile(), cartesian_affine=None)
    with pytest.raises(ValueError, match="cartesian_affine"):
        ExportNativeWriter(tmp_path / "new", p)
    assert not (tmp_path / "new").exists()


def test_carrada_writer_frame_labels_roundtrip_and_explicit_support_domain(tmp_path):
    p = profile("carrada")
    scene, refs, result = fixture(p)
    out = tmp_path / "native"
    writer = ExportNativeWriter(out, p, write_views=True)
    row = writer.write_frame(0, "train", scene, refs, result, sequence="synthetic_train_000")
    writer.finalize()
    reader = NativeReader(out, "carrada", p)
    assert reader.records[0].sequence == "synthetic_train_000"
    np.testing.assert_allclose(reader.load_tensor(0), result.tensor, rtol=1e-6, atol=1e-6)
    label = reader.load_annotations(0)
    assert label["labels"] == ["car"] and label["geometry_approximation"]
    assert label["label_domain"] == "Cartesian_enclosure_of_CARRADA_radar_annotation_support"
    assert not label["orientation_labels_available"]
    assert np.all(label["boxes_bev"][0, 2:] > label["boxes_bev"][0, :2])
    annotations = json.loads((out / "annotations_frame_oriented.json").read_text())
    views = annotations["synthetic_train_000"]["000000"]["car_7"]
    assert set(views) == {"range_angle", "range_doppler"}
    assert views["range_angle"]["label"] == 3
    assert len(views["range_angle"]["dense"]) >= len(views["range_angle"]["sparse"])
    # Native annotation Doppler is approaching-positive, so a receding car's
    # label lies below the zero-Doppler midpoint, independently of tensor flips.
    sparse_rd = np.asarray(views["range_doppler"]["sparse"])
    assert np.all(sparse_rd[:, 1] < p.shape_native[2] // 2)
    sparse_ra = np.asarray(views["range_angle"]["sparse"])
    expected_row = p.shape_native[0] - np.trunc(np.hypot(refs.xyz[:, 0], refs.xyz[:, 1]) / p.range_step).astype(int)
    assert set(sparse_ra[:, 0]) == set(expected_row)
    assert (out / "synthetic_train_000/range_angle_numpy/000000.npy").is_file()
    assert json.loads((out / "data_seq_ref.json").read_text())["synthetic_train_000"]["labels"] == [3]


def test_native_writer_refuses_existing_output_duplicates_and_cross_split_sequences(tmp_path):
    p = profile()
    scene, refs, result = fixture(p)
    out = tmp_path / "native"
    writer = ExportNativeWriter(out, p)
    writer.write_frame("0", "train", scene, refs, result)
    original = (out / "train/RAD/part1/000000.npy").read_bytes()
    with pytest.raises(FileExistsError):
        writer.write_frame("000000", "test", scene, refs, result)
    with pytest.raises(FileExistsError):
        ExportNativeWriter(out, p)
    with pytest.raises(ValueError, match="integer"):
        writer.write_frame("../unsafe", "train", scene, refs, result)
    assert original == (out / "train/RAD/part1/000000.npy").read_bytes()
    cp = profile("carrada")
    cs, cr, co = fixture(cp)
    cw = ExportNativeWriter(tmp_path / "carrada", cp)
    cw.write_frame(0, "train", cs, cr, co, sequence="same_sequence")
    with pytest.raises(ValueError, match="across"):
        cw.write_frame(1, "test", cs, cr, co, sequence="same_sequence")
    cw.finalize()
    with pytest.raises(RuntimeError):
        cw.write_frame(2, "train", cs, cr, co)


def test_empty_return_objects_remain_unlabeled_in_both_native_schemas(tmp_path):
    from radsimreal.simulation.reflections import ReflectionPoints
    for dataset in ("raddet", "carrada"):
        p = profile(dataset)
        scene, _, _ = fixture(p)
        refs = ReflectionPoints(np.empty((0, 3)), np.empty((0, 3)), np.empty(0, dtype=np.complex128))
        cfg = p.radar_config()
        cfg.noise_std = 0
        result = simulate(refs, cfg)
        writer = ExportNativeWriter(tmp_path / dataset, p)
        row = writer.write_frame(1, "train", scene, refs, result)
        writer.finalize()
        assert row["labeled_objects"] == 0
        reader = NativeReader(tmp_path / dataset, dataset, p, trusted_labels=True)
        assert reader.load_annotations(0)["boxes_bev"].shape == (0, 4)


@pytest.mark.parametrize("dataset", ["raddet", "carrada"])
@pytest.mark.parametrize("metadata_style", ["explicit_bool", "boundary_fallback"])
def test_periodic_doppler_alias_keeps_visible_fast_objects_and_wraps_native_labels(tmp_path, dataset, metadata_style):
    p = profile(dataset)
    scene = Scene([Vehicle(0., 12., 4., 2., 0., 0., 24., object_id=7)])
    refs = ReflectionPoints(np.array([[0., 10., 0.]]), np.array([[0., 24., 0.]]),
                            np.array([1. + 0j]), np.array([7]))
    cfg = p.radar_config()
    cfg.noise_std = 0
    cfg.doppler_wrap = True
    result = simulate(refs, cfg, psf=np.ones((1, 3, 1)), rng=np.random.default_rng(7))
    assert result.metadata["in_domain_reflection_count"] == 1
    # Exercise both v2 explicit metadata and compatible older periodic-boundary metadata.
    if metadata_style == "explicit_bool":
        result.metadata["doppler_wrap"] = True
    else:
        result.metadata.pop("doppler_wrap", None)
    out = tmp_path / dataset
    writer = ExportNativeWriter(out, p)
    row = writer.write_frame(1, "train", scene, refs, result)
    writer.finalize()
    assert row["labeled_objects"] == 1
    assert row["skipped_objects_without_in_domain_returns"] == []
    assert row["doppler_wrap"] is True
    assert "full Doppler period" in row["radar_box_wrap_policy"]
    reader = NativeReader(out, dataset, p, trusted_labels=True)
    np.testing.assert_allclose(reader.load_tensor(0), result.tensor, rtol=1e-6, atol=1e-6)
    assert reader.load_annotations(0)["labels"] == ["car"]
    metadata = json.loads((out / row["metadata"]).read_text())
    assert metadata["objects"][0]["doppler_support_crosses_periodic_boundary"]
    if dataset == "raddet":
        with (out / row["annotation"]).open("rb") as stream:
            native = pickle.load(stream)
        assert native["boxes"][0, 5] == p.shape_native[2]
    else:
        annotation = json.loads((out / "annotations_frame_oriented.json").read_text())
        rd = annotation["synthetic_train_000"]["000001"]["car_7"]["range_doppler"]
        assert rd["periodic_doppler_columns"]
        assert np.asarray(rd["sparse"])[0, 1] == 0  # approaching -24 m/s aliases to native zero column
        assert {0, p.shape_native[2] - 1} <= set(np.asarray(rd["dense"])[:, 1])


@pytest.mark.parametrize("dataset", ["raddet", "carrada"])
def test_external_zero_coefficients_do_not_create_native_ground_truth_objects(tmp_path, dataset):
    p = profile(dataset)
    scene, refs, _ = fixture(p)
    refs.coefficients[:] = 0
    cfg = p.radar_config()
    cfg.noise_std = 0
    result = simulate(refs, cfg, rng=np.random.default_rng(1))
    writer = ExportNativeWriter(tmp_path / dataset, p)
    row = writer.write_frame(1, "train", scene, refs, result)
    writer.finalize()
    assert row["labeled_objects"] == 0
    assert row["zero_coefficient_reflectors_ignored"] == len(refs)
    assert set(row["skipped_objects_without_in_domain_returns"]) == {7, 99}
    reader = NativeReader(tmp_path / dataset, dataset, p, trusted_labels=True)
    assert reader.load_annotations(0)["labels"] == []
