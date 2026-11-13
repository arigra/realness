"""Native-layout fixture tests; no access to user radar recordings is implied."""
from dataclasses import replace
import json
import pickle
import numpy as np
import pytest

from radsimreal.native import (NativeReader, NativeProfile, discover_records, make_native_manifest,
                               reference_profile, raddet_reference_cartesian_affine)
from radsimreal.native.annotations import polar_region_enclosure, load_json_sequence


def write_raddet(root, frame=0, tensor=None, labels=None, subset="train"):
    folder = root / subset / "RAD" / "part1"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{frame:06d}.npy"
    if tensor is None:
        tensor = np.ones((8, 10, 6), dtype=np.complex64)
    np.save(path, tensor)
    gt = root / subset / "gt" / "part1"
    gt.mkdir(parents=True, exist_ok=True)
    if labels is None:
        labels = {"classes": ["car", "person"],
                  "boxes": [[2, 5, 0, 2, 2, 2], [5, 3, 1, 1, 1, 1]],
                  "cart_boxes": [[4, 10, 2, 4], [2, 6, 1, 1]]}
    with (gt / f"{frame:06d}.pickle").open("wb") as stream:
        pickle.dump(labels, stream)
    return path


def write_carrada(root, sequence="seq1", tensor=None, frame="000001", annotations=None):
    folder = root / sequence / "RAD_numpy"
    folder.mkdir(parents=True, exist_ok=True)
    if tensor is None:
        tensor = np.ones((8, 10, 6), dtype=np.complex64)
    np.save(folder / f"{frame}.npy", tensor)
    if annotations is not None:
        (root / "annotations_frame_oriented.json").write_text(json.dumps(annotations), encoding="utf-8")


def test_profile_grid_matches_canonical_radar_config_and_roundtrip(tmp_path):
    p = reference_profile("raddet", (8, 10, 6), "complex")
    p.save(tmp_path / "profile.json")
    loaded = NativeProfile.load(tmp_path / "profile.json")
    assert loaded.to_dict() == p.to_dict()
    cfg = p.radar_config()
    assert cfg.shape == (8, 6, 10)
    np.testing.assert_allclose(cfg.angular_grid, p.angular_grid, atol=1e-14)
    np.testing.assert_allclose(cfg.ranges, p.ranges)
    np.testing.assert_allclose(cfg.velocities, p.velocities)
    assert p.cartesian_affine is None


def test_raddet_canonical_tensor_axes_range_flip_and_ids(tmp_path):
    native = (np.arange(8 * 10 * 6).reshape(8, 10, 6) + 1j).astype(np.complex64)
    write_raddet(tmp_path, 6609, native, subset="test")
    reader = NativeReader(tmp_path, "raddet")
    record = reader.records[0]
    assert record.frame_id == 6609
    assert record.sequence == "scene_9"
    assert record.split == "test"
    assert reader.profile.representation == "complex"
    np.testing.assert_array_equal(reader.load_tensor(record), native[::-1].transpose(0, 2, 1))
    np.testing.assert_allclose(reader.load_power(0), np.abs(native[::-1].transpose(0, 2, 1).astype(np.complex128)) ** 2)


def test_carrada_velocity_sign_and_asymmetric_fft_origin(tmp_path):
    native = np.arange(8 * 10 * 6).reshape(8, 10, 6).astype(np.complex64)
    write_carrada(tmp_path, tensor=native)
    reader = NativeReader(tmp_path, "carrada")
    assert reader.profile.doppler_positive == "approaching"
    expected = native[::-1, :, ::-1].transpose(0, 2, 1)
    np.testing.assert_array_equal(reader.load_tensor(0), expected)
    dv = reader.profile.velocity_step
    np.testing.assert_allclose(reader.profile.velocities, np.arange(-2, 4) * dv)
    assert reader.profile.radar_config().velocity_min == pytest.approx(-2 * dv)


def test_real_data_convention_is_required_and_integer_amplitude_safe(tmp_path):
    write_raddet(tmp_path, tensor=np.full((8, 10, 6), 200, dtype=np.uint8))
    with pytest.raises(ValueError, match="ambiguous"):
        NativeReader(tmp_path, "raddet")
    profile = reference_profile("raddet", (8, 10, 6), "amplitude")
    reader = NativeReader(tmp_path, "raddet", profile)
    np.testing.assert_array_equal(reader.load_tensor(0), 40000)
    db = replace(profile, representation="db")
    np.save(reader.records[0].path, np.full((8, 10, 6), 10.0))
    np.testing.assert_allclose(NativeReader(tmp_path, "raddet", db).load_tensor(0), 10.0)


def test_shape_and_numeric_validation(tmp_path):
    path = write_raddet(tmp_path)
    reader = NativeReader(tmp_path, "raddet")
    np.save(path, np.ones((7, 10, 6), dtype=np.complex64))
    with pytest.raises(ValueError, match="shape"):
        reader.load_tensor(0)
    np.save(path, np.full((8, 10, 6), np.nan, dtype=np.complex64))
    with pytest.raises(ValueError, match="Nonfinite"):
        reader.load_tensor(0)


def test_cartesian_native_labels_preserved_and_affine_explicit(tmp_path):
    write_raddet(tmp_path)
    reader = NativeReader(tmp_path, "raddet")
    with pytest.raises(ValueError, match="trusted_labels"):
        reader.load_annotations(0)
    reader.trusted_labels = True
    label = reader.load_annotations(0)
    np.testing.assert_array_equal(label["boxes_cartesian_native"], [[8, 3, 12, 5]])
    assert label["requires_cartesian_affine"]
    assert label["boxes_bev"].shape == (0, 4)
    assert label["oriented_boxes_bev"].shape == (0, 5)
    assert label["label_domain"] == "native_RADDet_Cartesian_rectangles"
    reader.profile = replace(reader.profile, cartesian_affine={"x_from_col_scale": .2,
        "x_from_col_offset": -2, "y_from_row_scale": -.2, "y_from_row_offset": 2})
    label = reader.load_annotations(0)
    np.testing.assert_allclose(label["boxes_bev"], [[-.4, 1., .4, 1.4]])
    assert not label["cartesian_mapping_validated"]
    assert len(label["boxes_ra"]) == 1


def test_explicit_raddet_reference_cartesian_mapping_matches_public_helper():
    p = reference_profile("raddet", representation="complex")
    affine = raddet_reference_cartesian_affine(p)
    dr = p.range_step
    forward, lateral = 20 * dr, -10 * dr
    row = p.shape_native[0] - round(forward / dr) - 1
    col = round((lateral + 50) / dr) - 1
    assert col * affine["x_from_col_scale"] + affine["x_from_col_offset"] == pytest.approx(lateral)
    assert row * affine["y_from_row_scale"] + affine["y_from_row_offset"] == pytest.approx(forward)


def test_raddet_missing_cartesian_ground_truth_is_not_replaced(tmp_path):
    write_raddet(tmp_path, labels={"classes": ["car"], "boxes": [[2, 5, 0, 2, 2, 2]]})
    reader = NativeReader(tmp_path, "raddet", trusted_labels=True)
    with pytest.raises(ValueError, match="cart_boxes"):
        reader.load_annotations(0)


def test_carrada_dense_support_labels_are_region_enclosures_not_yaw(tmp_path):
    annotations = {"seq1": {"000001": {
        "car1": {"range_angle": {"label": 3, "box": [[3, 4], [5, 6]],
                                    "dense": [[3, 4], [4, 5], [5, 6]]}},
        "ped": {"range_angle": {"label": 1, "box": [[1, 2], [2, 3]]}}}}}
    write_carrada(tmp_path, annotations=annotations)
    reader = NativeReader(tmp_path, "carrada")
    label = reader.load_annotations(0)
    assert label["labels"] == ["car"]
    assert label["geometry_approximation"]
    assert not label["orientation_labels_available"]
    assert label["oriented_boxes_bev"].shape == (0, 5)
    p = reader.profile
    support = np.array([[3, 4], [4, 5], [5, 6]])
    ranges = (8 - support[:, 0]) * p.range_step
    theta = (support[:, 1] - 5) * p.annotation_angle_step
    xy = np.c_[ranges * np.sin(theta), ranges * np.cos(theta)]
    np.testing.assert_allclose(label["boxes_bev"][0], np.r_[xy.min(0), xy.max(0)])
    np.testing.assert_allclose(label["boxes_ra"], [[4, 2, 6, 4]])


def test_carrada_sector_enclosure_includes_interior_angle_extremum(tmp_path):
    box = polar_region_enclosure(10, 20, -.2, .2)
    assert box[3] == pytest.approx(20)
    assert box[1] == pytest.approx(10 * np.cos(.2))
    annotations = {"seq1": {"000001": {"car1": {"range_angle": {
        "label": "car", "box": [[3, 4], [5, 6]]}}}}}
    write_carrada(tmp_path, annotations=annotations)
    label = NativeReader(tmp_path, "carrada").load_annotations(0)
    assert "annular sector" in label["conversion_provenance"][0]["conversion"]
    assert label["boxes_bev"][0, 3] == pytest.approx((8 - 3) * .1953125)


@pytest.mark.parametrize("coordinate", ["sin", "angle"])
@pytest.mark.parametrize("azimuth_reversed", [False, True])
def test_carrada_box_only_sector_uses_same_declared_azimuth_mapping_as_dense_points(
        tmp_path, coordinate, azimuth_reversed):
    corners = [[10, 3], [10, 7], [12, 3], [12, 7]]
    annotations = {"seq1": {"000001": {
        "box_only": {"range_angle": {"label": 3, "box": [[10, 3], [12, 7]]}},
        "dense_corners": {"range_angle": {"label": 3, "box": [[10, 3], [12, 7]],
                                            "dense": corners}}}}}
    write_carrada(tmp_path, tensor=np.ones((32, 32, 8), dtype=np.complex64), annotations=annotations)
    profile = replace(reference_profile("carrada", (32, 32, 8), "complex"),
                      range_step=1.0, azimuth_coordinate=coordinate,
                      angular_min=-.9, angular_step=1.8 / 31,
                      azimuth_reversed=azimuth_reversed, annotation_angle_step=None)
    labels = NativeReader(tmp_path, "carrada", profile).load_annotations(0)
    angular_endpoints = np.array([-.9 + col * 1.8 / 31
                                 for col in ([24, 28] if azimuth_reversed else [3, 7])])
    angles = np.arcsin(angular_endpoints) if coordinate == "sin" else angular_endpoints
    expected = polar_region_enclosure(20.0, 22.0, float(angles.min()), float(angles.max()))
    np.testing.assert_allclose(labels["boxes_bev"], np.tile(expected, (2, 1)), atol=1e-12)
    assert (labels["boxes_bev"][:, 0] > 0).all() if azimuth_reversed else (labels["boxes_bev"][:, 2] < 0).all()
    if azimuth_reversed and coordinate == "sin":
        assert labels["boxes_bev"][0, 0] == pytest.approx(9.870967741935486)
        assert labels["boxes_bev"][0, 2] == pytest.approx(15.96774193548387)


def test_carrada_missing_frame_is_not_empty_scene(tmp_path):
    write_carrada(tmp_path, annotations={"seq1": {"000002": {}}})
    with pytest.raises(KeyError, match="missing"):
        NativeReader(tmp_path, "carrada").load_annotations(0)


def test_separate_carrada_rad_and_annotation_roots(tmp_path):
    tensors = tmp_path / "RAD_archive"
    write_carrada(tensors)
    labels = tmp_path / "original/annotations_frame_oriented.json"
    labels.parent.mkdir()
    labels.write_text(json.dumps({"seq1": {"000001": {}}}))
    reader = NativeReader(tensors, "carrada", annotation_path=labels)
    result = reader.load_annotations(0)
    assert result["boxes_bev"].shape == (0, 4)


def test_sequence_json_streaming_crosses_chunks_and_preserves_only_selected(tmp_path):
    path = tmp_path / "annotations.json"
    data = {"large_first": {"filler": "x" * (1024 * 1024 + 17)}, "target": {"000001": {}}, "last": {}}
    path.write_text(json.dumps(data))
    assert load_json_sequence(path, "target") == {"000001": {}}
    with pytest.raises(KeyError):
        load_json_sequence(path, "absent")


def test_published_raddet_partition_uses_original_ids_and_scenes(tmp_path):
    for frame in (0, 440, 6609, 8635, 556):
        write_raddet(tmp_path, frame)
    reader = NativeReader(tmp_path, "raddet")
    manifest = make_native_manifest(reader, val_sequences=[1])
    assert manifest["counts"] == {"train": 1, "val": 1, "test": 2}
    assert [x["frame_id"] for x in manifest["unassigned_records"]] == [556]
    assert {x["sequence"] for x in manifest["splits"]["test"]} == {"scene_9", "scene_11"}
    with pytest.raises(ValueError, match="overlap"):
        make_native_manifest(reader, val_sequences=[9])


def test_partial_raddet_inventory_reports_missing_reference_test_scenes(tmp_path):
    for frame in (0, 440):
        write_raddet(tmp_path, frame)
    reader = NativeReader(tmp_path, "raddet")
    manifest = make_native_manifest(reader, val_sequences=[1])
    assert manifest["counts"] == {"train": 1, "val": 1, "test": 0}
    assert manifest["missing_reference_test_scenes"] == ["scene_11", "scene_9"]
    with pytest.raises(ValueError, match="absent"):
        make_native_manifest(reader, val_sequences=[1], test_sequences=[9])


def test_carrada_sequence_split_is_explicit_and_no_overlap(tmp_path):
    for sequence in ("a", "b", "c"):
        write_carrada(tmp_path, sequence)
    reader = NativeReader(tmp_path, "carrada")
    with pytest.raises(ValueError, match="explicit"):
        make_native_manifest(reader)
    manifest = make_native_manifest(reader, val_sequences=["b"], test_sequences=["c"])
    assert manifest["counts"] == {"train": 1, "val": 1, "test": 1}
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["a"], "val": ["b"], "test": ["b", "c"]}))
    with pytest.raises(ValueError, match="more than once"):
        make_native_manifest(reader, explicit_split_file=split)


def test_duplicate_original_raddet_frame_copy_is_rejected(tmp_path):
    write_raddet(tmp_path, 0, subset="train")
    write_raddet(tmp_path, 0, subset="test")
    with pytest.raises(ValueError, match="Duplicate"):
        discover_records(tmp_path, "raddet")


def test_reader_manifest_index_does_not_rediscover_new_raw_files(tmp_path, monkeypatch):
    write_raddet(tmp_path, 0)
    reader = NativeReader(tmp_path, "raddet")
    rows = [record.to_dict(reader.root) for record in reader.records]
    write_raddet(tmp_path, 1)
    import radsimreal.native.readers as module
    monkeypatch.setattr(module, "discover_records", lambda *args: (_ for _ in ()).throw(AssertionError("must not rediscover")))
    indexed = NativeReader.from_manifest(tmp_path, "raddet", rows, reader.profile, trusted_labels=True)
    assert len(indexed) == 1
    assert indexed.records[0].frame_id == 0
    assert indexed.load_tensor(0).shape == (8, 6, 10)
    assert indexed.load_annotations(0)["labels"] == ["car"]


def test_reader_manifest_rejects_escape_duplicates_and_identity_changes(tmp_path):
    root = tmp_path / "dataset"
    write_raddet(root, 0)
    reader = NativeReader(root, "raddet")
    row = reader.records[0].to_dict(root)
    outside = tmp_path / "outside.npy"
    np.save(outside, np.ones((8, 10, 6), dtype=np.complex64))
    with pytest.raises(ValueError, match="escapes"):
        NativeReader.from_manifest(root, "raddet", [dict(row, path="../outside.npy")], reader.profile)
    with pytest.raises(ValueError, match="unique"):
        NativeReader.from_manifest(root, "raddet", [row, row], reader.profile)
    with pytest.raises(ValueError, match="frame_id"):
        NativeReader.from_manifest(root, "raddet", [dict(row, frame_id=1)], reader.profile)
    with pytest.raises(ValueError, match="escapes"):
        NativeReader.from_manifest(root, "raddet", [dict(row, annotation_path="../outside.pickle")], reader.profile)


def test_reader_manifest_external_carrada_annotation_override_and_header_check(tmp_path):
    root = tmp_path / "tensors"
    write_carrada(root)
    reader = NativeReader(root, "carrada")
    rows = [r.to_dict(root) for r in reader.records]
    labels = tmp_path / "annotations.json"
    labels.write_text(json.dumps({"seq1": {"000001": {}}}))
    indexed = NativeReader.from_manifest(root, "carrada", rows, reader.profile,
                                         annotation_path=labels)
    assert indexed.load_annotations(0)["boxes_bev"].shape == (0, 4)
    wrong_profile = replace(reader.profile, shape_native=(9, 10, 6))
    with pytest.raises(ValueError, match="shape"):
        NativeReader.from_manifest(root, "carrada", rows, wrong_profile, annotation_path=labels)


def test_empty_native_index_requires_explicit_resolved_profile(tmp_path):
    p = reference_profile("raddet", (8, 10, 6), "complex")
    reader = NativeReader.from_manifest(tmp_path, "raddet", [], p)
    assert len(reader) == 0
    with pytest.raises(ValueError, match="explicit profile"):
        NativeReader.from_manifest(tmp_path, "raddet", [], None)


def test_carrada_processed_views_alone_are_not_called_full_tensor(tmp_path):
    folder = tmp_path / "sequence/range_angle_numpy"
    folder.mkdir(parents=True)
    np.save(folder / "000001.npy", np.zeros((256, 256)))
    with pytest.raises(FileNotFoundError, match="separate download"):
        discover_records(tmp_path, "carrada")


def test_raddet_label_found_in_doubly_nested_gt_part_folder(tmp_path):
    folder = tmp_path / "test" / "RAD" / "part_2"
    folder.mkdir(parents=True)
    np.save(folder / "006609.npy", np.ones((8, 10, 6), dtype=np.complex64))
    nested = tmp_path / "test" / "gt" / "part_2" / "part_2"
    nested.mkdir(parents=True)
    with (nested / "006609.pickle").open("wb") as stream:
        pickle.dump({"classes": [], "boxes": [], "cart_boxes": []}, stream)
    record = NativeReader(tmp_path, "raddet").records[0]
    assert record.annotation_path == nested / "006609.pickle"


def test_raddet_rad_boxes_use_absolute_native_doppler_indices():
    from radsimreal.datasets.bev_dataset import BEVDataset
    profile = reference_profile("raddet")
    r, a, d = profile.shape_native
    # Released labels reach the top Doppler bin (63.5) and bin 0; both must be valid.
    boxes = [[100.0, 120.0, d - 0.5, 10.0, 12.0, 4.0], [100.0, 120.0, 0.0, 10.0, 12.0, 4.0]]
    canonical = BEVDataset._canonical_rad_boxes(boxes, profile)
    expected_d = np.array([d - 0.5, 0.0])
    if profile.canonical_doppler_flip:
        expected_d = d - 1 - expected_d
    np.testing.assert_allclose(canonical[:, 2], np.mod(expected_d, d) / d, atol=1e-6)
    assert np.all((canonical[:, 2] >= 0) & (canonical[:, 2] < 1))
