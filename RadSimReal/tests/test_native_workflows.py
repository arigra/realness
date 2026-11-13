"""Complete native preparation workflows on controlled public-format fixtures."""
from dataclasses import replace
import hashlib
from pathlib import Path
import json
import numpy as np
import pytest

from native_fixtures import create_fixture
from radsimreal.native import NativeReader
from radsimreal.native.workflows import prepare_native, reader_from_prepared
from radsimreal.native.calibration import calibrate_native
from radsimreal.datasets.bev_dataset import BEVDataset
from radsimreal.config import load_config
from radsimreal.radar import load_psf


def file_hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("dataset", ["raddet", "carrada"])
def test_fixture_native_read_prepare_labels_and_no_source_writes(tmp_path, dataset):
    root, profile, selection, split_args = create_fixture(tmp_path, dataset)
    before = file_hashes(root)
    reader = NativeReader(root, dataset, profile, trusted_labels=dataset == "raddet")
    native = reader.load_tensor(reader.records[0])
    provenance = json.loads((root / "FIXTURE_PROVENANCE.json").read_text())
    assert np.unravel_index(np.abs(native).argmax(), native.shape) == tuple(provenance["reflector_center"])
    assert reader.profile.shape == (32, 8, 32)
    prepared = prepare_native(reader, tmp_path / "prepared", **split_args,
                              fit_normalization_frames=2, verify_all_labels=True)
    assert file_hashes(root) == before
    manifest = json.loads((prepared / "manifest.json").read_text())
    assert manifest["source"] == "native_reference"
    assert manifest["normalization"]["fit_source"] == "training_frames_only"
    assert manifest["normalization"]["frames_used"] == 2
    assert manifest["boundary_policy"] == "reject_outside_grid_no_silent_label_crop"
    assert manifest["counts"] == ({"train": 3, "val": 2, "test": 2} if dataset == "raddet" else {"train": 3, "val": 3, "test": 3})
    assert all(row.get("prepared_annotation") for rows in manifest["splits"].values() for row in rows)
    ds = BEVDataset(prepared, "train")
    item, raw = ds[0], ds.raw(0)
    assert item["image"].shape == (3, 32, 64)
    assert item["radar"].shape == (8, 32, 32)
    assert np.isfinite(item["image"].numpy()).all()
    assert item["boxes"].shape == (1, 4)
    assert item["oriented_boxes"].shape == (0, 5)
    np.testing.assert_allclose(ds.grid.denormalize_boxes(item["boxes"]).numpy(), item["boxes_physical"].numpy(), atol=5e-6)
    if dataset == "raddet":
        np.testing.assert_allclose(raw["boxes_physical"], [provenance["reference_car_box_bev"]], atol=1e-12)
        assert raw["label_domain"] == "native_RADDet_Cartesian_rectangles"
        assert len(item["boxes_rad"]) == 1
    else:
        assert raw["label_domain"] == "Cartesian_enclosure_of_CARRADA_radar_annotation_support"
        row = manifest["splits"]["train"][0]
        labels = json.loads((prepared / row["prepared_annotation"]).read_text())
        assert labels["geometry_approximation"]
        assert "dense/sparse" in labels["conversion_provenance"][0]["conversion"]
        assert not np.allclose(raw["boxes_physical"], [provenance["reference_car_box_bev"]])
    audit = json.loads((prepared / "label_audit.json").read_text())
    assert audit["frames_checked"] == len(reader)
    assert audit["cars"] == len(reader)


def test_prepare_raddet_requires_explicit_physical_affine(tmp_path):
    root, profile, _, split_args = create_fixture(tmp_path, "raddet")
    profile = replace(profile, cartesian_affine=None, cartesian_mapping_validated=False)
    reader = NativeReader(root, "raddet", profile, trusted_labels=True)
    with pytest.raises(ValueError, match="explicit Cartesian"):
        prepare_native(reader, tmp_path / "prepared", **split_args)


def test_lazy_manifest_index_excludes_added_files_and_cached_labels_avoid_redecode(tmp_path, monkeypatch):
    root, profile, _, split_args = create_fixture(tmp_path, "raddet")
    reader = NativeReader(root, "raddet", profile, trusted_labels=True)
    prepared = prepare_native(reader, tmp_path / "prepared", **split_args, fit_normalization_frames=1)
    new_path = root / "train/RAD/part1/000443.npy"
    np.save(new_path, np.ones((32, 32, 8), dtype=np.complex64))
    import radsimreal.native.readers as module
    monkeypatch.setattr(module, "discover_records", lambda *args: (_ for _ in ()).throw(AssertionError("unexpected raw rescan")))
    ds = BEVDataset(prepared, "train", include_radar=False)
    assert len(ds) == 3
    assert len(ds.native_reader) == 7
    monkeypatch.setattr(ds.native_reader, "load_annotations", lambda *args: (_ for _ in ()).throw(AssertionError("prepared labels should be cached")))
    assert ds[1]["boxes_physical"].shape == (1, 4)
    restored, manifest = reader_from_prepared(prepared)
    assert len(restored) == 7


@pytest.mark.parametrize("dataset", ["raddet", "carrada"])
def test_validation_and_test_values_cannot_change_fitted_training_normalization(tmp_path, dataset):
    root_a, profile_a, _, split_a = create_fixture(tmp_path / "a", dataset)
    root_b, profile_b, _, split_b = create_fixture(tmp_path / "b", dataset)
    reader_b = NativeReader(root_b, dataset, profile_b, trusted_labels=True)
    for record in reader_b.records:
        heldout = record.sequence in ({"scene_0", "scene_9", "scene_11"} if dataset == "raddet" else {"val_seq", "test_seq"})
        if heldout:
            array = np.load(record.path, allow_pickle=False)
            np.save(record.path, array * 1e5, allow_pickle=False)
    out_a = prepare_native(NativeReader(root_a, dataset, profile_a, trusted_labels=True),
                           tmp_path / "prepared_a", **split_a, fit_normalization_frames=2)
    out_b = prepare_native(reader_b, tmp_path / "prepared_b", **split_b, fit_normalization_frames=2)
    norm_a = json.loads((out_a / "manifest.json").read_text())["normalization"]
    norm_b = json.loads((out_b / "manifest.json").read_text())["normalization"]
    assert norm_a == norm_b


@pytest.mark.parametrize("dataset", ["raddet", "carrada"])
def test_fixture_selection_calibrates_only_declared_train_frames(tmp_path, dataset):
    root, profile, selection, split_args = create_fixture(tmp_path, dataset)
    reader = NativeReader(root, dataset, profile, trusted_labels=True)
    prepared = prepare_native(reader, tmp_path / "prepared", **split_args, fit_normalization_frames=1)
    manifest = json.loads((prepared / "manifest.json").read_text())
    allowed = {row["id"] for row in manifest["splits"]["train"]}
    assert {row["record_id"] for row in selection["targets"]} == allowed
    path = calibrate_native(reader, selection, tmp_path / "calibration", allowed_ids=allowed)
    cfg = load_config(path)
    kernel, metadata = load_psf(cfg.radar.psf_path, return_metadata=True, config=cfg.radar, strict_metadata=True)
    assert cfg.radar.doppler_wrap
    assert metadata["frames_averaged"] == 3
    assert metadata["retained_energy_fraction"] >= .99
    assert abs(kernel[tuple(np.array(kernel.shape) // 2)] - 1) < 1e-12
    truth = json.loads((root / "FIXTURE_PROVENANCE.json").read_text())
    np.testing.assert_allclose(metadata["noise"]["quadrature_covariance"], truth["noise_covariance"], rtol=.4, atol=.0004)


def test_carrada_missing_frame_excluded_not_empty_but_remains_train_calibration_input(tmp_path):
    root, profile, selection, split_args = create_fixture(tmp_path, "carrada")
    annotation_path = root / "annotations_frame_oriented.json"
    annotations = json.loads(annotation_path.read_text())
    annotations["train_seq"].pop("000000")
    # An explicitly present empty dictionary is a genuine negative example.
    annotations["val_seq"]["000001"] = {}
    annotation_path.write_text(json.dumps(annotations))
    before = file_hashes(root)
    reader = NativeReader(root, "carrada", profile)
    prepared = prepare_native(reader, tmp_path / "prepared", **split_args, fit_normalization_frames=1)
    manifest = json.loads((prepared / "manifest.json").read_text())
    assert manifest["source_counts_before_annotation_filter"] == {"train": 3, "val": 3, "test": 3}
    assert manifest["counts"] == {"train": 2, "val": 3, "test": 3}
    assert manifest["excluded_record_counts_by_split"] == {"train": 1, "val": 0, "test": 0}
    excluded = manifest["excluded_records"]
    assert len(excluded) == 1
    assert excluded[0]["id"] == "carrada/train_seq/000000"
    assert not excluded[0]["treated_as_empty_ground_truth"]
    assert excluded[0]["eligible_for_training_sequence_calibration"]
    assert len(manifest["native_records"]) == 9
    calibration_ids = set(manifest["calibration_record_ids"])
    assert calibration_ids == {f"carrada/train_seq/{i:06d}" for i in range(3)}
    assert not calibration_ids & {r["id"] for key in ("val", "test") for r in manifest["splits"][key]}
    train = BEVDataset(prepared, "train", include_radar=False)
    val = BEVDataset(prepared, "val", include_radar=False)
    assert len(train) == 2
    negative = next(i for i, row in enumerate(val.samples) if row["id"].endswith("000001"))
    assert val[negative]["boxes"].shape == (0, 4)
    # Measurement code does not require a car label for the excluded TRAIN frame.
    path = calibrate_native(reader, selection, tmp_path / "calibration", allowed_ids=calibration_ids)
    assert load_config(path).radar.doppler_wrap
    assert file_hashes(root) == before


def test_carrada_all_unannotated_training_frames_fail_after_filtering(tmp_path):
    root, profile, _, split_args = create_fixture(tmp_path, "carrada")
    path = root / "annotations_frame_oriented.json"
    annotations = json.loads(path.read_text())
    annotations["train_seq"] = {}
    path.write_text(json.dumps(annotations))
    before = file_hashes(root)
    with pytest.raises(ValueError, match="filtering left an empty training"):
        prepare_native(NativeReader(root, "carrada", profile), tmp_path / "prepared", **split_args)
    assert not (tmp_path / "prepared/manifest.json").exists()
    assert file_hashes(root) == before


@pytest.mark.parametrize("defect,expected", [("null_frame", ValueError), ("missing_label", ValueError),
                                            ("missing_view", ValueError), ("null_instance", ValueError),
                                            ("null_view", ValueError), ("missing_sequence", KeyError),
                                            ("missing_file", FileNotFoundError)])
def test_carrada_annotation_schema_or_release_errors_are_not_filtered_as_missing_frames(tmp_path, defect, expected):
    root, profile, _, split_args = create_fixture(tmp_path, "carrada")
    path = root / "annotations_frame_oriented.json"
    annotations = json.loads(path.read_text())
    if defect == "null_frame":
        annotations["train_seq"]["000000"] = None
    elif defect == "missing_label":
        annotations["train_seq"]["000000"]["car_0"]["range_angle"].pop("label")
    elif defect == "missing_view":
        annotations["train_seq"]["000000"]["car_0"].pop("range_angle")
    elif defect == "null_instance":
        annotations["train_seq"]["000000"]["car_0"] = None
    elif defect == "null_view":
        annotations["train_seq"]["000000"]["car_0"]["range_angle"] = None
    elif defect == "missing_sequence":
        annotations.pop("train_seq")
    if defect == "missing_file":
        path.unlink()
    else:
        path.write_text(json.dumps(annotations))
    before = file_hashes(root)
    with pytest.raises(expected):
        prepare_native(NativeReader(root, "carrada", profile), tmp_path / "prepared", **split_args)
    assert not (tmp_path / "prepared/manifest.json").exists()
    assert file_hashes(root) == before


def test_quick_carrada_label_check_still_excludes_unannotated_tail(tmp_path):
    root, profile, _, split_args = create_fixture(tmp_path, "carrada")
    path = root / "annotations_frame_oriented.json"
    annotations = json.loads(path.read_text())
    source = root / "test_seq/RAD_numpy/000000.npy"
    values = np.load(source, allow_pickle=False)
    for frame in range(3, 8):
        np.save(source.with_name(f"{frame:06d}.npy"), values, allow_pickle=False)
        if frame < 7:
            annotations["test_seq"][f"{frame:06d}"] = annotations["test_seq"]["000000"]
    path.write_text(json.dumps(annotations))
    reader = NativeReader(root, "carrada", profile)
    before = file_hashes(root)
    out = prepare_native(reader, tmp_path / "prepared", **split_args,
                          verify_all_labels=False, fit_normalization_frames=1)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"] == {"train": 3, "val": 3, "test": 7}
    assert manifest["excluded_records"][0]["id"] == "carrada/test_seq/000007"
    assert not manifest["excluded_records"][0]["eligible_for_training_sequence_calibration"]
    assert manifest["annotation_validation"]["all_carrada_frame_availability_checked"]
    assert not manifest["annotation_validation"]["full_label_check"]
    assert manifest["annotation_validation"]["checked_annotated_frames"] == 12
    assert len(manifest["native_records"]) == 14
    assert file_hashes(root) == before


def test_raddet_missing_label_file_still_fails_instead_of_excluding_frame(tmp_path):
    root, profile, _, split_args = create_fixture(tmp_path, "raddet")
    (root / "train/gt/part1/000440.pickle").unlink()
    reader = NativeReader(root, "raddet", profile, trusted_labels=True)
    with pytest.raises(FileNotFoundError, match="Missing native RADDet label"):
        prepare_native(reader, tmp_path / "prepared", **split_args)
    assert not (tmp_path / "prepared/manifest.json").exists()


@pytest.mark.parametrize("dataset", ["raddet", "carrada"])
def test_ra_cache_gives_identical_bev_items(tmp_path, dataset):
    import subprocess, sys, torch
    from radsimreal.datasets.bev_dataset import BEVDataset
    root, profile, _, split_args = create_fixture(tmp_path, dataset)
    reader = NativeReader(root, dataset, profile, trusted_labels=dataset == "raddet")
    prepared = prepare_native(reader, tmp_path / "prepared", **split_args, fit_normalization_frames=2)
    direct = BEVDataset(prepared, "train", include_radar=False)
    expected = [direct[i] for i in range(len(direct))]
    script = Path(__file__).resolve().parents[1] / "scripts" / "cache_ra_images.py"
    subprocess.run([sys.executable, str(script), "--data", str(prepared), "--workers", "2"], check=True)
    cached = BEVDataset(prepared, "train", include_radar=False)
    assert cached._cached_image(cached.samples[0]["id"]) is not None
    for i, item in enumerate(expected):
        torch.testing.assert_close(cached[i]["image"], item["image"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(cached[i]["boxes"], item["boxes"])
    # A model that consumes the full radar tensor never uses the cache.
    assert BEVDataset(prepared, "train", include_radar=True)._cached_image is not None
    assert "radar" in BEVDataset(prepared, "train", include_radar=True)[0]
