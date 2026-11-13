"""Actual native CLI pilots and resumability on synthetic native-format files.

These small end-to-end checks validate software execution and saved experiment
state. They are not measurements on the real RADDet or CARRADA datasets.
"""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from native_fixtures import create_fixture
from radsimreal.native.cli import main
from radsimreal.native.workflows import prepare_native, reader_from_prepared
from radsimreal.native import NativeReader
from radsimreal.datasets.bev_dataset import BEVDataset


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hashes(root, pattern="*"):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(root).rglob(pattern) if p.is_file()}


def _pilot_arguments(root, dataset, output, *, selection=True):
    args = ["run", "--dataset", dataset, "--root", str(root),
            "--profile", str(root / "fixture_profile.json"),
            "--split-file", str(root / "fixture_split.json"), "--out", str(output),
            "--models", "unet", "--train", "1", "--val", "1", "--test", "1",
            "--epochs", "1", "--max-steps", "1", "--base-channels", "2",
            "--batch-size", "1", "--max-samples", "128", "--half-widths", "2", "1", "3",
            "--device", "cpu", "--matching-policy", "best_unmatched", "--ap-aggregation", "frame_mean"]
    if dataset == "raddet":
        args.append("--trusted-labels")
    if selection:
        args.extend(["--selection", str(root / "fixture_selection.json")])
    return args


def _assert_completed(output, dataset):
    report = _read(output / "run_report.json")
    assert report["status"] == "pilot_completed"
    assert report["published_result_reproduced"] is False
    assert report["calibration_selection_status"] == "confirmed_input_assumptions"
    assert set(report["stages"]) >= {"inspection", "native_preparation", "calibration",
                                     "simulation", "distribution_diagnostics",
                                     "synthetic_unet_native_test", "finished"}
    model_dir = output / "models/synthetic_unet"
    payload = torch.load(model_dir / "last.pt", map_location="cpu", weights_only=True)
    assert payload["global_step"] == 1
    assert len(payload["history"]) == 1
    assert payload["history"][0]["updates"] == 1
    assert payload["optimizer_state"]["state"]
    assert payload["model_config"]["name"] == "unet"
    metrics = _read(model_dir / "native_test_metrics.json")
    assert metrics["num_images"] == (2 if dataset == "raddet" else 3)
    assert metrics["matching_policy"] == "best_unmatched"
    assert metrics["ap_aggregation"] == "frame_mean"
    assert metrics["label_domain"] == ("native_RADDet_Cartesian_rectangles" if dataset == "raddet"
        else "Cartesian_enclosure_of_CARRADA_radar_annotation_support")
    generated = _read(output / "synthetic/manifest.json")
    assert {key: len(rows) for key, rows in generated["splits"].items()} == {"train": 1, "val": 1, "test": 1}
    assert (output / "synthetic/native_export/native_manifest.json").is_file()
    return payload


@pytest.mark.parametrize("dataset", ["raddet", "carrada"])
def test_native_cli_pilot_complete_and_resume_preserves_trained_checkpoint_bytes(tmp_path, monkeypatch, dataset):
    root, _, _, _ = create_fixture(tmp_path / "source", dataset, R=16, A=16, D=8)
    raw_before = _hashes(root)
    output = tmp_path / "pilot"
    args = _pilot_arguments(root, dataset, output)
    main(args)
    _assert_completed(output, dataset)
    before_weights = _hashes(output / "models", "*.pt")
    before_sensor = _hashes(output / "calibration")
    before_synthetic = _hashes(output / "synthetic")
    assert set(before_weights) == {"synthetic_unet/best.pt", "synthetic_unet/last.pt"}

    import radsimreal.native.runner as runner

    def unexpected_mutation(*args, **kwargs):
        raise AssertionError("A completed resume must not regenerate, recalibrate or update weights")

    for name in ("train_bev", "generate_bev_dataset", "calibrate_native"):
        monkeypatch.setattr(runner, name, unexpected_mutation)
    main([*args, "--resume-run"])
    _assert_completed(output, dataset)
    assert _hashes(output / "models", "*.pt") == before_weights
    assert _hashes(output / "calibration") == before_sensor
    assert _hashes(output / "synthetic") == before_synthetic
    assert _hashes(root) == raw_before


def test_native_runner_candidate_handoff_resumes_with_first_verified_selection_and_rejects_later_change(tmp_path):
    root, _, _, _ = create_fixture(tmp_path / "source", "raddet", R=16, A=16, D=8)
    raw_before = _hashes(root)
    output = tmp_path / "pilot"
    initial = _pilot_arguments(root, "raddet", output, selection=False)
    main(initial)
    report = _read(output / "run_report.json")
    assert report["status"] == "needs_verified_calibration_selection"
    assert (output / "candidates/selection_template.json").is_file()
    assert not (output / "calibration/sensor_config.json").exists()
    assert not (output / "models").exists()
    prepared_before = _hashes(output / "prepared_native")
    first_inputs = _read(output / "run_inputs.json")
    assert first_inputs["selection"] is None

    confirmed = _pilot_arguments(root, "raddet", output, selection=True)
    main([*confirmed, "--resume-run"])
    _assert_completed(output, "raddet")
    transition = _read(output / "run_input_transitions.json")
    assert len(transition) == 1
    assert transition[0]["previous"]["selection"] is None
    assert transition[0]["current"]["selection"] == str(root / "fixture_selection.json")
    assert _hashes(output / "prepared_native") == prepared_before
    recorded_selection = _read(output / "calibration/selection.json")
    recorded_inputs_bytes = (output / "run_inputs.json").read_bytes()
    weights_before = _hashes(output / "models", "*.pt")

    # Editing the same path must not silently reuse a sensor fitted from older
    # ROI contents; input file identity alone is insufficient provenance.
    selection_path = root / "fixture_selection.json"
    original_selection_bytes = selection_path.read_bytes()
    changed = _read(selection_path)
    changed["energy_fraction"] = .98
    selection_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="Resume-run settings differ"):
        main([*confirmed, "--resume-run"])
    assert _read(output / "calibration/selection.json") == recorded_selection
    assert (output / "run_inputs.json").read_bytes() == recorded_inputs_bytes
    assert _hashes(output / "models", "*.pt") == weights_before
    # Restore the intentional test edit before checking that the runner itself
    # never changed the original source archive or annotations.
    selection_path.write_bytes(original_selection_bytes)
    assert _hashes(root) == raw_before


@pytest.mark.parametrize("dataset", ["raddet", "carrada"])
def test_relative_prepared_directory_still_loads_after_working_directory_changes(tmp_path, monkeypatch, dataset):
    root, profile, _, splits = create_fixture(tmp_path / "source", dataset, R=16, A=16, D=8)
    monkeypatch.chdir(tmp_path)
    prepared = prepare_native(NativeReader(root.relative_to(tmp_path), dataset, profile, trusted_labels=True),
                              Path("prepared"), **splits, fit_normalization_frames=1)
    assert prepared == Path("prepared")
    manifest = _read(prepared / "manifest.json")
    assert Path(manifest["native_root"]).is_absolute()
    assert all(not Path(row["path"]).is_absolute() for row in manifest["native_records"])
    elsewhere = tmp_path / "another_working_directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    restored, _ = reader_from_prepared(tmp_path / "prepared")
    assert restored.root == root
    sample = BEVDataset(tmp_path / "prepared", "train", include_radar=False)[0]
    assert sample["boxes_physical"].shape == (1, 4)
    assert tuple(sample["image"].shape) == (3, 16, 32)
