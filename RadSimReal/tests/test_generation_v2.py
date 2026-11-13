"""Full native-calibration-to-mesh-generation regression checks."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from native_fixtures import create_fixture
from radsimreal.config import load_config, RadarConfig, SceneConfig, SimulationConfig
from radsimreal.native import NativeReader
from radsimreal.native.workflows import make_reader, prepare_native, default_bev_grid, assert_sensor_sampling
from radsimreal.native.calibration import calibrate_native
from radsimreal.datasets.generation_v2 import generate_bev_dataset
from radsimreal.datasets.bev_dataset import BEVDataset
from radsimreal.bev.models import create_model
from radsimreal.bev.losses import bev_detection_loss


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("dataset", ["raddet", "carrada"])
def test_native_calibrated_mesh_generation_and_native_readback(tmp_path, dataset):
    raw, profile, selection, split_args = create_fixture(tmp_path / "source", dataset)
    reader = NativeReader(raw, dataset, profile, trusted_labels=dataset == "raddet")
    prepared = prepare_native(reader, tmp_path / "prepared", fit_normalization_frames=2, **split_args)
    native_manifest = json.loads((prepared / "manifest.json").read_text())
    sensor = calibrate_native(reader, selection, tmp_path / "sensor", allowed_ids=native_manifest["calibration_record_ids"])
    config = load_config(sensor)
    config.scene.max_vehicles = 2
    generated = generate_bev_dataset(tmp_path / "generated", config, {"train": 2, "val": 1, "test": 1},
        grid=default_bev_grid(profile), normalization=native_manifest["normalization"],
        native_profile=profile, max_samples=800)
    ds = BEVDataset(generated, "train")
    item = ds[0]
    assert item["image"].shape == (3, 32, 64)
    assert item["radar"].shape == (8, 32, 32)
    assert torch.isfinite(item["image"]).all()
    assert torch.isfinite(item["radar"]).all()
    assert len(item["boxes_physical"]) > 0
    assert (generated / "calibration.npz").is_file()
    assert (generated / "noise_bank.npz").is_file()
    reloaded = load_config(generated / "config.json")
    assert Path(reloaded.radar.psf_path).parent == generated
    assert Path(reloaded.radar.noise_bank_path).parent == generated
    native_reader = NativeReader(generated / "native_export", dataset, profile, trusted_labels=dataset == "raddet")
    native_row = ds.samples[0]["native_export"]
    rec = next(r for r in native_reader.records if r.path.resolve() == (generated / "native_export" / native_row["tensor"]).resolve())
    native_labels = native_reader.load_annotations(rec)
    np.testing.assert_allclose(native_labels["boxes_bev"], item["boxes_physical"], rtol=2e-6, atol=1e-5)
    if dataset == "raddet":
        canonical_rad = ds._canonical_rad_boxes(native_labels["native_rad_boxes_center_size"], profile)
        np.testing.assert_allclose(canonical_rad, item["boxes_rad"], rtol=1e-6, atol=1e-6)
        assert len(item["oriented_boxes"]) == len(item["boxes"])
    else:
        assert len(item["oriented_boxes"]) == 0
        assert ds.label_domain == "Cartesian_enclosure_of_CARRADA_radar_annotation_support"
        model = create_model("probabilistic", grid=ds.grid, base_channels=2, oriented=True)
        with pytest.raises(ValueError, match="genuine|one-to-one"):
            bev_detection_loss(model(item["image"][None]), [item["boxes"]], [item["oriented_boxes"]], ds.grid)


def test_profile_representation_override_and_full_axis_guard(tmp_path):
    raw, profile, _, _ = create_fixture(tmp_path, "raddet")
    profile.representation = None
    profile.save(tmp_path / "profile.json")
    reader = make_reader(raw, "raddet", profile=tmp_path / "profile.json", representation="complex")
    assert reader.profile.representation == "complex"
    config = reader.profile.radar_config()
    assert_sensor_sampling(config, reader.profile)
    config.velocity_min += config.velocity_step
    with pytest.raises(ValueError, match="origins"):
        assert_sensor_sampling(config, reader.profile)


def test_static_mesh_ids_cannot_create_phantom_vehicle_labels(tmp_path):
    # A large RF-visible foreground wall hides all vehicle geometry. Its default
    # OBJ ID0 must be remapped, never used as a return from vehicle0.
    wall = tmp_path / "wall.obj"
    wall.write_text("v -100 3 0\nv 100 3 0\nv 100 3 10\nv -100 3 10\nf 1 2 3 4\n")
    cfg = SimulationConfig(RadarConfig(range_bins=32, azimuth_bins=32, doppler_bins=8, range_step=.5),
        SceneConfig(min_vehicles=1, max_vehicles=1, min_range=7, max_range=12, lateral_limit=2))
    out = generate_bev_dataset(tmp_path / "out", cfg, {"train": 1, "val": 0, "test": 0},
                               static_mesh_path=wall, max_samples=1000)
    record = json.loads((out / "train/train_000000.json").read_text())
    assert record["objects"] == []
    assert record["reflection_metadata"]["nonzero_return_count"] > 0


def test_cad_asset_example_is_executable_and_preserves_cartesian_targets(tmp_path):
    project = Path(__file__).resolve().parents[1]
    config = load_config(project / "configs/cartesian_demo.yaml")
    out = generate_bev_dataset(tmp_path / "cad", config, {"train": 1, "val": 0, "test": 0},
        vehicle_assets_path=project / "configs/assets/vehicle_assets.json", max_samples=800)
    sample = BEVDataset(out)[0]
    assert len(sample["boxes_physical"])
    data = json.loads((out / "train/train_000000.json").read_text())
    assert data["reflection_metadata"]["geometry"]["vehicle_placements"]
    altered = data
    altered["objects"][0]["box_bev"][0] = -1000
    (out / "train/train_000000.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="Expand"):
        BEVDataset(out)[0]
