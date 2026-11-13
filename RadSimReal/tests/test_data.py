import json
import numpy as np
import pytest
from radsimreal.config import RadarConfig, SceneConfig, SimulationConfig, load_config
from radsimreal.datasets.loader import preprocess_image, RadarDataset, collate_fn
from radsimreal.datasets import generate_dataset
from radsimreal.datasets.external import import_manifest


def small_config():
    return SimulationConfig(RadarConfig(range_bins=32, azimuth_bins=32, doppler_bins=8, range_step=1.5),
                            SceneConfig(max_range=40, max_vehicles=2, points_per_vehicle=12, clutter_points=3), seed=7)


def test_fixed_normalization_and_coordinates():
    cfg = small_config().radar
    image = np.ones((32, 32))
    channels = preprocess_image(image, cfg)
    assert channels.shape == (3, 32, 32)
    assert channels.dtype == np.float32
    np.testing.assert_allclose(channels[0], 0.5)
    np.testing.assert_allclose(channels[1] ** 2 + channels[2] ** 2,
                               (cfg.ranges[:, None] / cfg.ranges[-1]) ** 2 * np.ones((1, 32)), atol=2e-7)
    with pytest.raises(ValueError):
        preprocess_image(-image, cfg)


def test_generation_splits_repeatability_and_loading(tmp_path):
    cfg = small_config()
    a = generate_dataset(tmp_path / "a", cfg, {"train": 2, "val": 1, "test": 1}, save_tensors=True)
    b = generate_dataset(tmp_path / "b", cfg, {"train": 2, "val": 1, "test": 1})
    ds = RadarDataset(a)
    assert len(ds) == 2
    item = ds[0]
    assert item["image"].shape == (3, 32, 32)
    assert item["boxes"].shape[1] == 4
    batch = collate_fn([ds[0], ds[1]])
    assert batch["image"].shape == (2, 3, 32, 32)
    with np.load(a / "train/train_000000.npz") as x, np.load(b / "train/train_000000.npz") as y:
        np.testing.assert_array_equal(x["image"], y["image"])
        assert x["tensor"].shape == cfg.radar.shape
    ids = [r["id"] for rows in ds.manifest["splits"].values() for r in rows]
    assert len(ids) == len(set(ids))
    with pytest.raises(FileExistsError):
        generate_dataset(a, cfg, {"train": 1, "val": 0, "test": 0})


def test_external_conversion_and_scene_leak_guard(tmp_path):
    cfg = small_config()
    np.save(tmp_path / "one.npy", np.zeros((32, 32)))
    rows = [{"id": "one", "split": "train", "scene_id": "scene1", "image_path": "one.npy",
             "representation": "db", "boxes": [[0.1, 0.2, 0.3, 0.4]]}]
    source = tmp_path / "manifest.jsonl"
    source.write_text(json.dumps(rows[0]), encoding="utf-8")
    out = import_manifest(source, tmp_path / "data", cfg)
    with np.load(out / "train/one.npz") as data:
        np.testing.assert_allclose(data["image"], 1)
    rows.append(dict(rows[0], id="two", split="test"))
    source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="across splits"):
        import_manifest(source, tmp_path / "bad", cfg)
    assert not (tmp_path / "bad").exists()


def test_configuration_rejects_unknown_and_invalid(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"radar":{"range_bns":16}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(p)
    with pytest.raises(ValueError):
        RadarConfig(noise_std=-1)
    with pytest.raises(ValueError):
        RadarConfig(psf_shape=(4, 5, 7))
    with pytest.raises(ValueError):
        SceneConfig(points_per_vehicle=8.5)
    with pytest.raises(ValueError):
        SceneConfig(lateral_limit=float("nan"))


def test_integer_amplitude_is_squared_without_overflow(tmp_path):
    cfg = small_config()
    np.save(tmp_path / "uint.npy", np.full((32, 32), 200, dtype=np.uint8))
    row = {"id": "one", "split": "train", "scene_id": "s1", "image_path": "uint.npy",
           "representation": "amplitude", "boxes": []}
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    import_manifest(path, tmp_path / "data", cfg)
    with np.load(tmp_path / "data/train/one.npz") as data:
        np.testing.assert_array_equal(data["image"], 40000)
    meta = json.loads((tmp_path / "data/train/one.json").read_text())
    assert meta["provenance"]["source_dtype"] == "uint8"
    assert len(meta["provenance"]["linear_image_sha256"]) == 64


def test_import_rejects_unrepresentable_float32_and_duplicate_content(tmp_path):
    cfg = small_config()
    np.save(tmp_path / "huge.npy", np.full((32, 32), 1e40))
    row = {"id": "one", "split": "train", "scene_id": "s1", "image_path": "huge.npy",
           "representation": "power", "boxes": []}
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="float32"):
        import_manifest(path, tmp_path / "bad", cfg)
    assert not (tmp_path / "bad").exists()
    np.save(tmp_path / "huge.npy", np.ones((32, 32)))
    rows = [row, dict(row, id="two", scene_id="s2", split="test")]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Identical"):
        import_manifest(path, tmp_path / "leak", cfg)


def test_measured_psf_is_portable_and_grid_checked(tmp_path):
    from radsimreal.radar import save_psf, load_psf
    cfg = small_config()
    radar = cfg.radar
    metadata = {"frames_averaged": 20, "grid": {k: getattr(radar, k) for k in
                ("range_step", "velocity_step", "angular_step", "azimuth_coordinate", "representation")}}
    save_psf(tmp_path / "calibration.npz", np.ones((1, 1, 1)), metadata)
    cfg.radar.psf_path = str(tmp_path / "calibration.npz")
    out = generate_dataset(tmp_path / "calibrated", cfg, {"train": 1, "val": 0, "test": 0})
    saved_cfg = load_config(out / "config.json")
    _, saved = load_psf(saved_cfg.radar.psf_path, return_metadata=True,
                        config=saved_cfg.radar, strict_metadata=True)
    assert saved["frames_averaged"] == 20
    assert saved["grid"] == metadata["grid"]
    saved_cfg.radar.range_step *= 2
    with pytest.raises(ValueError, match="mismatch"):
        load_psf(out / "calibration.npz", config=saved_cfg.radar, strict_metadata=True)
