"""Executed functional checks; these do not validate published detector AP."""

import numpy as np
import pytest
import torch

from radsimreal.models import CoordinateUNet, build_targets, detection_loss, detect, decode_detections
from radsimreal.evaluation import box_iou, nms, average_precision, evaluate_predictions


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_forward_odd_dimensions_and_training_step():
    torch.manual_seed(3)
    model = CoordinateUNet(base_channels=4)
    images = torch.randn(2, 3, 33, 35)
    boxes = [torch.tensor([[0.1, 0.2, 0.4, 0.5]]), torch.empty(0, 4)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    previous = model.heatmap_head.weight.detach().clone()
    outputs = model(images)
    assert outputs["heatmap"].shape == (2, 1, 33, 35)
    assert outputs["offset"].shape == (2, 2, 33, 35)
    assert torch.all((outputs["size"] > 0) & (outputs["size"] < 1))
    losses = detection_loss(outputs, boxes)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(previous, model.heatmap_head.weight)
    assert model.training
    predictions = detect(model, images, score_threshold=0.0, top_k=5)
    assert model.training
    assert len(predictions) == 2
    assert all(len(p["boxes"]) <= 5 for p in predictions)
    assert all(torch.isfinite(p["boxes"]).all() for p in predictions)


def test_empty_targets_have_finite_loss_and_zero_box_loss():
    model = CoordinateUNet(base_channels=2)
    output = model(torch.zeros(1, 3, 16, 16))
    losses = detection_loss(output, [torch.empty(0, 4)])
    assert losses["box"].item() == 0
    assert torch.isfinite(losses["total"])
    losses["total"].backward()


def test_center_collision_is_reported_and_smaller_box_retained():
    boxes = [torch.tensor([[0.1, 0.1, 0.9, 0.9], [0.4, 0.4, 0.6, 0.6]])]
    target = build_targets(boxes, 16, 16)
    assert target["collisions"] == 1
    assert target["mask"].sum() == 1
    assert torch.allclose(target["size"][0, :, 8, 8], torch.tensor([0.2, 0.2]))
    assert target["heatmap"][0, 0, 8, 8] == 1
    with pytest.raises(ValueError):
        build_targets([torch.tensor([[0.5, 0.2, 0.1, 0.4]])], 16, 16)


def test_decode_exact_center_and_size():
    outputs = {
        "heatmap": torch.full((1, 1, 20, 20), -20.0),
        "offset": torch.zeros(1, 2, 20, 20),
        "size": torch.full((1, 2, 20, 20), 0.2),
    }
    outputs["heatmap"][0, 0, 8, 10] = 20
    outputs["offset"][0, :, 8, 10] = torch.tensor([0.5, 0.25])
    prediction = decode_detections(outputs)[0]
    assert prediction["boxes"].shape == (1, 4)
    assert torch.allclose(prediction["boxes"][0], torch.tensor([0.425, 0.3125, 0.625, 0.5125]))


def test_iou_and_nms_have_known_answers():
    boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.], [0., 0., 0.5, 0.5]])
    assert box_iou(boxes[:1], boxes).tolist() == [[1.0, 1.0, 0.25]]
    assert nms(boxes, torch.tensor([0.9, 0.8, 0.7]), 0.3).tolist() == [0, 2]
    assert box_iou(torch.empty(0, 4), boxes).shape == (0, 3)
    assert nms(torch.empty(0, 4), torch.empty(0)).numel() == 0


def test_ap_score_ranking_and_duplicate_matching():
    gt = [np.array([[0, 0, 0.2, 0.2], [0.5, 0.5, 0.7, 0.7]])]
    prediction = [{"boxes": np.array([[0, 0, .2, .2], [0, 0, .2, .2], [.5, .5, .7, .7]]),
                   "scores": np.array([.9, .8, .7])}]
    result = average_precision(prediction, gt, .5)
    assert result["ap"] == pytest.approx(5 / 6)
    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["recall"] == [0.5, 0.5, 1.0]
    # A high-scoring false detection must lower AP below perfect.
    bad = [{"boxes": np.array([[.8, .8, .9, .9], [0, 0, .2, .2]]), "scores": [.99, .5]}]
    assert average_precision(bad, [gt[0][:1]], .5)["ap"] == pytest.approx(.5)


def test_ap_handles_empty_truth_and_predictions():
    empty = [{"boxes": [], "scores": []}]
    assert average_precision(empty, [np.empty((0, 4))])["ap"] is None
    assert average_precision(empty, [np.array([[0, 0, 1, 1]])])["ap"] == 0
    summary = evaluate_predictions(empty, [np.array([[0, 0, 1, 1]])])
    assert summary["AP@0.1"] == 0
    assert summary["AP@0.3"] == 0
    assert summary["AP@0.5"] == 0


def test_checkpoint_restricted_load_reproduces_outputs(tmp_path):
    from radsimreal.training import load_checkpoint
    torch.manual_seed(2)
    model = CoordinateUNet(base_channels=2).eval()
    image = torch.randn(1, 3, 16, 16)
    path = tmp_path / "checkpoint.pt"
    torch.save({"format_version": 1, "model_name": "coordinate_unet_reconstruction",
                "model_config": {"base_channels": 2, "in_channels": 3},
                "model_state": model.state_dict()}, path)
    loaded, _ = load_checkpoint(path)
    with torch.no_grad():
        assert torch.equal(model(image)["heatmap"], loaded(image)["heatmap"])


def test_training_validation_evaluation_and_optimizer_resume(tmp_path):
    from radsimreal.config import SimulationConfig, RadarConfig, SceneConfig
    from radsimreal.datasets import generate_dataset
    from radsimreal.training import train, load_checkpoint
    from radsimreal.evaluation.runner import evaluate

    config = SimulationConfig(
        RadarConfig(range_bins=32, azimuth_bins=32, doppler_bins=8, range_step=1.5, velocity_step=4),
        SceneConfig(max_vehicles=2, points_per_vehicle=12, clutter_points=2),
    )
    generate_dataset(tmp_path / "data", config, counts={"train": 2, "val": 1, "test": 1})
    checkpoint, history = train(tmp_path / "data", tmp_path / "training", epochs=1,
                                batch_size=2, max_steps=1, base_channels=2)
    _, payload = load_checkpoint(checkpoint)
    assert payload["global_step"] == 1
    assert payload["normalization"]["db_min"] == -30
    assert np.isfinite(history[0]["val_loss"])
    result = evaluate(checkpoint, tmp_path / "data", top_k=3)
    assert result["num_images"] == 1
    assert set(result["image_ids"]) == {"test_000000"}
    assert result["metric_domain"] == "normalized_RA_bin_area_not_Cartesian_BEV"
    assert all(result[key] is None or 0 <= result[key] <= 1 for key in ("AP@0.1", "AP@0.3", "AP@0.5"))
    resumed, history = train(tmp_path / "data", tmp_path / "resumed", epochs=1,
                              batch_size=2, max_steps=2, resume=checkpoint)
    _, payload = load_checkpoint(resumed.parent / "last.pt")
    assert payload["global_step"] == 2
    assert len(history) == 2


def test_resume_retains_historical_best_and_reports_restored_optimizer(tmp_path, monkeypatch):
    from radsimreal.config import SimulationConfig, RadarConfig, SceneConfig
    from radsimreal.datasets import generate_dataset
    from radsimreal.training import train, load_checkpoint
    import radsimreal.training.engine as engine

    config = SimulationConfig(
        RadarConfig(range_bins=16, azimuth_bins=16, doppler_bins=8, range_step=3, velocity_step=4),
        SceneConfig(max_vehicles=1, points_per_vehicle=8, clutter_points=0),
    )
    generate_dataset(tmp_path / "data", config, counts={"train": 1, "val": 1, "test": 0})
    monkeypatch.setattr(engine, "_validation_loss", lambda *args: .1)
    best, _ = train(tmp_path / "data", tmp_path / "original", epochs=1, max_steps=1,
                    base_channels=2, learning_rate=.007, weight_decay=.012)
    original_bytes = best.read_bytes()
    monkeypatch.setattr(engine, "_validation_loss", lambda *args: .9)
    retained, _ = train(tmp_path / "data", best.parent, epochs=1, max_steps=2,
                        resume=best.parent / "last.pt", learning_rate=.2, weight_decay=.5)
    assert retained.read_bytes() == original_bytes
    _, last = load_checkpoint(best.parent / "last.pt")
    assert last["best_val_loss"] == .1
    assert last["history"][-1]["val_loss"] == .9
    assert last["training_recipe"]["learning_rate"] == .007
    assert last["training_recipe"]["weight_decay"] == .012
    assert last["training_recipe"]["effective_parameter_groups"][0]["learning_rate"] == .007
    copied, _ = train(tmp_path / "data", tmp_path / "new_output", epochs=1, max_steps=3,
                      resume=best.parent / "last.pt")
    assert copied.read_bytes() == original_bytes
    _, last = load_checkpoint(copied.parent / "last.pt")
    assert last["best_selection_scope"] == "continued_run_with_historical_best_weights"
    assert last["global_step"] == 3


def test_resume_without_earlier_best_weights_declares_segment_scope(tmp_path, monkeypatch):
    from radsimreal.config import SimulationConfig, RadarConfig, SceneConfig
    from radsimreal.datasets import generate_dataset
    from radsimreal.training import train, load_checkpoint
    import radsimreal.training.engine as engine

    config = SimulationConfig(
        RadarConfig(range_bins=16, azimuth_bins=16, doppler_bins=8, range_step=3, velocity_step=4),
        SceneConfig(max_vehicles=1, points_per_vehicle=8, clutter_points=0),
    )
    generate_dataset(tmp_path / "data", config, counts={"train": 1, "val": 1, "test": 0})
    values = iter((.1, .9))
    monkeypatch.setattr(engine, "_validation_loss", lambda *args: next(values))
    best, _ = train(tmp_path / "data", tmp_path / "original", epochs=2, base_channels=2)
    # Export only last.pt, as when historical best weights have been lost.
    standalone = tmp_path / "exported_last.pt"
    standalone.write_bytes((best.parent / "last.pt").read_bytes())
    monkeypatch.setattr(engine, "_validation_loss", lambda *args: .8)
    resumed, _ = train(tmp_path / "data", tmp_path / "segment", epochs=1, resume=standalone)
    _, payload = load_checkpoint(resumed)
    assert payload["prior_best_val_loss"] == .1
    assert payload["best_val_loss"] == .8
    assert payload["best_selection_scope"] == "resumed_segment_only_historical_best_weights_unavailable"
