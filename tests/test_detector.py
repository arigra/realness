"""The frame detector, its training batches, scoring and label packing."""
import torch

from src.data import pack_labels
from src.detector import FrameDetector, average_precision, detect, heatmap_targets, recall
from src.train import frame_batch


def test_targets_mark_present_vehicles_only():
    traj = torch.zeros(1, 2, 2, 2)
    traj[0, 0] = torch.tensor([[100.0, 40.0], [102.0, 41.0]])
    traj[0, 1] = torch.tensor([[300.0, 80.0], [300.0, 80.0]])
    present = torch.tensor([[[True, True], [True, False]]])
    h = heatmap_targets(traj, present, n_range=512, n_doppler=256)
    assert h.shape == (1, 2, 512, 256)
    assert float(h[0, 0, 100, 40]) == 1.0
    assert float(h[0, 1, 102, 41]) == 1.0
    assert float(h[0, 0, 300, 80]) == 1.0
    assert float(h[0, 1, 300, 80]) == 0.0        # absent in frame 1


def test_targets_wrap_around_the_doppler_axis():
    traj = torch.zeros(1, 1, 1, 2)
    traj[0, 0, 0] = torch.tensor([100.0, 255.0])
    h = heatmap_targets(traj, torch.ones(1, 1, 1, dtype=torch.bool),
                              n_range=512, n_doppler=256)
    assert float(h[0, 0, 100, 0]) > 0.5          # bin 0 is one step from 255


def test_detector_returns_one_heatmap_per_frame():
    torch.manual_seed(0)
    model = FrameDetector(width=8)
    out = model(torch.randn(2, 4, 512, 256))
    assert out.shape == (2, 4, 512, 256) and torch.isfinite(out).all()


def test_detector_is_per_frame_so_frames_do_not_leak():
    """Each frame is detected on its own; changing frame 1 must not change
    frame 0's output, or a detector could cheat by reading other frames."""
    torch.manual_seed(0)
    model = FrameDetector(width=8).eval()
    x = torch.randn(1, 2, 512, 256)
    y = x.clone()
    y[:, 1] += 5.0
    with torch.no_grad():
        a, b = model(x), model(y)
    assert torch.allclose(a[:, 0], b[:, 0], atol=1e-5)


def test_average_precision_wraps_in_doppler():
    """Many RADIal vehicles sit at the Doppler axis edge. A detection at bin 255
    is two bins from a target at bin 1, not 254 -- without wrapping every such
    vehicle counts as a miss and every arm is scored unfairly."""
    dets = [[(0.9, 100.0, 255.0)]]
    tgts = [torch.tensor([[100.0, 1.0]])]
    assert average_precision(dets, tgts, 256, radius=2.0) == 1.0
    assert recall(dets, tgts, 256, radius=2.0) == 1.0


def test_wrapped_average_precision_still_rejects_far_detections():
    dets = [[(0.9, 100.0, 128.0)]]
    tgts = [torch.tensor([[100.0, 1.0]])]
    assert average_precision(dets, tgts, 256, radius=2.0) == 0.0


def test_training_batches_are_frames_not_whole_sequences():
    """The detector is per-frame, and a batch of whole 512x256 sequences does
    not fit in GPU memory (it ran out overnight). Batches sample one frame at a
    time, and the target must be that frame's vehicles."""
    pool = [{"x": torch.zeros(8, 512, 256, dtype=torch.float16),
             "traj": torch.zeros(8, 8, 2), "present": torch.zeros(8, 8, dtype=torch.bool),
             "stats": {"mean": 0.0, "std": 1.0}}
            for _ in range(3)]
    pool[0]["traj"][0, 5] = torch.tensor([100.0, 40.0])
    pool[0]["present"][0, 5] = True
    x, heat = frame_batch(pool, 4, torch.device("cpu"), torch.Generator().manual_seed(0))
    assert x.shape == (4, 1, 512, 256)
    assert heat.shape == (4, 1, 512, 256)


def test_each_source_is_normalised_by_its_own_statistics():
    """Real maps sit near 97 dB and simulated ones near 12 dB. Normalising the
    simulated frames by the real statistics put them ~15 std off the real data,
    which made the real+sim arm meaningless."""
    def item(level, mean, std):
        return {"x": torch.full((8, 512, 256), level, dtype=torch.float16),
                "traj": torch.zeros(8, 8, 2), "present": torch.zeros(8, 8, dtype=torch.bool),
                "stats": {"mean": mean, "std": std}}
    pool = [item(97.0, 97.0, 5.6), item(12.5, 12.5, 5.5)]
    x, _ = frame_batch(pool, 16, torch.device("cpu"), torch.Generator().manual_seed(0))
    assert float(x.abs().max()) < 0.05


def test_every_vehicle_gets_exactly_one_positive_cell_even_off_grid():
    """The focal loss counts a cell as a vehicle only if its target is exactly
    1.0. RADIal ranges are fractional (e.g. 157.69 bins), so an unrounded
    Gaussian never reaches 1.0 -- 596 vehicles produced 1 positive cell, the
    detector learned to predict nothing, and every arm scored AP 0."""
    traj = torch.zeros(1, 2, 1, 2)
    traj[0, 0, 0] = torch.tensor([157.69, 40.0])
    traj[0, 1, 0] = torch.tensor([300.31, 254.6])
    h = heatmap_targets(traj, torch.ones(1, 2, 1, dtype=torch.bool),
                              n_range=512, n_doppler=256)
    assert int(h.eq(1).sum()) == 2
    assert float(h[0, 0, 158, 40]) == 1.0
    assert float(h[0, 0, 300, 255]) == 1.0


def test_a_peak_across_the_doppler_seam_is_detected_once():
    """RADIal's Doppler axis wraps. With plain max-pool padding, a vehicle
    straddling bins 255 and 0 gave two local maxima, one of them a false alarm."""
    prob = torch.zeros(1, 32, 256)
    prob[0, 10, 255], prob[0, 10, 0], prob[0, 10, 1] = 0.8, 0.9, 0.3
    assert len(detect(prob)[0]) == 1


def test_batches_take_their_grid_from_the_data():
    """CARRADA maps are 256 x 64; targets built on RADIal's 512 x 256 grid
    would not match the detector output."""
    pool = [{"x": torch.zeros(8, 256, 64, dtype=torch.float16),
             "traj": torch.zeros(4, 8, 2), "present": torch.zeros(4, 8, dtype=torch.bool),
             "stats": {"mean": 0.0, "std": 1.0}}]
    pool[0]["traj"][0, :] = torch.tensor([100.0, 40.0])
    pool[0]["present"][0, :] = True
    x, heat = frame_batch(pool, 4, torch.device("cpu"), torch.Generator().manual_seed(0))
    assert x.shape == heat.shape == (4, 1, 256, 64)
    assert int(heat.eq(1).sum()) == 4


def test_labels_keep_vehicle_identity_when_one_leaves_the_view():
    lab = lambda i, r: {"id": i, "range_bin": r, "doppler_bin": 1.0}
    frames = [[lab(0, 10.0), lab(1, 50.0)],   # both visible
              [lab(1, 51.0)]]                 # vehicle 0 has left
    traj, present = pack_labels(frames, 2)
    assert present[0].tolist() == [True, False]
    assert present[1].tolist() == [True, True]
    assert traj[1, :, 0].tolist() == [50.0, 51.0]   # vehicle 1 stayed in slot 1


def test_carrada_ids_fill_slots_in_sorted_order():
    lab = lambda i, r: {"id": i, "range_bin": r, "doppler_bin": 1.0}
    traj, present = pack_labels([[lab(17, 10.0), lab(4, 50.0)]], 1, slot_by_rank=True)
    assert traj[:2, 0, 0].tolist() == [50.0, 10.0] and present[:2, 0].tolist() == [True, True]


def test_recall_ignores_false_alarms_and_counts_each_object_once():
    dets = [[(0.9, 10.0, 5.0), (0.8, 10.0, 6.0), (0.7, 200.0, 50.0)]]
    tgts = [torch.tensor([[10.0, 5.0], [300.0, 9.0]])]
    assert recall(dets, tgts, 64, radius=2.0) == 0.5
