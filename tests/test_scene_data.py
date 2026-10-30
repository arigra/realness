import json

import torch

from src.scene_data import MAX_VEHICLES, SceneSequenceDataset, pack_labels


def test_labels_keep_vehicle_identity_when_one_leaves_the_view():
    lab = lambda i, r: {"id": i, "range_bin": r, "doppler_bin": 1.0}
    frames = [[lab(0, 10.0), lab(1, 50.0)],   # both visible
              [lab(1, 51.0)]]                 # vehicle 0 has left
    traj, present = pack_labels(frames, 2)
    assert present[0].tolist() == [True, False]
    assert present[1].tolist() == [True, True]
    assert traj[1, :, 0].tolist() == [50.0, 51.0]   # vehicle 1 stayed in slot 1


