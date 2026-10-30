import numpy as np
import torch

from src.carrada_sim import CarradaSimulator
from src.damage import DamagedCarradaSimulator, clutter_scenario, drop_labels, shift_labels


def _seq(sim_cls=CarradaSimulator, seed=10_000_003, **kw):
    return sim_cls(generator=torch.Generator().manual_seed(seed), **kw).gen_sequence()


def test_undamaged_matches_simulator():
    a, b = _seq(), _seq(DamagedCarradaSimulator, blink_generator=torch.Generator().manual_seed(0))
    assert torch.equal(a["x"], b["x"]) and a["labels"] == b["labels"]


def test_target_offset_changes_only_targets():
    seed = next(s for s in range(10_000_000, 10_000_100) if _seq(seed=s)["classes"])
    a = _seq(seed=seed)
    b = _seq(DamagedCarradaSimulator, seed=seed, target_offset_db=-10.0)
    for fa, fb in zip(a["labels"], b["labels"]):              # same labels up to float rounding
        assert [(l["id"], round(l["range_bin"], 3), round(l["doppler_bin"], 3)) for l in fa] == \
               [(l["id"], round(l["range_bin"], 3), round(l["doppler_bin"], 3)) for l in fb]
    lab = a["labels"][0][0]
    r, d = round(lab["range_bin"]), round(lab["doppler_bin"]) % 64
    assert float(a["x"][0, r, d] - b["x"][0, r, d]) > 3.0       # target cell weaker
    assert float((a["x"][0, :, 32] - b["x"][0, :, 32]).abs().median()) < 0.5   # static ridge intact


def test_full_blink_removes_all_objects():
    seed = next(s for s in range(10_000_000, 10_000_100) if _seq(seed=s)["classes"])
    b = _seq(DamagedCarradaSimulator, seed=seed, blink_frac=1.0,
             blink_generator=torch.Generator().manual_seed(0))
    assert all(len(f) == 0 for f in b["labels"])


def test_clutter_scenario_offsets_ground_and_poles():
    sc = clutter_scenario(6.0)
    assert sc.ground_rcs_dbsm == (-19.0, 6.0) and sc.pole_rcs_dbsm == (11.0, 4.0)


def test_label_defects():
    traj = torch.tensor([[[255.0, 63.0], [10.0, 0.0]]])
    assert shift_labels(traj, 0, 3, 256, wrap=False)[0, :, 0].tolist() == [255.0, 13.0]
    assert shift_labels(traj, 1, 3, 64, wrap=True)[0, :, 1].tolist() == [2.0, 3.0]
    present = torch.ones(100, 8, dtype=torch.bool)
    kept = drop_labels(present, 0.25, torch.Generator().manual_seed(0)).float().mean()
    assert abs(float(kept) - 0.75) < 0.03
    assert np.array_equal(drop_labels(present, 0.0, torch.Generator()).numpy(), present.numpy())


def _busy_seed():
    return next(s for s in range(10_000_000, 10_000_100) if _seq(seed=s)["classes"])


def test_unrender_keeps_labels_and_removes_target():
    seed = _busy_seed()
    a = _seq(seed=seed)
    b = _seq(DamagedCarradaSimulator, seed=seed, unrender_frac=1.0,
             blink_generator=torch.Generator().manual_seed(0))
    assert [[l["id"] for l in f] for f in a["labels"]] == [[l["id"] for l in f] for f in b["labels"]]
    lab = a["labels"][0][0]
    r, d = round(lab["range_bin"]), round(lab["doppler_bin"]) % 64
    assert float(a["x"][0, r, d] - b["x"][0, r, d]) > 6.0


def test_extra_object_adds_power_without_label():
    seed = next(s for s in range(10_000_000, 10_000_100) if not _seq(seed=s)["classes"])
    a = _seq(seed=seed)
    b = _seq(DamagedCarradaSimulator, seed=seed, extra_frac=1.0,
             blink_generator=torch.Generator().manual_seed(3))
    assert all(len(f) == 0 for f in b["labels"])
    diff = b["x"] - a["x"]
    assert float(diff.min()) >= -1e-3 and float(diff.max()) > 6.0   # only added power


def test_noisy_spec_and_smoothing():
    from src.damage import noisy_spec, smooth_background
    assert noisy_spec(6.0).noise_figure_db == 20.5
    x = torch.randn(2, 16, 8)
    traj = torch.tensor([[[8.0, 4.0], [8.0, 4.0]]])
    present = torch.tensor([[True, False]])
    y = smooth_background(x, traj, present, 0.5)
    assert torch.equal(y[0, 8, 4], x[0, 8, 4]) and not torch.equal(y[1, 8, 4], x[1, 8, 4])
    assert float(y[1].std()) < float(x[1].std())


def _scene(seed=10_000_000, **kw):
    from src.damage import DamagedSceneSimulator
    from src.radial_sim import SceneSimulator
    cls = DamagedSceneSimulator if kw else SceneSimulator
    if kw:
        kw.setdefault("blink_generator", torch.Generator().manual_seed(1))
    return cls(seq_len=2, generator=torch.Generator().manual_seed(seed), **kw).gen_sequence()


def test_radial_undamaged_and_damaged_scenes():
    a = _scene()
    assert torch.equal(a["x"], _scene(target_offset_db=0.0)["x"])
    gone = _scene(unrender_frac=1.0)
    assert [[l["id"] for l in f] for f in gone["labels"]] == [[l["id"] for l in f] for f in a["labels"]]
    lab = a["labels"][0][0]
    r, d = round(lab["range_bin"]), round(lab["doppler_bin"]) % a["x"].shape[-1]
    assert float(a["x"][0, r - 2:r + 3, d] .max() - gone["x"][0, r - 2:r + 3, d].max()) > 6.0
    extra = _scene(extra_frac=1.0)
    assert extra["labels"] == a["labels"] and float((extra["x"] - a["x"]).min()) >= -1e-3
    assert float((extra["x"] - a["x"]).max()) > 6.0
    clut = _scene(clutter_offset_db=6.0)
    assert float((clut["x"] - a["x"]).quantile(0.9)) > 3.0      # clutter cells rise; noise cells do not
