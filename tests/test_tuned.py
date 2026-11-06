"""The tuned simulators (src/tuned.py): with no switch on they are the base
simulators, and each switch does what it says."""
import pytest
import torch

from src.carrada_sim import CarradaSimulator
from src.raddet_sim import RaddetSimulator
from src.tuned import TunedCarrada, TunedRaddet, TunedRadial


def g(seed=3):
    return torch.Generator().manual_seed(seed)


@pytest.mark.parametrize("tuned, base", [(TunedCarrada, CarradaSimulator), (TunedRaddet, RaddetSimulator)])
def test_no_switch_is_the_base_simulator(tuned, base):
    a, b = tuned(generator=g()).gen_sequence(), base(generator=g()).gen_sequence()
    assert torch.equal(a["x"], b["x"]) and a["labels"] == b["labels"]


def test_a_scene_puts_the_object_at_the_real_labels_range_and_doppler():
    scene = [[{"range_bin": 100.0, "doppler_bin": 40.0, "cls": 1}]]
    lab = TunedCarrada(generator=g(), scenes=scene).gen_sequence()["labels"][0][0]
    assert lab["doppler_bin"] == pytest.approx(40.0, abs=0.05)     # body velocity, frame 0
    assert lab["range_bin"] == pytest.approx(100.0, abs=2.0)        # power-weighted over the limbs


def test_raddet_scene_keeps_the_class_and_its_size():
    scene = [[{"range_bin": 120.0, "doppler_bin": 20.0, "name": "bus"}]]
    sim = TunedRaddet(generator=g(), scenes=scene)
    o = sim._objects()[0]
    assert o["name"] == "bus" and o["rcs_dbsm"] == 20.0
    assert float(o["car"]["oy"].max() - o["car"]["oy"].min()) > 5.0 or float(o["car"]["ox"].max() - o["car"]["ox"].min()) > 5.0


def test_box_label_is_the_centre_of_the_returns():
    sim = TunedCarrada(generator=g(), scenes=[[{"range_bin": 80.0, "doppler_bin": 44.0, "cls": 3}]], label="box")
    o = sim._objects()[0]
    px, py, v_app, rcs = sim._returns(o, 0.0)
    r, v = sim._label(o, 0.0, px, py, v_app, rcs)
    rng = torch.sqrt(px ** 2 + py ** 2)
    assert float(rng.min()) <= r <= float(rng.max())
    assert float(v_app.min()) <= v <= float(v_app.max())


def test_range_gain_is_added_per_row():
    gain = torch.linspace(-5, 5, 256)
    a = TunedCarrada(generator=g(), range_gain_db=gain).gen_sequence()["x"]
    b = CarradaSimulator(generator=g()).gen_sequence()["x"]
    assert torch.allclose(a - b, gain[None, :, None].expand_as(a), atol=1e-4)


def test_radial_scene_vehicle_has_the_labels_relative_doppler():
    scene = [[{"x": 1.0, "y": 40.0, "doppler_bin": 250.0}]]       # -0.6 m/s relative
    out = TunedRadial(generator=g(), scenes=scene, seq_len=1).gen_sequence()
    lab = out["labels"][0][0]
    assert lab["doppler_bin"] == pytest.approx(250.0, abs=0.3)
    assert lab["range_bin"] * 0.2 == pytest.approx(40.0, abs=0.5)  # nearest face at the label


def test_target_snr_sets_the_strongest_return():
    sim = TunedCarrada(generator=g(), scenes=[[{"range_bin": 150.0, "doppler_bin": 40.0, "cls": 3}]],
                       target_snr_db=(12.0, 0.0))
    o = sim._objects()[0]
    px, py, _, rcs = sim._returns(o, 0.0)
    assert float(sim._snr_db(px, py, rcs).max()) == pytest.approx(12.0, abs=1e-3)


def test_annotation_threshold_leaves_weak_objects_unlabelled():
    scene = [[{"range_bin": 150.0, "doppler_bin": 40.0, "cls": 1}]]
    kw = {"scenes": scene, "target_snr_db": (5.0, 0.0)}
    assert TunedCarrada(generator=g(), **kw).gen_sequence()["labels"][0]
    assert not TunedCarrada(generator=g(), label_min_snr_db=10.0, **kw).gen_sequence()["labels"][0]


# ---- the damage mixins on the tuned simulators (src/damage.py)
from src.damage import RadialDamage, TrackDamage, range_tail  # noqa: E402


class _DC(TrackDamage, TunedCarrada):
    pass


def test_no_damage_is_the_tuned_simulator():
    kw = {"scenes": [[{"range_bin": 100.0, "doppler_bin": 40.0, "cls": 3}]], "label": "box", "processing": True}
    a = TunedCarrada(generator=g(), **kw).gen_sequence()
    b = _DC(generator=g(), damage_generator=g(1), **kw).gen_sequence()
    assert torch.equal(a["x"], b["x"]) and a["labels"] == b["labels"]


@pytest.mark.parametrize("damage", [{"blur": 3.0}, {"tail": 4.0}, {"extra_frac": 1.0}, {"unrender_frac": 1.0},
                                    {"sidelobes_db": 10.0}])
def test_map_defects_keep_the_labels(damage):
    kw = {"scenes": [[{"range_bin": 100.0, "doppler_bin": 40.0, "cls": 3}]], "label": "box", "processing": True}
    a = TunedCarrada(generator=g(), **kw).gen_sequence()
    b = _DC(generator=g(), damage_generator=g(1), **kw, **damage).gen_sequence()
    assert a["labels"] == b["labels"] and not torch.equal(a["x"], b["x"])


def test_range_tail_keeps_power_and_trails_away_from_the_radar():
    p = torch.zeros(64, 8)
    p[20, 3] = 1.0
    q = range_tail(p, 3.0)
    assert float(q.sum()) == pytest.approx(1.0, abs=1e-5)
    assert float(q[:20].sum()) == 0.0 and float(q[21:].sum()) > 0.3


def test_confusers_scale_the_discrete_static_structures():
    """More poles change the scene's draws, so only the scenario is checked."""
    assert _DC(generator=g(), damage_generator=g(1), confusers=4.0).sc.poles_mean == 4 * TunedCarrada().sc.poles_mean
