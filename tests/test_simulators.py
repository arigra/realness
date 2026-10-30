"""The simulators are physics from published sources (nothing fitted to the
recordings), so these tests pin each specification and mechanism rather than
any match to real pixels."""
import math

import pytest
import torch

from src.carrada_sim import CarradaSimulator, TrackScenario
from src.radar import RADIAL_SPEC, window
from src.radial_sim import DDMA_OFFSETS, SceneSimulator


# ---- the RADIal radar specification

def test_published_radial_specification():
    s = RADIAL_SPEC
    assert (s.n_range, s.n_doppler, s.n_rx, s.n_tx) == (512, 256, 16, 12)
    assert s.range_res_m == 0.2 and s.velocity_res_mps == 0.1
    assert s.frame_rate_hz == 5.0


def test_derived_waveform_quantities_match_the_grid():
    s = RADIAL_SPEC
    assert s.bandwidth_hz == pytest.approx(750e6)
    assert s.max_range_m == pytest.approx(102.2)
    assert s.frame_interval_s == pytest.approx(0.2)
    assert s.chirp_interval_s == pytest.approx(7.6e-5, rel=1e-2)


def test_looks_come_from_the_receive_array_not_a_fit():
    """radar_FFT carries one RD spectrum per receive antenna, so summing it in
    power integrates exactly n_rx looks."""
    assert RADIAL_SPEC.n_looks == RADIAL_SPEC.n_rx == 16


def test_hamming_window_matches_radials_processing_code():
    w = window(256, "hamming")
    assert float(w[0]) == pytest.approx(0.08, abs=1e-6)
    assert float(w.max()) == pytest.approx(1.0, abs=1e-3)
    assert torch.equal(window(64, "none"), torch.ones(64))
    with pytest.raises(ValueError):
        window(16, "blackman")


# ---- CARRADA simulator

def sim(seed=0, **kw):
    return CarradaSimulator(scenario=TrackScenario(**kw), generator=torch.Generator().manual_seed(seed))


def test_sequence_shape_and_reproducible_from_its_seed():
    a, b = sim(3).gen_sequence(), sim(3).gen_sequence()
    assert a["x"].shape == (8, 256, 64) and torch.isfinite(a["x"]).all()
    assert torch.equal(a["x"], b["x"])


def test_static_scene_sits_at_zero_doppler_bin_32():
    """The radar does not move, so ground and poles have zero radial velocity:
    the zero-Doppler ridge CARRADA shows at bin 32."""
    x = sim(1, p_empty=1.0).gen_sequence()["x"]
    prof = x.median(dim=1).values.median(dim=0).values
    assert int(prof.argmax()) == 32


def test_an_approaching_car_sits_above_bin_32_on_its_label():
    s = sim(0)
    o = s._object(3, 0)
    o.update(x0=0.0, y0=20.0, vx=0.0, vy=-5.0, speed=5.0)
    o["car"] = s._car_outline(math.pi)
    px, py, v_app, rcs = s._returns(o, 0.0)
    assert bool((v_app > 4.9).all())                 # moving towards the radar at 5 m/s
    p = s._render(px, py, v_app, rcs)
    r, d = divmod(int(p.argmax()), 64)
    assert d == 32 + round(5.0 / 0.42) and abs(r - (20.0 - 2.25) / 0.2) <= 2


def test_a_walker_spreads_in_doppler_and_a_car_does_not():
    """Limbs swing to ~3.3x the body's speed (micro-Doppler); a car's body moves as one."""
    s = sim(0)
    ped = s._object(1, 0)
    ped.update(x0=0.0, y0=15.0, vx=0.0, vy=-1.5, speed=1.5, phase=math.pi / 2)
    _, _, v_ped, _ = s._returns(ped, 0.0)
    car = s._object(3, 1)
    car.update(x0=0.0, y0=15.0, vx=0.0, vy=-5.0, speed=5.0)
    car["car"] = s._car_outline(math.pi)
    _, _, v_car, _ = s._returns(car, 0.0)
    assert float(v_ped.max() - v_ped.min()) > 2 * 1.5
    assert float(v_car.max() - v_car.min()) < 0.2


def test_rolling_wheel_rims_run_from_zero_to_twice_the_bike_speed():
    s = sim(0)
    bike = s._object(2, 0)
    bike.update(x0=0.0, y0=20.0, vx=0.0, vy=-5.0, speed=5.0)
    _, _, v, _ = s._returns(bike, 0.0)
    wheels = v[-16:]
    assert float(wheels.min()) < 1.0 and float(wheels.max()) > 8.5


def test_about_43_percent_of_sequences_hold_no_object():
    """CARRADA, Table II: 7,193 of 12,666 frames carry an object."""
    n = 300
    empty = sum(not sim(1000 + i).gen_sequence()["classes"] for i in range(n))
    assert abs(empty / n - (1 - 7193 / 12666)) < 0.07


# ---- RADIal simulator

def _one(power_db=30.0, rng=50.0, az=0.0, vr=-3.0):
    sim = SceneSimulator(seq_len=1, generator=torch.Generator().manual_seed(0))
    p = sim._render(torch.tensor([rng]), torch.tensor([az]), torch.tensor([vr]),
                    torch.tensor([power_db]), None)
    return 10 * torch.log10(p / 16 + 1e-12)


def test_ddma_offsets_match_radials_processing_code():
    """rpl.py: slots every 16 bins, Tx0 plus slots 5..15 (1-4 empty)."""
    assert DDMA_OFFSETS == (0, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240)


def test_every_return_is_replicated_once_per_transmitter():
    db = _one()
    tx0 = (-30) % 256                       # -3 m/s at 0.1 m/s per bin
    row = db[250]                           # 50 m at 0.2 m per bin
    top = sorted(int(k) for k in torch.topk(row, 12).indices)
    assert top == sorted((tx0 + o) % 256 for o in DDMA_OFFSETS)


def test_rendered_peak_carries_the_requested_power():
    assert abs(float(_one(power_db=30.0).max()) - 30.0) < 0.5


def test_near_range_is_dark_but_floored():
    """IF high-pass darkens near range; ADC noise stops it going to -inf."""
    sim = SceneSimulator(seq_len=1, generator=torch.Generator().manual_seed(1))
    x = sim.gen_sequence()["x"][0]
    near, mid = float(x[:10].median()), float(x[100:300].median())
    assert near < mid
    assert float(x.max() - x.min()) < 120.0


def test_labels_cover_moving_traffic_at_the_published_rate():
    """RADIal publishes 9,550 vehicles over 8,252 labelled frames (~1.16 per
    frame), so parked cars are scene rather than labels."""
    counts = []
    for seed in range(40):
        sim = SceneSimulator(seq_len=1, generator=torch.Generator().manual_seed(seed))
        labels = sim.gen_sequence()["labels"][0]
        assert all(l["moving"] for l in labels)
        counts.append(len(labels))
    assert 0.5 < sum(counts) / len(counts) < 1.6


def test_a_car_ahead_shows_only_its_near_face():
    """The car was once five equally bright points (corners and centre), which
    rendered as three range spikes 2.25 m apart. The body shadows its far side:
    straight ahead, only the rear face reflects, so the whole return sits in
    one or two range bins, and the label is at that face."""
    sim = SceneSimulator(seq_len=1, generator=torch.Generator().manual_seed(0))
    car = sim._car(0.0, 30.0, 10.0, vy=5.0)
    visible, rcs = sim._car_returns(car, car["x"], car["y"])
    rng = torch.sqrt(car["x"][visible] ** 2 + car["y"][visible] ** 2)
    assert bool((car["ny"][visible] == -1).all())          # rear face only
    assert float(rng.max() - rng.min()) < 2 * 0.2           # within ~2 range bins
    assert abs(float(rng.min()) - (30.0 - 4.5 / 2)) < 0.05


def test_an_oblique_car_shows_its_near_end_and_one_side_weakly():
    sim = SceneSimulator(seq_len=1, generator=torch.Generator().manual_seed(0))
    car = sim._car(8.0, 20.0, 10.0, vy=5.0)
    car["scint"] = torch.zeros_like(car["scint"])
    visible, rcs = sim._car_returns(car, car["x"], car["y"])
    nx, ny = car["nx"][visible], car["ny"][visible]
    assert set(zip(nx.tolist(), ny.tolist())) == {(0.0, -1.0), (-1.0, 0.0)}
    side, rear = rcs[nx == -1.0], rcs[ny == -1.0]
    assert float(side.max()) < float(rear.min())            # grazing side is weaker
    total = 10 * torch.log10((10 ** (rcs / 10)).sum())
    assert abs(float(total) - 10.0) < 1e-3                  # published RCS kept


def test_a_car_over_the_radar_has_no_visible_face():
    """Traffic is not kept from driving through the ego car's position; with the
    radar inside a footprint no face points at it, and the frame must still
    render (the cache build crashed on exactly this)."""
    sim = SceneSimulator(seq_len=1, generator=torch.Generator().manual_seed(0))
    car = sim._car(0.0, 0.5, 10.0, vy=5.0)
    visible, rcs = sim._car_returns(car, car["x"], car["y"])
    assert not bool(visible.any()) and rcs.numel() == 0


def test_same_direction_traffic_moves_with_the_ego_car():
    """Speeds were drawn independently of the ego speed, so highway traffic
    closed at up to +/-16 m/s, past the +/-12.8 m/s the Doppler axis holds. A
    traffic stream's speeds spread ~15% around its mean."""
    from src.radial_sim import ROAD_TYPES
    sim = SceneSimulator(seq_len=1, generator=torch.Generator().manual_seed(0))
    road = ROAD_TYPES["highway"]
    rel = []
    for _ in range(200):
        for v in sim._vehicles(road, 7.0, v_ego=30.0):
            if v["vy"] > 0:
                rel.append(v["vy"] - 30.0)
    rel = torch.tensor(rel)
    assert len(rel) > 50
    assert abs(float(rel.mean())) < 1.0 and 3.5 < float(rel.std()) < 7.0
