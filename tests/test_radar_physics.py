"""The simulator's parameters must be traceable to published specifications,
so these tests pin the derivations rather than any value fitted to recordings."""
import pytest
import torch

from src.radar_physics import (RADIAL_SPEC, clutter_cnr_db, sample_rcs,
                               target_snr_db, window)


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


def test_target_snr_follows_the_radar_equation():
    near, far = target_snr_db(50.0, 10.0), target_snr_db(100.0, 10.0)
    expect = 40 * torch.log10(torch.tensor(2.0)).item()
    assert float(near - far) == pytest.approx(expect, rel=1e-4)
    big, small = target_snr_db(100.0, 20.0), target_snr_db(100.0, 10.0)
    assert float(big - small) == pytest.approx(10.0)


def test_clutter_falls_more_slowly_than_targets():
    """The illuminated ground patch grows with range, so clutter falls as
    1/R^3 while a point target falls as 1/R^4."""
    d_clutter = float(clutter_cnr_db(50.0) - clutter_cnr_db(100.0))
    d_target = float(target_snr_db(50.0, 10.0) - target_snr_db(100.0, 10.0))
    assert d_clutter < d_target
    expect = 30 * torch.log10(torch.tensor(2.0)).item()
    assert d_clutter == pytest.approx(expect, rel=1e-4)


def test_rcs_population_is_dominated_by_cars():
    torch.manual_seed(0)
    rcs, idx = sample_rcs(2000)
    assert (idx == 0).float().mean() > 0.7
    assert 8.0 < float(rcs[idx == 0].mean()) < 12.0


def test_hamming_window_matches_radials_processing_code():
    w = window(256, "hamming")
    assert float(w[0]) == pytest.approx(0.08, abs=1e-6)
    assert float(w.max()) == pytest.approx(1.0, abs=1e-3)
    assert torch.equal(window(64, "none"), torch.ones(64))
    with pytest.raises(ValueError):
        window(16, "blackman")


# ---- the simulator driven by the specification, with nothing fitted --------







