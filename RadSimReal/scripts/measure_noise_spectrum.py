"""Measure the per-axis noise power spectrum of the real radar from the TRAIN-only noise regions
of the verified selection (the same regions that give the noise level), and write a sensor config
that uses coloured Gaussian noise instead of independent per-bin draws.

Azimuth: the regions span all 256 azimuth bins, so the spectrum is the averaged |FFT_azimuth|^2.
Range/Doppler: the regions are shorter than the tensor axis, so the complex autocorrelation is
estimated for small lags and Fourier-transformed to a full-length spectrum (negatives clipped).
Each spectrum is scaled to mean one so the measured per-bin variance is unchanged.

  python scripts/measure_noise_spectrum.py --config runs/raddet_calibration/carla_vehicle_gain_config.json \
      --out-config runs/raddet_calibration/carla_vehicle_coloured_config.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from radsimreal.config import load_config, save_config
from radsimreal.native.workflows import reader_from_prepared
from radsimreal.native.calibration import record_for_id, _roi, json_write
from radsimreal.radar.noise import coloured_gaussian


def acf(x, axis, maxlag):
    x = x - x.mean()
    n = x.shape[axis]
    p = np.mean(np.abs(x) ** 2)
    return np.array([np.mean(np.take(x, range(0, n - L), axis=axis) * np.conj(np.take(x, range(L, n), axis=axis))) / p
                     for L in range(maxlag + 1)])


def spectrum_from_acf(c, length):
    full = np.zeros(length, dtype=complex)
    full[: len(c)] = c
    full[length - len(c) + 1:] = np.conj(c[1:][::-1])
    s = np.clip(np.real(np.fft.fft(full)), 0, None)
    return s / s.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default="runs/raddet_prepared_v")
    ap.add_argument("--selection", default="selections/raddet_verified_selection.json")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-config", required=True)
    ap.add_argument("--range-lags", type=int, default=16); ap.add_argument("--doppler-lags", type=int, default=4)
    args = ap.parse_args()
    reader, manifest = reader_from_prepared(args.prepared)
    train_ids = {r["id"] for r in manifest["splits"]["train"]}
    sel = json.loads(Path(args.selection).read_text())
    rois = []
    for reg in sel["noise_regions"]:
        if reg["record_id"] not in train_ids:
            raise ValueError(f"noise region {reg['record_id']} is not a TRAIN record")
        rois.append(_roi(reader.load_tensor(record_for_id(reader, reg["record_id"])), reg["bounds"]).astype(np.complex128))
    cfg = load_config(args.config)
    R, D, A = cfg.radar.shape
    if any(r.shape[2] != A for r in rois):
        raise ValueError("azimuth spectrum needs regions spanning the full azimuth axis")
    s_az = np.mean([np.mean(np.abs(np.fft.fft(r - r.mean(), axis=2)) ** 2, axis=(0, 1)) for r in rois], axis=0)
    s_az = s_az / s_az.mean()
    c_r = np.mean([acf(r, 0, args.range_lags) for r in rois], axis=0)
    c_d = np.mean([acf(r, 1, args.doppler_lags) for r in rois], axis=0)
    s_r, s_d = spectrum_from_acf(c_r, R), spectrum_from_acf(c_d, D)
    out_cfg = Path(args.out_config)
    spec_path = out_cfg.with_name(out_cfg.stem + "_noise_spectrum.npz")
    np.savez_compressed(spec_path, range=s_r, doppler=s_d, azimuth=s_az)
    # check: synthetic coloured noise reproduces the measured neighbour correlations
    rng = np.random.default_rng(0)
    sim = coloured_gaussian((R, D, A), 1.0, (s_r, s_d, s_az), rng)
    check = {}
    for name, ax, L in (("range", 0, 4), ("doppler", 1, 2), ("azimuth", 2, 12)):
        real = np.mean([acf(r, ax, L) for r in rois], axis=0)
        check[name] = {"real_abs_acf": np.abs(real).round(3).tolist(), "synthetic_abs_acf": np.abs(acf(sim, ax, L)).round(3).tolist()}
    xr = np.concatenate([r.reshape(-1) for r in rois]).real
    cfg.radar.noise_model = "coloured_gaussian"
    cfg.radar.noise_spectrum_path = str(spec_path.resolve())
    save_config(cfg, out_cfg)
    json_write(out_cfg.with_suffix(".noise_spectrum.json"), {
        "method": "per-axis separable power spectrum from TRAIN noise regions; Gaussian law; per-bin variance unchanged",
        "regions": len(rois), "noise_std": cfg.radar.noise_std,
        "real_part_kurtosis_gaussian_is_3": float(np.mean((xr - xr.mean()) ** 4) / np.var(xr) ** 2),
        "test_frames_used": 0, "autocorrelation_check": check})
    print(json.dumps(check, indent=1))
    print("wrote", out_cfg, spec_path)


if __name__ == "__main__":
    main()
