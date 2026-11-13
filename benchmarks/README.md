# Detector benchmarks: reproduce each dataset's published detector

Step 1 of the simulator check: train the official detector of each real dataset with the official code and
recipe, and reach (or beat) the published numbers, before any simulator comparison.

| Dataset | Detector (official code) | Published target | Split |
|---|---|---|---|
| RADIal | FFT-RadNet | AP 96.84, AR 82.18 (IoU 0.5); R err 0.11 m, A err 0.17 deg | sequence split 6231 / 986 / 1035 |
| CARRADA | TMVA-Net (MVRSS) | mIoU RD 58.7, RA 41.3 | official 18 / 6 / 6 sequences |
| RADDet | RADDet (RadarResNet + YOLO) | 2D AP 0.796 / 0.727 / 0.516 / 0.205 at IoU 0.1 / 0.3 / 0.5 / 0.7; 3D 0.764 / 0.563 / 0.251 / 0.059 | official random 80/20 |

## Setup
Clone the three repos at the commits in `patches/UPSTREAM.txt` into this folder, apply `patches/*.diff`, and
copy `configs/` into place (`raddet_config.json` -> `RADDet/config.json`, `mvrss_config.ini` ->
`MVRSS/mvrss/config_files/config.ini`, the others to `runs/radial/config.json`, `runs/carrada/tmvanet.json`).
Envs: `.venv` (system torch + pkbar, shapely, polarTransform, mvrss `-e --no-deps`), `.venv_tf` (tensorflow[and-cuda]==2.15.1, numpy<2).

## What we changed in the official code, and why
- **MVRSS**: resume from a per-epoch state (`MVRSS_RESUME`); early stop after `MVRSS_PATIENCE`=30 epochs without a
  better validation precision (paper config otherwise; model selection by val as in the official code).
- **RADDet**, TF 2.15 instead of 2.3 (2.3 does not run on this GPU / Python):
  - `model/layers.py`: their BatchNorm is meant to always run in inference mode. TF 2.3 constant-folded the
    `tf.logical_and(False, trainable)` flag; inside `tf.function` in TF 2.15 it does not, Keras then updates the
    moving statistics and the loss is NaN from step 2. A Python `False` restores the TF 2.3 behaviour.
  - `util/loader.py`, `util/helper.py`: optional local cache of `log10(|RAD|^2 + 1)` (float32, bit-identical to
    their preprocessing; `scripts/cache_raddet_logmag.py`). Reading 33 MB complex files from network storage made
    training I/O-bound (~6 s/step).
- **RADIal**: unmodified.

## Running
`run_all.sh` starts every job that is neither running nor finished (`runs/<name>.done`); `supervise.sh` calls it
every 5 min (stop with `touch STOP`). Jobs resume from their own checkpoints.

    nohup ./supervise.sh > /dev/null 2>&1 &
