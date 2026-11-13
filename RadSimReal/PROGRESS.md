# Progress log (cluster run)

## 2026-11-10 — restore, environment, stage 1, start of stage 2

- Restored with `RESTORE_RadSimReal_v2.txt` from `../RadSimReal_bundle/RadSimReal_v2_SOURCE.txt`
  (sha256 846fa855414dd6fb0d09f423feb59253f82fd3dc820526afa3401c8c00666dfb): 113 checksum-verified files;
  `sha256sum -c MANIFEST.sha256` passes.
- Env: `.venv` (system Python 3.10.12, `--without-pip`) + `zz_shared_readonly.pth` pointing read-only at
  `~/.venv` site-packages (torch 2.12.1+cu130, numpy 2.2.6, scipy 1.15.3, matplotlib 3.10.9, pytest 9.1.1).
  Installed with `python -m pip install --no-build-isolation -e ".[test]"` (no-build-isolation to use local setuptools 81 offline).
  Nothing installed into `~/.venv`.
- Tests: `190 passed, 68 subtests passed` (logs/pytest_initial.log).
- Smoke: mesh_demo + synthetic generate/bev-train/bev-evaluate all OK (logs/smoke_workflow.log). Execution check only.

### Native inspection
- RADDet root `radar data/RADDet/raw`: 10,158 frames, scenes 0–14, complex RAD, labels decode (`--trusted-labels`).
  Overlay (runs/raddet_inspection) looks plausible; affine still UNVALIDATED (needs systematic overlay check).
- CARRADA root `radar data/datasets_master/Carrada_RAD`, annotations `radar data/Carrada/annotations_frame_oriented.json`:
  12,666 frames, 30 sequences.
  - Evidence for representation = **amplitude** (seq 2019-09-16-12-52-12, frames 100 and 500):
    processed `range_angle_numpy` == max over Doppler of RAD exactly (rel RMS ~1e-16) → RAD axes (R,A,D);
    processed `range_doppler_numpy` ≈ 10·log10(mean_A RAD²) with range axis flipped
    (slope 0.9995, offset 0.02 dB, rel RMS 4.5e-4). Amplitude-sum alternatives fit 200× worse.
  - The 4 inspected frames had 0 cars → geometry/angle-grid not yet verified.

## Next
1. CARRADA: inspect frames that contain cars (choose IDs from annotations) to check range direction,
   angle grid (uniform angle vs sin) and Doppler sign; same for RADDet across several scenes.
2. Save verified local profiles (new JSON, not overwriting configs/).
3. `native prepare` with explicit splits (RADDet test scenes 9/11; CARRADA from data_seq_ref.json).

## 2026-11-10 (evening) — data verified, RADDet calibrated; decision needed on scene clutter

Paper targets (arXiv 2404.18150 Table 1, AP@0.1/0.3/0.5, RADDet test scenes 9/11):
U-Net real 84.76/83.01/55.53, sim 85.63/82.16/57.64; Probabilistic real 83.31/74.80/40.68, sim 82.75/75.83/52.36;
RADDet real 83.69/72.96/47.95, sim 83.48/73.91/46.63. Paper: input = RA image (Doppler max); no epochs/LR/batch given.

### Bugs fixed (tests added; suite 192 passed)
- Reader: local RADDet copy has test/gt/part_2/part_2/*.pickle (1,000 files) -> fallback lookup (dataset untouched).
- RADDet gt 'boxes' Doppler centre is ABSOLUTE [0,64) (all 28,401 boxes: min 0, median 31.5, max 63.5);
  code added D//2 (reader) and subtracted it (native export); fixture encoded the same error. Fixed all three.

### Verified profiles (profiles/*_verified.json; evidence inside)
- RADDet: label RAD<->Cartesian consistency (0.077 m / 0.0033 sin); Doppler positive = APPROACHING (reference said receding);
  paper's 'car' = car+truck+bus (original split 16,755/4,135 exact) -> car_labels [car,truck,bus].
- CARRADA: amplitude; annotations uniform-angle (pi/256) but TENSOR azimuth is sine-spaced (peak col = 122 sin + 128);
  range origin R; Doppler positive approaching.

### Prepared (runs/*_prepared_v)
- RADDet: train 7,987 (13 scenes minus scene_4) / val 209 (scene_4, my choice; paper has no val) / test 1,962 (scenes 9,11 = paper); 20,890 objects.
- CARRADA: official data_seq_ref split 8,088 / 2,448 / 2,130 (paper's 2,208/276 not reproducible from any recorded split).

### RADDet calibration (runs/raddet_calibration, selections/raddet_verified_selection.json)
- Target scene_12 canonical (99, zero-D 31, 88): 205 captures, phase coherence 0.9998, CV 0.008, SNR ~45 dB;
  azimuth cut identical to 4 other independent stationary reflectors. PSF 15x1x129 (99.7% energy).
- Noise: far range |v|>10.9 m/s, var 2771, circular, zero-mean; empirical law.
- Train-only gain fit: runs/raddet_calibration/gain_fitted_config.json.

### Blocking issue
Bench synthetic (8 frames) vs real train: synthetic is nearly empty (1-5 cars + 30 clutter points); real frames are dense
static clutter (median cell 47.8 vs 41.5 dB, p90 62.3 vs 44.7 dB). Paper used CARLA city environments.
See runs/bench/raddet_validation8/comparison.png.

### Compute
VS Code job 18568 ends 2026-11-11 03:17 (16 GB mem cap); no sbatch inside container. Training loader is
num_workers=0 reading 32 MiB cubes (>=17 min/epoch on RADDet).

## 2026-11-10 (night) — Step 1 (RADDet, U-Net, real vs sim) in progress

### More fixes (tests added; suite 195 passed)
- RADDet gt boxes use INCLUSIVE bin centres (odd size <-> integer centre, all axes) and Doppler boxes wrap
  (span -32..99); centre 63.5 flipped to -0.5 and failed validation -> canonical Doppler centre taken mod D
  (reader and native export).
- Added RA-image cache (scripts/cache_ra_images.py -> runs/raddet_prepared_v/ra_cache, 10,158 frames) used by
  BEVDataset only when the model does not consume the full cube; identical items verified by test.
  Generated datasets no longer decompress the cube when the model does not need it.

### Real-trained U-Net (running)
- `scripts/step1_raddet_unet.sh real` (fixed hyperparameters documented in the script). ~12 min/epoch, 40 epochs.
  Resume by rerunning the same command (uses last.pt). Log: logs/step1_unet_real.log.

### CARLA path
- CARLA 0.9.15 tarball: /truenas/home/arigra/permuter/ariGranevich/carla/CARLA_0.9.15.tar.gz (8.4 GB).
  Vulkan works without root: loader + X11 libs in .../carla/vulkan (VK_ICD_FILENAMES=.../vulkan/nvidia_icd.json,
  LD_LIBRARY_PATH=.../vulkan/root/usr/lib/x86_64-linux-gnu); 1 physical device found.
- Extracted to node-local scratch for speed (re-extract per node, ~8 min).
- Starting the CARLA server was BLOCKED by the session's permission check (downloaded binary) -> needs Ari.
- Code ready: scripts/carla_capture.py (sidewalk static sensor, semantic LiDAR rays + vehicle boxes),
  radsimreal/simulation/carla_reflectors.py (ITU-R P.2040 materials, patch area r^2 dOmega/cos), tested;
  scripts/generate_carla_dataset.py (train-only gain fit, position-disjoint val, parallel, resumable).

### Next commands (after CARLA server is up on port 2000)
  .venv/bin/python scripts/carla_capture.py --out runs/carla_capture --positions-per-town 100 --frames-per-position 20
  .venv/bin/python scripts/generate_carla_dataset.py --captures runs/carla_capture --config runs/raddet_calibration/sensor_config.json \
      --prepared runs/raddet_prepared_v --fit-gain --gain-out runs/raddet_calibration/carla_gain_config.json
  .venv/bin/python scripts/generate_carla_dataset.py --captures runs/carla_capture --config runs/raddet_calibration/carla_gain_config.json \
      --prepared runs/raddet_prepared_v --out runs/raddet_carla_synthetic --train 10000 --val 1000
  ./scripts/step1_raddet_unet.sh sim

## 2026-11-10 23:30 — CARLA running (Ari approved)
- Server: scripts/carla_server.sh (map argument is ignored -> starts in Town10HD_Opt; other towns via load_world).
  Town03 crashed on load while U-Net training shares the GPU (NVRM out-of-memory in dmesg); Town10HD fine.
- Capture teardown fixed (release autopilot/walker AI before destroy). Never `pkill -f` with the process name
  inside the same shell command (kills the shell) -> scripts use `pkill -x CarlaUE4-Linux-`.
- Pilot (Town10HD, 30 frames) vs real train: RA dB p50 42.7/47.8, p90 61.5/62.3, p99 76.2/74.2,
  W1 3.3 dB (shipped simulator: 6.8). Gaps: synthetic too strong at 2-10 m (no elevation antenna
  pattern modelled), too weak beyond ~15 m. NOT tuned; decision for Ari (datasheet elevation pattern?).
- Full capture running: scripts/carla_capture_all.sh ... runs/carla_capture 100 20 <6 towns>, ~16 s/position
  (20 frames) -> ~27 min/town. Log logs/carla_capture_full.log. Resumable (DONE per position).
- Real U-Net: ~6-7 min/epoch once cache warm; epoch 7 at 23:28.
- DECISION (Ari, 2026-11-10): no elevation antenna pattern. Paper models azimuth-only radar
  ("only azimuth angle and without elevation"), no antenna gain/pattern, no mounting height. Keep simulator paper-faithful.
- Correction: paper Table 1 row 3 (U-Net 22.33/19.46/14.04) is test RADDet / train CARRADA.
