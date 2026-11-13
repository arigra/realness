# Array, calibration and label contracts

This document describes the retained **version-1 generic RA-image schema** and
low-level array interfaces. V2 native readers, physical Cartesian targets,
sensor calibration and native exports are documented in
[NATIVE_FORMATS.md](NATIVE_FORMATS.md) and [HPC_GUIDE.md](HPC_GUIDE.md). V2 uses
`manifest.json` with `format_version:2`, physical `box_bev`/optional genuine
`oriented_box_bev` targets, and separate original/native label provenance.
Its Cartesian evaluation does not use the legacy RA-image rectangle metric.

All arrays are linear response values; dB exists only at the display/network-input boundary. Distances are metres, velocities m/s, angles radians internally. World coordinates are x right (lateral), y forward, z up. Positive radial velocity is receding. Do not apply propagation loss twice.

## Radar grid

Tensor order is **(range, Doppler velocity, azimuth coordinate)**, image order **(range, azimuth)**. A range-bin center is `range_min + i*range_step`; a velocity-bin center is `velocity_min + j*velocity_step`. Azimuth centers are uniformly spaced in either `sin(theta)` (default) or angle, with endpoints from the configured degree limits. A single shift-invariant kernel assumes stationarity in that declared coordinate. A measured PSF in a different coordinate cannot be substituted by relabeling its axis.

## Dataset on disk

`config.json` records the grid/scene and seed. `manifest.json` contains `format_version:1` and `splits` mapping `train`, `val`, `test` to lists of `{id,image,annotation}`. Paths are relative to the dataset root. An image NPZ holds float32 `image` of shape `(R,A)` and, optionally, full signed/complex `tensor` of shape `(R,D,A)`. JSON annotations contain `objects`, each with `label:car`, `box:[xmin,ymin,xmax,ymax]` and optional object/scene/visibility provenance.

Boxes use **normalized RA pixel-edge XYXY**: x is azimuth column, y is range row. Fractional bin center `i` maps to normalized edge coordinate `(i+0.5)/number_of_bins`. The upper/lower extents are continuous and boxes clip to [0,1]. Procedural labels are amodal full ground footprints, sampled on all four edges then enclosed in an RA rectangle. They can include fully occluded objects; `has_simulated_return` and `in_domain_reflection_count` expose that policy. Surface samples have height, while labels use the ground footprint. These choices are not the authors' unpublished label generator.

**RA-bin area IoU is not Cartesian/BEV box IoU.** The legacy v1 AP is an internal research diagnostic and is not a reproduced benchmark score. V2 supplies Cartesian heads/evaluation using matching original labels or explicitly declared converted-region targets. No conversion of four corners can recover geometry lost in an enclosing RA box.

`RadarDataset` returns 3 channels: fixed normalized `10*log10(max(image,1e-12))` clipped between -30 and +30 dB, lateral coordinate divided by maximum range, forward coordinate divided by maximum range. These bounds are choices, not estimated from test images. Checkpoints store them. Calibration may justify other fixed bounds; specify them before training. No absolute RCS is implied by demo units.

## External real arrays

Convert your dataset to the declared sensor grid yourself; importing never resizes an array or guesses channel/axis conventions. Write a JSONL record per frame, for example:

```json
{"id":"frame_0001","split":"train","scene_id":"townA","image_path":"arrays/frame_0001.npy","representation":"power","boxes":[[0.3,0.2,0.4,0.3]],"provenance":"My calibrated grid and labels"}
```

`image_path` resolves relative to the JSONL file. NPY arrays or NPZ arrays (key `image`, override `array_key`) may contain `(R,A)` images or `(R,D,A)` tensors. Declare `representation` as `power`, `amplitude` (converted using squared magnitude), or `db` (power dB converted using 10**(dB/10)). Full tensors reduce by Doppler maximum. Convert amplitude before averaging/projection if required by your sensor processing; the importer follows this specified order only.

`boxes` contains car labels only in the normalized RA convention. Preserve all distinct test scene IDs. The importer rejects scene overlap and identical converted image content across splits, including images with renamed IDs. It stores source path, dtype, shape, representation, transformation and SHA256 of the converted image. It preflights inputs then rereads them to keep memory bounded to a frame. Source changes during import raise an error; if an I/O failure occurs during writing, discard the incomplete directory before retrying. The fixed-grid class-car schema does not decode native RADDet pickle labels, CARRADA JSON conventions or CRUW point labels automatically.

```sh
python -m radsimreal import-real --manifest converted.jsonl --config sensor.yaml --out data/real
```

## External ray-tracer or LIDAR-derived reflectors

NPZ keys: `xyz` (or `positions`) float `(N,3)`, `velocity` (or `velocities`) float `(N,3)` **already relative to radar ego motion**, `coefficients` `(N,)`, optional `object_ids` `(N,)` and JSON-string `metadata`. Coefficients must already include propagation, material response and coherent phase when using complex mode, in the same units as the PSF. Intensity-mode coefficients must be real and nonnegative. Raw LIDAR intensity is not RF reflectivity. Calibrated point/material/mesh processing is your responsibility.

```sh
python -m radsimreal simulate-points --points reflections.npz --config sensor.yaml --out runs/points
```

## Measured PSF and noise

Calibration NPY shape is `(frames,R,D,A)` (NPZ key `tensors`). Supply a stationary isolated target bin and a reflection-free boolean NPY mask of shape `(R,D,A)` (NPZ key `mask`). Phase-aligned complex frames are necessary for coherent averaging. A measured intensity PSF supports the intensity surrogate; it cannot recreate a phase-bearing field PSF. The CLI records the declared representation, noise moments, selected support, retained energy and all grid spacings. It preserves target gain; calibrate scattering coefficients to that gain rather than assuming alpha=absolute RCS.

```sh
python -m radsimreal calibrate --tensors stationary.npy --noise-mask empty_roi.npy --target 40 8 48 --half-widths 4 3 10 --config sensor.yaml --out runs/calibration
python -m radsimreal demo --config runs/calibration/calibrated_config.json --out runs/calibrated_demo
```

Generated `calibration.npz` stores `psf` plus a JSON `metadata` field. Metadata `grid` requires `range_step`, `velocity_step`, `angular_step`, `azimuth_coordinate`, `representation`. High-level generation checks those values strictly and copies the full calibration metadata into the dataset. Direct `load_psf(...,strict_metadata=False)` exists for controlled legacy use. Noise-law/covariance validation and spatial-invariance validation remain separate measurements; matching the grid alone does not establish sensor fidelity.
