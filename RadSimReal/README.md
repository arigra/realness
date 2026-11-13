# RadSimReal — executable Python reconstruction, v0.2

This project implements RadSimReal's **reflectors → shifted 3D PSFs → tensor noise → Doppler maximum** pipeline, native **RADDet and CARRADA** workflows, mesh/RF simulation, and three trainable Cartesian detector families. It runs beside datasets mounted on your HPC; datasets do not need to be uploaded to this chat.

Version 2 fixes the previous implementation's main gaps: native readers/exports are implemented, RADDet and the probabilistic detector are implemented, and evaluation uses **Cartesian boxes in metres**. RADDet preserves original Cartesian rectangles; CARRADA explicitly converts its radar-support annotations to Cartesian enclosures. Genuine oriented boxes are supported when supplied. The old RA-image detector remains available for compatibility with its own different metric domain.

**This is an independent reconstruction, not the authors' released implementation or a verified reproduction of their published scores.** Their experiment-specific code, calibrated assets/materials, complete training recipes and exact CARRADA split are unavailable. No actual RADDet/CARRADA recordings were accessible in this delivery environment. Native-format tests use marked synthetic fixtures. [Reproduction notes](REPRODUCTION_NOTES.md) distinguish implemented method, adaptations and missing experimental evidence.

Reference: Bialer & Haitman, *RadSimReal*, CVPR 2024. [Paper and supplementary Appendices A–D](https://arxiv.org/html/2404.18150v1) · [Project](https://yuvalhg.github.io/RadSimReal/) · [Official repository](https://github.com/yuvalHG/RadSimReal).

## Install

Use Python 3.10+ with an appropriate CPU/CUDA PyTorch build. Python 3.12 and CPU execution were used for delivery.

```sh
python -m venv .venv
# Linux / HPC:
. .venv/bin/activate
# Windows PowerShell instead: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q -p no:cacheprovider --basetemp work/pytest
```

The simulator downloads no datasets or weights at runtime. A GPU job needs the cluster's normal allocation and compatible PyTorch installation. See [TEST_REPORT.md](TEST_REPORT.md) for actual execution evidence.

## Run without datasets

From the extracted project directory, use fresh output directories:

```sh
# Geometry, visible reflection points, PSF, RA/RD figures and full tensor
python scripts/mesh_demo.py --out runs/mesh_demo

# Full 3D tensors, moving CAD vehicles, physical annotations and fixed splits
python -m radsimreal native generate --config configs/cartesian_demo.yaml --vehicle-assets configs/assets/vehicle_assets.json --out runs/cartesian_data --train 16 --val 4 --test 4 --max-samples 2000

# Short training check; omit --max-steps for all requested epochs
python -m radsimreal bev-train --model unet --data runs/cartesian_data --out runs/unet --epochs 2 --max-steps 4 --base-channels 8 --batch-size 4
python -m radsimreal bev-evaluate --data runs/cartesian_data --checkpoint runs/unet/best.pt --out runs/unet/test_metrics.json
python -m radsimreal bev-infer --data runs/cartesian_data --checkpoint runs/unet/best.pt --index 0 --out runs/unet/predictions.png
```

`--model raddet` uses the native 3D tensor and its source-derived Cartesian head. `--model probabilistic` uses a ResNet18/skip-decoder anchor detector with localization variance. Small `--base-channels` pilots are explicitly reduced models; see [detector documentation](docs/DETECTOR_IMPLEMENTATION.md) for reference widths, two-stage RADDet training, losses, adaptations and source hashes. `--oriented` requires genuine yaw for every target.

`python -m radsimreal demo --config configs/demo.yaml --out runs/legacy_demo` retains the original procedural reflection-point overview. The `native generate` command above exercises the new mesh pipeline. The general `generate`, `train`, `evaluate` and `infer` commands also dispatch version-2 data to Cartesian workflows.

## Use your HPC datasets

Read [docs/HPC_GUIDE.md](docs/HPC_GUIDE.md), or give the cluster agent [docs/HPC_AGENT_PROMPT.txt](docs/HPC_AGENT_PROMPT.txt). Enter paths on the cluster itself. Start with native inspection:

```sh
python -m radsimreal native inspect --dataset raddet --root /cluster/path/RADDet --profile configs/native_raddet_reference.json --trusted-labels --reference-cartesian --out runs/raddet_inspection
python -m radsimreal native inspect --dataset carrada --root /cluster/path/Carrada_RAD --profile configs/native_carrada_reference.json --out runs/carrada_inspection
```

CARRADA's full `RAD_numpy` tensors are a separate release from its processed RA/RD images. If annotations are separate, add `--annotations /cluster/path/annotations_frame_oriented.json`. Real-valued tensors need an explicit verified `--representation power|amplitude|db`; shape alone cannot identify those units. Profiles record unresolved coordinate assumptions that must be checked against local metadata and overlays.

The workflow provides:

1. Native indexing and scene/sequence-disjoint splits, keeping original arrays untouched.
2. Diagnostic plots and candidate stationary-target/noise regions from TRAIN sequences only.
3. Selected-target PSF measurement, energy cropping, measured noise covariance or empirical samples, and portable sensor products.
4. Mesh simulation using matching range/angle/Doppler coordinates, optional moving OBJ vehicle assets/static geometry, full tensor storage and native-format export.
5. Held-out distribution diagnostics, all three detector families, optional matched native-data baselines, physical IoU/AP and checkpoint resume.

The integrated `native run` command records inputs and completed stages. Without a verified selection it writes candidates and requests inspection. `--allow-provisional` explicitly enables an unvalidated execution pilot. A bright clutter peak is never silently certified as a point reflector. The guide contains complete commands, selection format, resume rules and storage estimates.

## Included executed example

The self-contained mesh example is in `examples/v2/mesh_demo/`, including
[scene/reflector/PSF overview](examples/v2/mesh_demo/mesh_overview.png),
[radar heatmap](examples/v2/mesh_demo/radar.png), full tensor and reflection arrays.

![Cartesian mesh simulation and actual predictions](examples/v2/predictions.png)

`examples/v2/cartesian_demo/` contains 16/4/4 synthetic CAD scenes and full tensors. `examples/v2/unet_training/` contains an actual 100-epoch, 400-update CPU run, optimizer state and history. Validation selected the best checkpoint at update **96**; `last.pt` records update 400. Neither is an author-pretrained or real-data-trained model.

```sh
python -m radsimreal bev-infer --data examples/v2/cartesian_demo --checkpoint examples/v2/unet_training/best.pt --index 0 --score-threshold 0.2 --out runs/included_predictions.png
```

On its four synthetic test scenes, this checkpoint achieved Cartesian AP **0.9503 / 0.6941 / 0.0625** at IoU **0.1 / 0.3 / 0.5**, using global PR-area AP, best-overall matching, score cutoff 0.05 and NMS 0.3. These small synthetic results are an executable example; the low AP at 0.5 exposes limited localization accuracy. The displayed overlay uses cutoff 0.2 for legibility. See the saved metrics and [TEST_REPORT.md](TEST_REPORT.md), including separate fixture-native pilots.

## Scientific implementation map

| Component | Implemented behavior | Fidelity boundary |
|---|---|---|
| Radar operator | Sparse shifted kernels and independent FFT convolution; sub-bin placement; finite range/angle and optional periodic Doppler | Appendix B Eq. (4)–(5) agree for the declared discrete samples; sub-bin interpolation is a discretization |
| PSF/noise | Streaming stationary-target crops; complex alignment; 99% energy support; measured quadrature covariance or empirical IID noise | Appendix A reconstructed; isolation, shift invariance, target gain and spatial statistics need measurements |
| Environment | Triangle meshes, BVH visibility, area sampling, dielectric/Fresnel and roughness terms, moving CAD, optional read-only CARLA bridge | Approximate single-bounce RF model; author assets/materials, multipath and weather are unavailable |
| Data | Native RADDet/CARRADA tensors, axis/domain profiles, original labels, lazy manifests, native synthetic exports | Local conventions need verification; CARRADA radar-support enclosures differ from physical vehicle footprints |
| Models | Coordinate U-Net, released RADDet topology/Cartesian head plus 3D pretraining, Dong probabilistic encoder/decoder and uncertainty loss | Source-derived families with documented adaptations; unreleased RadSimReal recipes are not asserted |
| Evaluation | Physical Cartesian and exact rotated polygon IoU/NMS; global PR AP and released RADDet frame-mean AP | Reports record protocol/domain; unreleased RadSimReal evaluator settings remain unverified |
| Published protocol | RADDet scene intervals and test scenes 9/11 intersected with observed original IDs | Missing available-file IDs and exact CARRADA author split are not invented |

Complex mode sums fields before magnitude-squared conversion. Power-only input uses a declared incoherent surrogate and cannot recover phase. Empirical noise preserves a marginal distribution, not spatial correlations. Material gain and scene geometry must be calibrated separately from the PSF.

## Project map

`radar/` contains formation, PSF and noise; `simulation/` contains geometry, materials, assets and scattering; `native/` contains readers, calibration, export and resumable workflows; `datasets/` connects generated/native data to `bev/` models, training, inference and evaluation. Legacy RA modules remain under `models/`, `training/` and `evaluation/`.

Further details: [native contracts](docs/NATIVE_FORMATS.md), [scattering assumptions](docs/SCATTERING.md), [detector sources](docs/DETECTOR_IMPLEMENTATION.md), [validation protocol](docs/VALIDATION_PROTOCOL.md), [source audit](docs/SOURCE_AUDIT.md).

MIT covers this project's code; the RADDet port includes upstream attribution. External datasets, papers, CARLA assets and weights retain their own terms and are not redistributed.

