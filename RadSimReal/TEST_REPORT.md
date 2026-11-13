# Executed validation report — v0.2

Delivery date: **2026-11-10**. Windows, Python **3.12.10**, CPU execution. This records software execution and synthetic experiments, not reproduction of the CVPR real-data benchmark.

## Environment

NumPy **2.5.1**, SciPy **1.18.0**, PyTorch **2.15.0+cpu**, Matplotlib **3.11.1**, PyYAML **6.0.3**, pytest **9.2.0**. A task-local environment was used. Standard wheel construction for version 0.2.0 and package/script compilation succeeded. Only this environment was executed; broad dependency bounds are not a cross-platform lock.

## Full automated suite

```sh
python -m pytest -q -p no:cacheprovider --basetemp <workspace>/work/pytest-v2-release
```

Result: **190 passed, 68 subtests passed in 44.35 seconds**.

Coverage includes:

- Shifted PSFs versus independent FFT convolution, edges/duplicates, off-grid discretization, matched-filter identity, measured support/energy and metadata, extreme finite scales, domain conversion and overflow.
- Complex Gaussian covariance/mean, empirical marginal sampling, noise-free mode, Doppler aliasing and deterministic generation.
- Triangle/BVH visibility, material/incidence/range behavior, zero-return filtering, static object IDs, OBJ parsing/triangulation, moving CAD placement/velocity/material provenance and preserved-unit dimension checks.
- Native RADDet/CARRADA layouts, frame identities, axis/order/sign/domain conversions, native label preservation, streamed annotation loading, missing versus explicitly empty CARRADA frames, training-only calibration eligibility, native export/read-back and wrapped support.
- The CARRADA box-only angle-reversal defect found in final review: four angle/sin-angle and forward/reversed cases now verify the same mapping as dense support.
- Physical Cartesian and exact rotated polygon geometry, projection masks, target assignment, known IoU/NMS/AP cases, both AP aggregation/matching protocols, source objectness scoring and unclipped predicted boxes.
- All three distinct detector families' forward/backward, optimizer update, validation, checkpoint reload, physical inference/evaluation and resume; RADDet 3D loss/targets/pretraining, frozen parameter/BN preservation, source sampling transfer checks and probabilistic physical-variance transforms.
- Complete native CLI pilots for both formats, with actual tensor I/O, calibration, mesh simulation/export, distribution reporting, training and AP; completed-run resume forbids retraining and preserves checkpoint/calibration/synthetic bytes. First-selection transition and later selection-mutation rejection are covered.
- Original RADDet interval endpoints, holes, observed-ID semantics and exactly 1,962 held-out mapped IDs; 68 subtests exercise the scene table.

Writable test directories were explicitly supplied to avoid the host's restricted shared temporary folder. No tests depend on access to private recordings.

## Fresh all-model native-format pilots

[examples/v2/native_pilots.json](examples/v2/native_pilots.json) records commands, source locks, histories, protocols and actual scores for two fresh completed workflows.

**Inputs were controlled generated native-format fixtures, not real RADDet/CARRADA recordings.** Each fixture contains an isolated synthetic 3D return and generated noise, with labels intended to exercise coordinate/file interfaces; it is not a physically measured car. Confirmation flags apply only to those controlled synthetic assumptions.

For each dataset format, execution completed native inspection/preparation, selected PSF/noise measurement, TRAIN-only relative gain fitting, moving CAD mesh generation, native export, distribution diagnostics, and three synthetic-trained plus three native-fixture-trained Cartesian model arms. The generated set was 4/2/2 scenes. All **12 Cartesian arms** performed exactly **two optimizer updates**. Both RADDet comparison arms additionally performed **two 3D-pretraining updates**, then used a frozen backbone; parameters and BN moments matched the transferred best checkpoints. CARRADA used the matched end-to-end recipe in both arms because public native 3D RAD boxes are absent.

Global PR-area and reference frame-mean AP were both executed and explicitly recorded. These tiny pilots do not establish convergence, sensor realism, real-data AP or superiority of any model. The report does not redistribute large pilot checkpoints.

The portable fixture generator is `tests/native_fixtures.py`; the entire small native workflow can be repeated by running the test suite. To create larger explicit fixtures from the source checkout:

```sh
python -c "import sys; sys.path.insert(0,'tests'); from native_fixtures import create_fixture; create_fixture('runs/fixtures','raddet'); create_fixture('runs/fixtures','carrada')"
```

## Actual included synthetic training run

`examples/v2/cartesian_demo/` contains 16 training, four validation and four test CAD mesh scenes, full complex tensors and original physical Cartesian/oriented geometry. The sensor/PSF/material settings are illustrative and uncalibrated.

An eight-base-channel coordinate U-Net ran **100 epochs / 400 optimizer updates** on CPU. Final training loss was **0.37298535**, final validation loss **2.97807956**. Validation selected the best checkpoint at **update 96**, with validation loss **1.88585**; `last.pt` records update 400. Both optimizer states and full history are included. Further training did not consistently improve validation, which is why best and last differ.

The untouched four-image synthetic test set gave:

| Physical Cartesian IoU threshold | Global PR-area AP |
|---|---|
| 0.1 | 0.9502840909 |
| 0.3 | 0.6941287879 |
| 0.5 | 0.0625 |

Evaluation used `best_overall` matching, score cutoff 0.05, NMS IoU 0.3 and top-k 100. This small result is not real-data accuracy evidence. AP at 0.5 exposes limited localization quality. The included PNG uses score cutoff 0.2 solely for legibility; its boxes/scores are actual checkpoint predictions, not hand-drawn outputs.

```sh
python -m radsimreal bev-evaluate --data examples/v2/cartesian_demo --checkpoint examples/v2/unet_training/best.pt --out runs/reloaded_metrics.json
python -m radsimreal bev-infer --data examples/v2/cartesian_demo --checkpoint examples/v2/unet_training/best.pt --index 0 --score-threshold 0.2 --out runs/reloaded_predictions.png
```

The independent `scripts/mesh_demo.py` also executed: saved geometry/visible reflection coefficients, complex tensor, PSF, RA/RD images and annotations are in `examples/v2/mesh_demo/`. Both overview and detection figures were visually inspected.

## Commands, packaging and boundaries

All 16 complete command lines in HPC_GUIDE.md passed argument parsing; this checks CLI compatibility, not access to their placeholder data paths. Fresh archive execution is recorded in [examples/v2/ARCHIVE_VALIDATION.json](examples/v2/ARCHIVE_VALIDATION.json). The packaging process verifies CRC and SHA256 contents, excludes caches/build folders and old v1 run artifacts, and checks the extracted module location before executing it.

Fresh extraction passed the complete **190-test / 68-subtest suite**, module-location check, compilation, CAD mesh demo/generation, saved checkpoint evaluation/inference and all standalone/selected CLI help commands. Reloaded AP and complete prediction records matched the included originals exactly. See the archive evidence JSON for individual outputs and timings.

No actual user dataset or measured point-target capture was accessible here. No held-out real AP, exact author split reproduction, full paper-scale training, GPU experiment, live CARLA server, FID/weather study or published timing speedup was executed. The code supports the next real-data experiments through the HPC guide; those results must be produced on the cluster holding the recordings.

