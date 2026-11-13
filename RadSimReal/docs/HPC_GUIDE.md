# Run the native simulators beside RADDet and CARRADA

The software runs on the machine where the datasets are mounted. VS Code's
remote terminal is sufficient; no SSH connection or upload is made by these
commands. Dataset paths are entered locally. Original recordings remain in
place. Put run outputs outside the raw dataset directories.

The native readers, measurement workflow, mesh simulator, native-format exports
and all three Cartesian detector families are implemented. Their execution on
controlled fixtures is documented in the test report. **Sensor realism and
real-data AP must be measured on the supplied recordings.** Matching file
formats or completing a pilot does not establish the paper's reported accuracy.

## Installation and local paths

Use Python 3.10+ and an environment suitable for the cluster's installed
CPU/CUDA PyTorch build. From the extracted project folder, for example:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q --basetemp work/pytest
```

Examples below use a Linux shell in VS Code. Replace the three path values on
the cluster; they are not addresses to send to this chat:

```sh
RADDET_ROOT=/path/on/cluster/RADDet
CARRADA_ROOT=/path/on/cluster/Carrada_RAD
RADSIM_RUNS=/path/on/cluster/radsimreal_runs
mkdir -p "$RADSIM_RUNS"
```

Use the cluster's allocations/scheduler for substantial array processing and
GPU training. Bounded inspection/pilots identify coordinate or unit problems
before large jobs. Offline systems can use an existing environment or a local
wheelhouse; the simulator does not download a dataset or checkpoint at runtime.

## 1. Identify the exact native representation

RADDet normally contains complex `(R,A,D)=(256,256,64)` arrays under
`train|test/RAD/part*/`. CARRADA's full tensors under `sequence/RAD_numpy/` are
distributed separately from its processed RA/RD views. Native tensor dtype and
processing must be checked locally. Shape cannot identify a real-valued array
as amplitude, power or logarithmic response.

Reference profiles are provided in `configs/native_raddet_reference.json` and
`configs/native_carrada_reference.json`. They record evidence and unresolved
assumptions. They are **source-derived references**, not locally verified
calibrations. `representation:null` allows a complex dtype to identify a field;
real arrays require an explicit `complex|power|amplitude|db` declaration. Here
`db` means power dB, `10*log10(power)`. A different logarithmic convention must
be converted explicitly.

```sh
python -m radsimreal native inspect --dataset raddet --root "$RADDET_ROOT" --profile configs/native_raddet_reference.json --trusted-labels --reference-cartesian --max-frames 4 --out "$RADSIM_RUNS/raddet_inspection"
python -m radsimreal native inspect --dataset carrada --root "$CARRADA_ROOT" --profile configs/native_carrada_reference.json --max-frames 4 --out "$RADSIM_RUNS/carrada_inspection"
```

If CARRADA's annotation JSON is in the other archive, add
`--annotations /local/path/Carrada/annotations_frame_oriented.json` to its
source commands. If its tensors are real, declare the verified convention with
`--representation power`, `--representation amplitude` or `--representation db`.
The command-line declaration overrides the profile's representation. Do not
choose one merely to suppress a dtype error.

`--trusted-labels` enables Python pickle for trusted original RADDet label
files. The reference
RADDet affine comes from its official projection helper and remains marked
unvalidated until an actual overlay confirms it. Native Cartesian rectangles
are retained before the affine, whose common per-axis scaling/reflection and
translation preserve their IoU.

Inspect the generated physical-coordinate overlays. Confirm range direction,
left/right angle, Doppler sign and physical placement against native labels
and release metadata. **CARRADA's annotation angle increment is not proof that
its full FFT tensor is uniformly sampled in physical angle.** The supplied
reference records this unresolved tensor-grid issue; change the profile's
angular coordinate/endpoints when local metadata or a known geometry requires
it. No automatic histogram fit can resolve a coordinate convention.

Save any verified changes in a new local profile JSON. The complete field
contracts and primary sources are in [NATIVE_FORMATS.md](NATIVE_FORMATS.md).

## 2. Prepare scene-disjoint datasets without copying raw tensors

Preparation writes a version-2 manifest, the declared grid/profile, fixed
training-only normalization and cached per-frame annotations. Arrays remain
in the original dataset. The first header and every subsequently loaded array
are validated. The frozen index prevents files added later from entering an
existing split. Targets outside the BEV grid raise an error; they are not
silently cropped for training while retaining different evaluation boxes.

For an initial pilot, choose the disclosed deterministic scene/sequence split:

```sh
python -m radsimreal native prepare --dataset raddet --root "$RADDET_ROOT" --profile configs/native_raddet_reference.json --trusted-labels --reference-cartesian --auto-split --out "$RADSIM_RUNS/raddet_prepared"
python -m radsimreal native prepare --dataset carrada --root "$CARRADA_ROOT" --profile configs/native_carrada_reference.json --auto-split --out "$RADSIM_RUNS/carrada_prepared"
```

Include CARRADA's verified representation and separate annotation path if
required. For a defined study, replace `--auto-split` with `--split-file` or
`--val-sequences` / `--test-sequences`. A sequence split file contains, e.g.:

```json
{"train":["actual_train_sequence"],"val":["actual_val_sequence"],"test":["actual_test_sequence"]}
```

For RADDet use original scene names such as `scene_0`. The reference test
scenes are 9 and 11; validation is an additional explicitly recorded choice.
The integer scene intervals must be intersected with observed original IDs.
Their incomplete available-file manifest does not by itself produce the
reported 8,196 training frames. For CARRADA, the paper's exact sequence/frame
manifest is unavailable; the pilot split is not presented as that manifest.

`label_audit.json` records what was decoded. RADDet preserves native Cartesian
rectangles. CARRADA produces explicitly named Cartesian enclosures of native
radar annotation support; those are not physical car footprints or yaw labels.
Both use physical Cartesian geometry, but their annotation definitions remain
recorded and distinct.

CARRADA's full RAD archive may include boundary/warm-up frames absent from
`annotations_frame_oriented.json`. Preparation excludes these frames from all
supervised train/validation/test splits, records each exclusion and reason,
and reports retained counts. It never turns a missing annotation into an empty
scene. An explicitly present empty frame dictionary remains a genuine negative
example. Missing complete sequences/files or malformed annotations still raise
errors. Training and validation must remain nonempty after filtering.

The frozen `native_records` index retains all raw frames. Separately,
`calibration_record_ids` lists every raw frame in the declared TRAIN scenes or
sequences, including unannotated frames. Such frames can support PSF/noise/gain
measurement without requiring a car label; no held-out sequence is admitted.
Full label checking is the default. `--quick-label-check` is a pilot: it checks
every CARRADA annotation frame key to exclude missing tails, but only the first
12 retained frames receive full box/schema validation and caching. It does not
certify all remaining labels.

## 3. Select and measure the PSF/noise from training frames

```sh
python -m radsimreal native candidates --data "$RADSIM_RUNS/raddet_prepared" --max-frames 16 --out "$RADSIM_RUNS/raddet_candidates"
```

Repeat with `carrada_prepared` for that sensor. The command writes candidate
PNGs, `candidates.json` and `selection_template.json`. A bright, narrow peak
does not establish a stationary isolated pole/corner reflector. A lowest-energy
block can also bias a noise estimate downward. Inspect genuine target identity,
stationarity, interference and reflection-free noise regions before setting the
selection's confirmation flags. Use several repeated captures of the same
stationary target when available.

Selection coordinates are **canonical `(range,Doppler,azimuth)` bin indices**.
Use full string `record_id` values from the prepared manifest, not integer frame
numbers or positions in a sorted file list. Every selected frame must belong
to the prepared training partition. Half-widths define equal odd-sized target
crops; noise bounds are `[start,stop)` intervals. Save the edited selection
outside the raw data tree, e.g. `raddet_verified_selection.json`.

```sh
python -m radsimreal native calibrate --data "$RADSIM_RUNS/raddet_prepared" --selection raddet_verified_selection.json --noise-model empirical --phase-align --out "$RADSIM_RUNS/raddet_calibration"
```

`--phase-align` applies only to complex fields and explicitly rotates each
target capture to its center phase; it does not recover sub-bin motion or
prove phase stability. Omit it for scalar power-domain inputs. By default the
PSF is normalized by its complex center; gain and baseline are retained in
metadata. `--keep-target-gain` preserves the measured response scale instead.

Outputs are `psf.npz`, `noise_bank.npz`, `calibration.json`, `selection.json`,
`calibration_diagnostics.png` and `sensor_config.json`. Metadata includes the
native profile, selected original frame IDs, noise mean/covariance, phase
coherence, target gain, energy reference and confirmation status. Empirical
noise samples preserve a measured marginal, not spatial correlation. Gaussian
mode uses the measured real/imaginary covariance when applicable. The default
measured sensor config enables periodic Doppler processing; range/angle edges
remain finite. See [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md).

An optional relative gain fit uses unlabeled **training** response quantiles:

```sh
python -m radsimreal native gain --data "$RADSIM_RUNS/raddet_prepared" --config "$RADSIM_RUNS/raddet_calibration/sensor_config.json" --real-frames 16 --synthetic-frames 4 --vehicle-assets configs/assets/vehicle_assets.json --out "$RADSIM_RUNS/raddet_calibration/gain_fitted_config.json"
```

This records its quantile ratio and changes the relative scattering gain. It
does not establish absolute RCS or compensate for an incorrect scene/material
distribution. Use the resulting config below only if that declared fit is
part of the intended experiment.

## 4. Generate matching full tensors and native-format exports

```sh
python -m radsimreal native generate --data "$RADSIM_RUNS/raddet_prepared" --config "$RADSIM_RUNS/raddet_calibration/sensor_config.json" --scene-backend mesh --vehicle-assets configs/assets/vehicle_assets.json --export-native --train 16 --val 4 --test 4 --max-samples 10000 --out "$RADSIM_RUNS/raddet_synthetic"
```

Repeat for the CARRADA prepared/calibrated paths. Full physical sampling,
response representation, BEV grid and fixed normalization must agree. Mesh
generation uses geometric surfaces, explicit materials and ray visibility;
an optional `--static-mesh` accepts an OBJ or project mesh NPZ in scene-world
metres. This improves the former simple box-face model; it does not supply the
authors' undisclosed CARLA assets/material tables or validate multipath/weather.

The example above places the included illustrative low-poly sedan template.
Supply your actual vehicle OBJ/RF-material manifest with `--vehicle-assets` to
use measured or otherwise justified assets. Omitting that option uses closed
box extrusions. Template fitting, scene poses/velocities, original object IDs,
source hashes and material declarations are recorded in frame provenance;
see [SCATTERING.md](SCATTERING.md). Use the same asset library for any relative
gain fit and subsequent generation. Optical texture/MTL files do not establish
RF material properties, and the included sedan is not an author CARLA asset.

The generated root contains the training manifest, full canonical tensors,
physical annotations, scene/reflection provenance and copied PSF/noise artifacts
with portable relative paths. `native_export/` adds declared native `R,A,D`
files and native annotation schemas. `native_manifest.json`, `NATIVE_EXPORT.json`
and per-frame `native_metadata` record source, conversion, support policy and
geometry. CARRADA label JSON is streamed at finalization. Its synthetic target
regions are converted through the same native reader as real annotations;
original synthetic physical geometry is retained separately.

Native synthetic IDs are new generated IDs. **Do not apply the original RADDet
published scene table to these IDs.** The generated version-2 manifest is their
split authority. The native export is a format-compatible synthetic dataset,
not a relabeling of original real recordings or a claim of identical annotation
generation.

The artifact contracts used by the code and a continuing cluster agent are:

| Location | Fields and purpose |
|---|---|
| Prepared `manifest.json` | `format_version:2`, `source:native_reference`, `native_root`, `native_profile`, frozen `native_records`, `splits`, `bev_grid`, `normalization`, `label_domain`, `boundary_policy`; split rows may point to cached `prepared_annotation` JSON |
| Prepared `native_profile.json` / `label_audit.json` | Declared transforms plus decoded frame/car counts and missing-yaw counts; source tensors are not copied |
| Calibration `sensor_config.json` | Canonical simulation sampling and paths to `psf.npz` / optional `noise_bank.npz`; `calibration.json` preserves original target/noise selection and gain/phase provenance |
| Generated `config.json` | Sensor paths rewritten to self-contained copies `calibration.npz` / `noise_bank.npz` by `materialize_sensor`; relocating this generated directory does not require the old calibration working path |
| Generated `manifest.json` | Version-2 training split records, target/metric domain, sensor configuration and `native_profile` when exporting; original geometry remains in per-frame annotation provenance |
| `native_export/native_manifest.json` | `format_version:1`, `dataset`, `profile`, `source`, `counts`, `rows`, `label_policy`, `benchmark_equivalence_claimed:false`; each row records `tensor`, `annotation`, `metadata`, split/sequence and physical boxes |
| `native_export/NATIVE_EXPORT.json` | Explicit conversion/profile and synthetic origin; optional views are declared linear-power maxima, not silently claimed to match every original processed-view pipeline |
| Run `run_inputs.json` / `run_report.json` | Recorded arguments/resolved profile/selection hash, completed stages, provenance, metrics and execution status; these determine whether an unchanged run may resume |

`NativeReader.from_manifest` validates indexed paths and original identities
without rediscovering the raw tree. A separately stored CARRADA annotation JSON
is an explicit `annotation_path` override. Prepared datasets are intentionally
references to local raw files, so relocating only the prepared folder requires
updating its declared native root rather than expecting missing data to appear.

A `256*256*64` complex64 tensor is 32 MiB. Ten thousand uncompressed cubes are
about 312.5 GiB per copy; retaining internal tensors and native exports can
roughly double that before compression. Use bounded pilots first, then choose
storage and sample counts deliberately. `--no-tensors` saves images only and
therefore makes full-tensor RADDet training unavailable.

## 5. Validate held-out data and train the three detector families

```sh
python -m radsimreal native validate --data "$RADSIM_RUNS/raddet_prepared" --synthetic "$RADSIM_RUNS/raddet_synthetic" --split test --max-frames 32 --out "$RADSIM_RUNS/raddet_validation"
python -m radsimreal bev-train --model unet --data "$RADSIM_RUNS/raddet_synthetic" --out "$RADSIM_RUNS/raddet_unet" --epochs 1 --max-steps 4 --base-channels 8 --batch-size 2
python -m radsimreal bev-train --model probabilistic --data "$RADSIM_RUNS/raddet_synthetic" --out "$RADSIM_RUNS/raddet_probabilistic" --epochs 1 --max-steps 4 --base-channels 64 --batch-size 2
```

RADDet's source-derived two-stage route is executable:

```sh
python -m radsimreal bev-pretrain --data "$RADSIM_RUNS/raddet_synthetic" --out "$RADSIM_RUNS/raddet_rad_pretrain" --epochs 1 --max-steps 2 --batch-size 2
python -m radsimreal bev-train --model raddet --data "$RADSIM_RUNS/raddet_synthetic" --out "$RADSIM_RUNS/raddet_cartesian" --pretrained-backbone "$RADSIM_RUNS/raddet_rad_pretrain/best.pt" --freeze-backbone --epochs 1 --max-steps 4 --batch-size 2
python -m radsimreal bev-evaluate --checkpoint "$RADSIM_RUNS/raddet_cartesian/best.pt" --data "$RADSIM_RUNS/raddet_prepared" --out "$RADSIM_RUNS/raddet_cartesian/native_test_metrics.json"
python -m radsimreal bev-infer --checkpoint "$RADSIM_RUNS/raddet_cartesian/best.pt" --data "$RADSIM_RUNS/raddet_prepared" --index 0 --out "$RADSIM_RUNS/raddet_cartesian/native_overlay.png"
```

Use `--device cuda` when running in an appropriate GPU allocation. Remove
`--max-steps` on `bev-train` / `bev-pretrain` to run the requested full epochs.
Those standalone commands expect positive explicit step limits. The integrated
`native run` command instead accepts `--max-steps 0` for all requested epochs.

The RADDet architecture is ported from pinned 2021 released source, with
class-car, coordinate, assignment and numerical adaptations disclosed in
[DETECTOR_IMPLEMENTATION.md](DETECTOR_IMPLEMENTATION.md). The U-Net's adapted
heads and training choices are explicit reconstructions. The probabilistic
model implements the published encoder/decoder, anchor and uncertainty
equations, with unresolved/adapted details recorded. Reduced pilot widths do
not constitute the full reference architecture. Public RADDet/CARRADA labels
do not supply genuine per-car yaw for the oriented probabilistic task; the
axis-aligned adaptation masks that supervision. Use `--oriented` only when
genuine orientation targets exist for every object.

For comparisons whose recorded label provenance differs, evaluation requires
`--allow-label-domain-change`. This flag acknowledges a difference; it does
not make the labels equivalent. Native full-tensor models also check sensor
coordinate provenance; cross-sensor transfer needs its explicit supported
acknowledgment and unchanged architecture dimensions. Fixed preprocessing comes
from the checkpoint, not a newly fitted test set.

## Integrated pilot and resume

For a bounded pipeline execution test with automatic **unconfirmed** candidates:

```sh
python -m radsimreal native run --dataset raddet --root "$RADDET_ROOT" --profile configs/native_raddet_reference.json --trusted-labels --reference-cartesian --allow-provisional --vehicle-assets configs/assets/vehicle_assets.json --models unet raddet probabilistic --raddet-pretrain --train 16 --val 4 --test 4 --epochs 1 --max-steps 4 --base-channels 8 --batch-size 2 --out "$RADSIM_RUNS/raddet_provisional_pilot"
```

This can exercise the entire chain without pretending an automatic bright
candidate is a verified calibration reflector. A scientific calibration run
should instead supply `--selection /local/verified_selection.json`. Omitting
both options writes candidate diagnostics and returns a report requesting a
verified selection. Use a new output directory when adding/changing a
profile, split or training budget. Before any calibration product exists,
`--resume-run` permits supplying the first verified `--selection` after the
candidate-inspection stage, preserving the transition and selection hash in
`run_input_transitions.json`. After calibration, a changed selection requires a
new output directory. Other resumed stages require the same recorded inputs;
standalone training also has its own optimizer/checkpoint `--resume` route.

`--train-real-baseline` adds a real-data training comparison with the same
declared partitions. `--fit-gain` enables the optional training-only quantile
fit. `--models` selects completed model implementations; initial CPU pilots can
use just `unet`. On CARRADA, add its verified representation/annotation options
and use a separate run directory.

Inspect `run_report.json`, model histories, native test metrics, calibration
diagnostics, distribution reports and overlays. Keep those artifacts with the
profiles and exact split/selection manifests. Logs intentionally contain local
paths for reproducibility; review them before sharing outside the cluster.

## Continue with an agent on the cluster

The cluster agent can execute every command here without direct access from
this chat. It should read `run_inputs.json`, `run_report.json`, profiles and
selection provenance, correct any local format mismatch, and resume with
unchanged inputs or create a distinct run for a changed experiment. Preserve
the distinction between verified original Cartesian labels, reconstructed
CARRADA support enclosures, synthetic geometry and genuinely measured sensor
statistics. A fixture-backed successful run is an execution result; a run on
actual mounted recordings supplies the missing real-data evidence.
