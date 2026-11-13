# Scientific validation protocol for the native simulators

The central operator, dataset adapters and end-to-end software can be verified
independently. Sensor fidelity and object-detection generalization require
additional measurements on the actual recordings. This protocol specifies
those measurements and the evidence that should accompany a result.

## A. Freeze the input and coordinate contract

Archive the native profile, installed software versions, original frame IDs,
dataset release metadata, split manifest and signal-domain declaration. Record
tensor shape/dtype and canonical transforms. Resolve range orientation,
angular sampling, Doppler FFT shift/sign and physical origins using metadata
and overlays with known geometry. Equal tensor dimensions do not establish
equal sensor processing.

Internal tensors use range, Doppler, azimuth; positive velocity means receding.
Native RADDet/CARRADA arrays use range, azimuth, Doppler. CARRADA's annotation
velocity is positive approaching. An even-length sign-reversed Doppler axis
has an asymmetric endpoint; preserve its actual origin rather than shifting
the sign by a bin. The angular grid must describe the tensor processing, not
merely the annotation's displayed angle increment.

Retain native Cartesian RADDet rectangles before the physical affine. A shared
invertible per-axis scale/reflection/translation preserves axis-aligned IoU.
Normalization within a Cartesian grid is likewise harmless. An enclosing
polar rectangle converted to Cartesian space does not preserve a footprint
or its IoU. No new ground-truth box may be silently substituted for the native
one. A declared BEV grid must contain complete targets; otherwise expand the
grid or define an explicit different protocol before training.

## B. Separate calibration, validation and test evidence

Assign whole original scenes/sequences. Use original RADDet IDs and its
published test scenes 9/11. Intersect the scene table with observed files and
report missing/unassigned IDs. Keep the added validation scenes explicit.
CARRADA's exact RadSimReal sequence manifest is not released; record the
chosen held-out sequences and distinguish them from the author's study.

Fit normalization, relative gain, scene distributions, materials and anchors
using training/calibration data only. `calibrate_native` requires the declared
training IDs and rejects ambiguous integer selection identifiers. Do not tune
the simulator repeatedly against the final test distribution or AP. Use a
separate validation partition for engineering/model selection; retain test
as final evidence. Report seeds, sample counts and any departures from the
reference protocol.

## C. Measure and check the PSF

Select a stationary, isolated narrow target, preferably a suitable corner
reflector. A bright local maximum does not establish a point target. Archive
target identity evidence, selected frame IDs, canonical center/crop indices,
the relevant scenes and why interference is negligible. Repeated captures
should represent the same stationary geometry. Confirm complex phase stability
or explicitly record center-phase alignment; it does not correct unmodeled
motion, sub-bin migration or different target geometry.

Estimate and subtract a declared background baseline before target averaging.
Keep the gain convention explicit: normalization by the complex center changes
the units of reflector coefficients. Retain the original center gain and do
not interpret arbitrary simulation weights as absolute RCS. The implementation
records those quantities and whether target assumptions were confirmed.

Inspect range, Doppler and angular PSF cuts, the multi-axis support and
sidelobes. The implemented 99% energy threshold refers to the squared L2 norm
of the **selected candidate support**, not unseen portions of the full sensor
response. Increase the candidate crop and verify convergence. Compare removed
response with the unaveraged sensor noise on a consistent field/power scale.
Test spatial PSF invariance at multiple ranges/azimuths when data permit. A
single fixed kernel is a model assumption when that evidence is unavailable.

A power-only target measurement cannot recreate coherent phase. The real
intensity/power surrogate and the complex-field simulator are distinct declared
representations. Record which is supported by the input and which assumptions
are added. Do not use a logged display image as a linear PSF.

## D. Measure noise independently from clutter

Select reflection-free native tensor regions independently of target peaks.
Record why they contain no object/clutter returns. Automatically selecting
minimum-energy blocks can underestimate variance; candidate selection is not
a validated measurement. Include several representative times/ranges where
appropriate and inspect nonstationarity.

For complex fields, report total `E|n-E[n]|²`, real/imaginary means and their
2x2 covariance. Examine skew, kurtosis, marginal distributions and neighboring
cell dependence. The supplied KS diagnostics use a fitted mean/std and possibly
correlated cells; their p-values are diagnostic, not a valid proof of Gaussian
noise. Empirical IID sampling retains the observed marginal and quadrature
relationship; it does not reproduce spatial/temporal covariance. Gaussian
sampling uses the measured quadrature covariance. Keep that limitation visible
when comparing real and synthetic backgrounds.

Clutter is a scene/scattering contribution, not synonymous with thermal noise.
Do not absorb strong background surfaces into a purported reflection-free
noise bank. Materials, mesh visibility, roughness, incidence, range attenuation,
multipath and weather need separate evidence or explicit approximation labels.

## E. Check tensor and native-export equivalence

For deterministic reflector fixtures verify shifted-kernel/FFT agreement,
sub-bin deposition, finite boundaries and optional periodic Doppler tails.
Round-trip canonical tensors through native layout/response conversion. Complex
export requires fields; amplitude/power/dB export has an explicitly recorded
conversion and storage precision. Real signed-surrogate noise cells require
nonnegative clipping for a native power representation; that transformation
must remain documented.

Compare complete sensor coordinates—counts, origins, spacings and angular
coordinate—not only shapes. Preserve periodic Doppler policy for both simulated
returns and exported RF-support labels. A visible aliased return must not lose
its object label simply because its unaliased physical speed lies outside one
Doppler interval. Use actual output metadata to verify the selected policy.

Generated Cartesian labels originate in original scene geometry. RADDet native
Cartesian labels are an invertible declared affine of those enclosures. RF
RAD/CARRADA labels are reconstructed from sampled return support expanded by
the truncated PSF; that is not the author's unpublished label generator.
CARRADA enclosures remain radar annotation regions rather than true car
footprints. No oriented yaw may be inferred from a 2D radar box. Native export
schema compatibility and matched annotation semantics are separate checks.

## F. Compare real and synthetic data fairly

The supplied `native validate` workflow compares untouched held-out native
and synthetic splits under identical sensor coordinates and common physical
display axes. It reports response quantiles, Wasserstein/KS distribution
diagnostics, average range profiles and box dimensions. Comparisons are
**unpaired** unless an actual shared 3D scene is supplied. Do not present
different scenes as pixelwise ground-truth pairs.

Histograms can reveal gross gain/noise mismatches. Matching them does not prove
the correct PSF, material response, object scattering or generalization. Do
not use a globally fitted gain to conceal an incorrect geometry/scene model.
Retain a native-only baseline, controlled ablations and diagnostic failure
examples. The program's `sensor_realism_established:false` status is deliberate;
scientific interpretation requires the accumulated evidence, not a scalar
software pass/fail threshold.

## G. Evaluate the downstream task in the declared physical domain

Train and evaluate on the same target definition and coordinate convention.
The v2 models predict physical Cartesian boxes. Axis-aligned evaluation uses
Cartesian area IoU; rotated evaluation uses physical oriented polygons when
genuine yaw targets exist. Retain original target extents without silent
training/evaluation boundary disagreement. The default `global_pr` AP uses
globally score-ranked, one-to-one matching and the all-points monotonic PR
envelope. Explicit `frame_mean` aggregation instead follows the released
RADDet reference's per-frame averaging choice. The selected aggregation and
matching policy are recorded; neither is silently identified as RadSimReal's
unreleased evaluator. Retain these choices when comparing reported AP.

Report architecture, anchors, losses, optimization, training length, validation
selection, normalization, sensor profile and source provenance. The pinned
RADDet port supplies its 3D pretraining and frozen Cartesian stage; a direct
end-to-end Cartesian recipe is a distinct choice. U-Net adaptations and the
probabilistic model's unresolved assignment/input settings remain disclosed.
Uncertainty outputs also need empirical calibration against held-out errors;
finite predicted variances are not evidence of calibrated uncertainty.

Synthetic-vs-native and cross-dataset tests must record label-domain differences.
For CARRADA, a converted radar-support enclosure is not automatically the
physical vehicle annotation definition used in another dataset. Explicitly
acknowledging a domain change allows the experiment to run; it does not make
the resulting AP directly equivalent to a differently defined benchmark.

## Evidence to retain

- Native release/version, source file index, profile and coordinate overlays.
- Scene/sequence splits, calibration selection and preprocessing fit provenance.
- PSF/noise products, crop/gain/phase choices and calibration diagnostics.
- Scene/material configuration, mesh inputs and native-export conversion policy.
- Training/evaluation commands, seeds, histories, checkpoints and resource usage.
- Native-only and synthetic-trained metrics on the same untouched test targets.
- Failed examples, ablations, remaining unsupported physics and unavailable
  author details.

The software can execute the available reconstruction and the complete
data-dependent experiment on the HPC host. Exact numerical reproduction of
the original paper additionally depends on author assets/settings, original
derived annotations/manifests and sufficient training. Record any remaining
differences as experimental limits rather than silently replacing them.
