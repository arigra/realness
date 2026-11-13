# Reproduction status and scientific assumptions — v0.2

The implementation is executable, including native RADDet/CARRADA interfaces, all three detector families and Cartesian evaluation. It is an independent reconstruction of the published method. Actual sensor calibration, training on the authors' scene distribution and reproduction of real-data results are **not established** by the fixture tests shipped here.

The previous version used RA-image rectangles and omitted two models. Those engineering gaps are fixed in v2. Exact published-result parity still depends on resources/settings the paper and repository do not supply, and on experiments against recordings available only on the user's HPC.

Primary reference: [RadSimReal paper and supplementary appendices](https://arxiv.org/html/2404.18150v1). Source evidence and hashes: [SOURCE_AUDIT.md](docs/SOURCE_AUDIT.md), [DETECTOR_SOURCE_LOCK.json](docs/DETECTOR_SOURCE_LOCK.json).

## Mathematical contract

Appendix B.2 models weighted delayed replicas, introduces matched filtering, and expresses output as shifted PSFs (Eq. 4) or convolution (Eq. 5):

```text
rho(t) = sum_i alpha_i delta(t - t_i)
y(t)   = sum_i alpha_i h(t - t_i) = (h * rho)(t)
```

Canonical axes are **range, Doppler, azimuth**. Sparse addition and an independent FFT reference agree for the same sampled reflector field, kernel origin and boundaries. This establishes the discrete operator's identity, not the adequacy of the reflector field or sensor kernel.

Sub-bin placement uses trilinear splatting. Range and angle have finite support. Optional Doppler periodicity represents aliasing on a shifted FFT grid; kernels and labels share that policy. A seam-crossing RADDet linear Doppler box conservatively spans the whole Doppler dimension, with this loss of tightness recorded.

Complex mode sums fields and complex noise, then computes power and Doppler maximum. Scalar intensity mode is an explicit incoherent surrogate with nonnegative reflector weights, optional signed measured PSF tails and nonnegative final power. A power-only response cannot determine phase or interference. The publication leaves field/power conventions incomplete; no implicit equivalence is assumed.

## Component fidelity and calibration

| Component | Publication anchor | Implemented status and remaining assumptions |
|---|---|---|
| Geometry | Sec. 3, Fig. 2(a) | Triangle vehicles/OBJ assets, poses/velocities, static OBJ/NPZ and ground. A read-only CARLA bridge uses actor boxes. Author routes/assets/distributions are absent. |
| RF reflectors | Sec. 3 | Area-weighted triangle sampling, BVH ray/triangle visibility, material/incidence terms. Single-bounce approximation, without full electromagnetic propagation, multipath or diffraction. |
| Reflectivity | Sec. 3 | TE/TM lossy dielectric Fresnel and roughness diffuse/specular terms; relative r^-4 power or parameterized radar-equation scaling. Patch RCS, materials and gain require calibration. |
| PSF | Sec. 3; Appendix A | Analytic demo kernels and streaming native target measurement. Verify target identity/isolation/stationarity, phase coherence and sensor shift invariance locally. |
| Support crop | Appendix A | Energy fraction of a finite selected candidate with normalization metadata. Unobserved sidelobes are outside this guarantee; compare discarded response with unaveraged noise. |
| Noise | Fig. 2(c); Appendix A | Measured mean/covariance Gaussian or empirical IID samples and diagnostics. Spatial/temporal correlations are not preserved by IID sampling. |
| Clutter | Environment simulation | Static geometric returns, separate from thermal noise. Realism depends on surface/material coverage. |
| Processing | Sec. 3; Appendix B | Full 3D response, explicit domain conversion and Doppler maximum. PSF is the receiver interface, not a calibrated chirp/ADC simulator. |
| Native data | Sec. 4 | Axis/domain profiles, original IDs, lazy tensors, cached labels, native-layout synthetic export. Release-specific metadata/coordinates need verification. |
| Detectors | Sec. 4, Table 1 | Three distinct implementations and losses, plus RADDet 3D pretraining/frozen Cartesian stage. Unspecified heads, anchors and recipes have disclosed adaptations. |
| Evaluation | Sec. 4 | Original Cartesian targets, exact rotated intersections for genuine yaw, two explicit AP aggregation rules. RadSimReal's experiment-specific evaluator is unreleased. |

The low-poly sedan is original illustrative geometry. Its metal/glass parameters are uncalibrated examples. An OBJ improves geometry specificity without calibrating RF behavior. The live CARLA bridge was not exercised against a server here.

## Coordinates and annotation definitions

Physical x points right, y forward, z up. Axis-aligned targets are `[xmin,ymin,xmax,ymax]` in metres. Training uses invertible Cartesian affine normalization; evaluation retains original physical boxes without silent clipping. Oriented boxes are `[cx,cy,width,length,theta]`, theta counterclockwise from the +x width axis. Genuine per-object yaw is mandatory for oriented training.

V1 RA-image boxes had different geometry. Normalization alone preserves IoU; a polar rectangle is not a Cartesian footprint. Re-enclosing it cannot recover lost shape/yaw. V2 predicts Cartesian boxes directly and preserves native Cartesian labels.

RADDet's source-helper-derived Cartesian affine preserves rectangle IoU under common per-axis scaling/reflection/translation. Its physical origin/direction still needs real overlays. CARRADA labels describe radar support, not oriented physical car footprints. Their Cartesian enclosures are named accordingly. Synthetic CARRADA targets pass through the same native support parser; original vehicle geometry is saved separately. The filename `annotations_frame_oriented.json` does not provide vehicle yaw.

Profiles declare range reversal, angle coordinate/order, Doppler origin/sign and representation. CARRADA's annotation angular increment does not establish its full FFT tensor's angular sampling; the reference profile records that uncertainty. Native export inverts declared reader transforms, not an unverified hidden convention.

## Calibration and separation

Use repeated stationary isolated-target captures and reflection-free noise blocks from TRAIN scenes/sequences. Selection stores full original IDs, centers, crop sizes, noise intervals and confirmation flags. Automatic bright-peak/low-energy proposals can select extended objects or bias noise downward. Provisional execution never certifies them as verified measurements.

Products record phase alignment, target gain/baseline, energy reference, noise moments/correlations, hashes and profile. Normalization and optional relative scattering-gain fitting use training data only. Gain fitting is an empirical quantile match, not absolute RCS calibration or proof of scene realism. Held-out data are for reporting, not automatic fitting.

The frozen native index prevents added files from entering existing splits. Missing CARRADA annotation frames are explicitly excluded, never treated as empty scenes. Unannotated TRAIN-sequence frames can still support calibration. Malformed/missing annotation sources raise errors.

## Models and evaluator parity

The coordinate U-Net uses Cartesian intensity/x/y channels and center/offset/size heads. Its adaptations and AdamW/focal/L1 recipe are reconstructed.

RADDet ports the pinned released RadarResNet (2/4/8/16 repeats), channel-wise Cartesian spatial MLP and YOLO head, including objectness-only scoring. Genuine 3D RAD-anchor pretraining and a frozen-backbone Cartesian stage are executable. Single-class handling, assignment, transforms and signed bounded size decoding are documented adaptations. Literal upstream size-clamp behavior is optional for parity experiments. No TensorFlow checkpoint conversion or bitwise training parity is asserted.

The Dong probabilistic model implements a ResNet18-style encoder, skip decoder, oriented anchors, heteroscedastic smooth-L1 loss and six localization variances. Decoder/skip choices, physical anchor sizes and assignment are disclosed where details are incomplete. Native axis-aligned adaptation masks unsupplied yaw; oriented mode requires genuine yaw. Inference exposes encoded and delta-method physical variances with supervision/undefined-yaw masks.

AP defaults to global monotonic PR-envelope area with source-style best-overall matching. Optional `frame_mean` follows released RADDet per-car-positive-frame averaging, excluding car-empty frames and their false positives. These rules can give different scores. Reports record matching, aggregation, cutoffs, NMS, label domain and sensor comparison. Exact RadSimReal evaluator parity remains unverified.

## Published experiment dependencies

RADDet original scene intervals and test scenes 9/11 are implemented. Table 4 covers 10,284 possible mapped IDs versus 10,158 reported actual frames. Test scenes map to exactly 1,962 IDs; 126 other missing files cannot be identified from intervals alone. Intersect with observed original filenames; do not manufacture frames or use renamed sorted positions.

The paper reports CARRADA 2,208/276 train/test and CRUW 9,623/2,226. Exact CARRADA manifests and derived CRUW CFAR boxes/tuning are unavailable. The deterministic sequence pilot is not the author split. This delivery targets RADDet/CARRADA; no native CRUW benchmark is claimed.

The authors' 10,000-image synthetic distribution, measured sensor products, RF assets/material settings, full detector adaptations, seeds and complete recipes were not supplied by the repository audited here. These missing resources remain distinct from fixed engineering omissions.

## What delivery execution establishes

[TEST_REPORT.md](TEST_REPORT.md) records numerical tests, native-format fixture pipelines, all model families, resume and actual synthetic training. No real recordings, real point-target calibration, held-out real AP, GPU benchmark, FID, weather robustness or published speedup was measured here.

The [HPC guide](docs/HPC_GUIDE.md) and [agent prompt](docs/HPC_AGENT_PROMPT.txt) carry the project to the machine holding the recordings. Verify profiles/selections, run bounded pilots, correct release-specific mismatches and retain evidence before full training. Published scores require compatible data, labels and evaluation; identical results cannot be promised.

