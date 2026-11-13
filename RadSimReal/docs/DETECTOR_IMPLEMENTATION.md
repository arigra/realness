# Cartesian detection implementation and fidelity

The v2 `radsimreal.bev` package removes the previous RA-image-coordinate output
and evaluation shortcut. Its boxes are Cartesian ground-plane objects in a
declared `BEVGrid`. A linear change from Cartesian grid indices to metres
preserves axis-aligned IoU. Polar-to-Cartesian re-enclosure does not generally
preserve IoU or an original physical footprint.

No measured radar dataset was accessed on this machine. Native readers and
cluster runs are responsible for checking the actual label conventions. A
working model implementation does not imply reproduced RadSimReal accuracy.

## Coordinates, inputs and labels

- Physical x is lateral right and y is forward. A Cartesian box is
  `[xmin,ymin,xmax,ymax]` in metres. Training normalizes it by the fixed Cartesian
  grid limits; this affine normalization is invertible.
- Oriented boxes are `[cx,cy,width,height,theta]` in metres and radians, with
  theta counterclockwise rotating the width axis from +x. Procedural
  `simulation.Vehicle` uses the opposite rotation sign, so theta is `-yaw`.
- `boxes` are original Cartesian targets normalized by the declared grid.
  Labels outside that grid raise an error requiring a larger grid; they are not
  silently cropped or filtered. `boxes_physical` retains the same original
  metre-coordinate targets for evaluation. `oriented_boxes` is a genuine
  yaw-bearing target aligned one-to-one with training boxes;
  `oriented_boxes_physical` retains the original evaluation labels. These fields
  must never be fabricated from an RA rectangle.
- `PolarToBEV` samples scalar radar intensity at Cartesian pixel centers using
  the actual range and angular axis vectors. It supports nonuniform increasing
  axes, masks the rear hemisphere and coordinates outside the available radar
  domain, and is differentiable with respect to image values. It interpolates
  scalar intensity; it is not a conservation-of-RCS operation.
- A sample's `image` is `[3,Hbev,Wbev]`: chosen fixed-normalized log intensity
  and Cartesian x/y coordinate channels. RADDet additionally consumes `radar`
  `[D,R,A]`, already converted to the declared real normalized log-power domain.
  No complex-field phase is silently cast to real by the projector.

## Coordinate U-Net

`CartesianUNet` uses the existing four-level U-Net implementation on Cartesian
input. Its heatmap, subpixel offsets and box-size heads predict directly in
the Cartesian grid. Training uses Gaussian center focal loss and masked L1
offset/size loss. The model architecture, heads, normalization and optimizer
are disclosed reconstructions: RadSimReal Section4 does not release its adapted
U-Net implementation or all training settings.

## RADDet Cartesian detector

`RADDetBEV` implements the released repository's RadarResNet and Cartesian YOLO
head in PyTorch, rather than assigning the RADDet name to the U-Net head.

Primary source files checked on 2026-11-10. File history was inspected and the
reference was pinned to commit`e18b4cfaf3823ffd9983b48591d2d30a1082f56d`
(2021-05-05). The backbone's latest file change was this2021 revision, predating
RadSimReal2024. Source URLs and SHA256 hashes are recorded in
[DETECTOR_SOURCE_LOCK.json](DETECTOR_SOURCE_LOCK.json). RADDet's original paper
TableII independently confirms the2,4,8,16 repeat counts.

- [RadarResNet](https://github.com/ZhangAoCanada/RADDet/blob/main/model/backbone_radarResNet.py)
- [Cartesian model and loss](https://github.com/ZhangAoCanada/RADDet/blob/main/model/model_cart.py)
- [Layers](https://github.com/ZhangAoCanada/RADDet/blob/main/model/layers.py)
- [Released Cartesian anchors](https://github.com/ZhangAoCanada/RADDet/blob/main/anchors_cartboxes.txt)
- [Repository and its backbone-version notice](https://github.com/ZhangAoCanada/RADDet)
- [Paper, SectionIV-C](https://arxiv.org/html/2105.00363v1)

The implemented backbone has four stages containing 2,4,8,16 residual units.
Each unit uses 3x3,3x3,1x1 convolution/BN/ReLU layers and a residual sum.
The final unit in stages3/4 doubles channels. Each stage ends in 2x2 max pooling.
Doppler bins are input channels, matching the released 2D-convolution code.

The Cartesian head applies two shared-per-channel spatial MLP layers, changing
each `Fr*Fa` feature map to `Fr*(2*Fa)`, then a residual unit, a channel-doubling
3x3 convolution and six Cartesian anchor predictions. It predicts x/y offsets,
log width/height, objectness and class-car logits. Released anchor pairs are
reordered from forward/lateral axes into this package's lateral/forward axes,
and normalized to the released 256x512 Cartesian extent. They should be fitted
again on training labels for a materially different domain/grid.

The Cartesian YOLO loss follows the released square-root-size regression,
negative-anchor focal weighting0.01, overlap-ignore threshold0.3 and class BCE.
Box regression operates in Cartesian grid-pixel units with coefficient0.1.
Explicit adaptations are single-class output, declared physical grid/label
mapping, deterministic best-size anchor assignment, and numerical stabilization.
The source clamps log-size below at `1e-12` and above at `1e12`; this can prevent
shrinking anchors and overflow exponentiation. The default port uses a disclosed
signed[-12,12] clamp. `upstream_size_clamp=True` preserves the literal released
decoder clamp for controlled source-parity experiments; it also preserves its
possible exponential overflow. The default therefore does not claim bitwise output/training parity with
the TensorFlow implementation. TensorFlow checkpoint conversion is not supplied.

The **Cartesian RADDet detector** is used for the2D object task. The separate
`RADDetRAD` model in`bev/rad_pretraining.py` also implements the original3D RAD
YOLO head, with six released3D anchors, R/A/D center-size decoding,3D overlap
ignore rule, focal objectness and source box/class losses. Its targets are
normalized range/azimuth/Doppler coordinates, not physical3D object geometry.

RADDet's published training protocol pretrains the backbone with this3D task,
then freezes it for Cartesian-head training. Both stages are executable:

```sh
python -m radsimreal.bev.pretrain --data data/native_raddet --out runs/rad_pretrain --epochs 1 --max-steps 2
```

Then call`train_bev(...,model_name='raddet',pretrained_backbone='runs/rad_pretrain/best.pt',freeze_backbone=True)`
or the corresponding integrated CLI flags. The3D stage requires genuine
`boxes_rad` annotations in exactly the same canonical R/A/D convention as its
input tensor. It fails when those labels are absent; it does not manufacture
them from CARRADA's two-dimensional projections. Its Adam1e-4 reference schedule
decays by0.96 each10k updates after60k warm-up updates, following RADDet Sec.V.

The transfer checks input shape, normalization and native tensor coordinate/sign/
representation metadata. Freezing preserves both trainable backbone parameters
and BN running moments. Resume preserves this freeze state. End-to-end Cartesian
training is also supported as a separately declared recipe. The upstream
repository's generic notice that backbones may vary does not override the
dated-source and paper-table evidence above, but neither source establishes
which exact variant RadSimReal used in its unreleased experiment implementation.

The3D pretraining stage also accepts `--resume runs/rad_pretrain/last.pt` and
restores optimizer, scheduler, history and validated historical-best weights.
`--epochs` requests additional epochs and `--max-steps` remains a total update
bound. Actual sensor sampling, normalization and split-ID separation are checked
before continuing. Resume begins at the next saved epoch with reseeded shuffle;
it does not claim bitwise recovery of an interrupted minibatch. Complete state
files are written to a temporary path and atomically replace the destination,
so a failed write does not destroy the previous checkpoint.

The source-derived topology is attributed under the upstream MIT license; see
the notice below. No upstream pretrained weights or datasets are redistributed.

## Dong probabilistic detector

`ProbabilisticBEV` implements the architecture and equations described in
[Dong et al.,2020, Section4 and Fig.5](https://arxiv.org/html/2004.05310v2):

- ResNet18-style five-stage encoder: initial7x7 convolution, then four stages
  with two residual units each; stride2 convolution downsamples.
- Four bilinear-upsample decoder stages with two3x3 convolutions and lateral
  1x1 skip connections, followed by two3x3 neck convolutions.
- Three anchors per output cell, each producing objectness, six localization
  outputs `(dx,dy,logw,logh,cos(delta),sin(delta))`, and six log variances.
- R-CNN-style center/size offsets. Smooth-L1 regression with the paper's
  heteroscedastic equation `exp(-log_variance)*smoothL1 + log_variance` and
  localization weight100. Binary focal objectness uses alpha0.25,gamma2.

The paper names ResNet18 but calls its units “bottleneck” blocks without enough
layer detail to uniquely resolve that terminology. This implementation uses
the standard two3x3-convolution ResNet18 basic unit. Decoder widths, additive
versus concatenated skip fusion and complete training assignment are not fully
specified; this implementation uses configurable widths, projected concatenated
skips and best physical rotated-IoU matching at the GT center cell, and records collisions.
Those are explicit reconstruction choices.

The original15x42-pixel anchors depend on the authors' own BEV projection. This
package defaults to a disclosed2m x4.5m size in the declared physical grid, and
retains the published three30/115/136-degree anchor orientations. Supply/fix
anchors from **training** labels for the actual dataset. Applying original
pixel sizes to a different physical grid would silently change geometry.

`oriented=True` requires genuine yaw labels for every object; incomplete yaw
labels raise an error. The default axis-aligned adaptation masks orientation
loss and ignores untrained angle outputs during axis-aligned decoding. CARRADA
radar-view box labels are not automatically vehicle yaw labels. “Instance-
oriented” in a CARRADA filename describes JSON grouping, not rotated boxes.
Log variance is bounded to[-8,8] for numerical stability. No assertion is made
that all RadSimReal adaptations of the Dong model are recovered.

## Training, inference and evaluation

`train_bev(train_dataset,val_dataset,output_dir,model_name=...)` trains each head
with its own loss. Default reference recipes are AdamW1e-3 for the U-Net,
Adam1e-4 for Cartesian RADDet and Adam5e-4 with100k-step cosine schedule for
the probabilistic model. Configurable gradient clipping10 is a stabilization
choice. These are disclosed recipes, not a claim of recovered RadSimReal settings.
Models and optimizer/scheduler state, geometry, normalization, native data
provenance, label/metric domain and history are saved. Restricted checkpoint
loading uses `weights_only=True`. Resume preserves a validated prior best model
where available; otherwise selection is explicitly limited to the resumed segment.
Interrupted minibatch order is not restored bit-for-bit.

Execution checks cover all three model families' forward/backward, optimizer
updates, validation, restricted checkpoint reload, Cartesian evaluation and
resume. A 3D-pretrain → frozen-Cartesian → resume regression verifies unchanged
backbone parameters and running moments. Additional executed forward passes used RADDet's full
native `(D,R,A)=(64,256,256)` input and the probabilistic encoder's full64-channel
base width. These are execution checks on synthetic fixtures, not real-data
accuracy evidence. The full RADDet Cartesian output shape was`(1,16,32,6,6)`;
the full-width probabilistic model output`(1,64,128,3,13)` for a128x256 BEV input.
Tests also check known 3D grid centers/sizes/objectness scores, near-zero source
loss on an exact confident prediction, known-volume 3D IoU, native signed-Doppler
target conversion, independently known physical variance transforms,
undefined-yaw handling, strict native-sensor transfer guards and ignored
nuisance noise/path changes. Checkpoint interruption/resume preserves optimizer,
epoch/step history and historical best weights; atomic-write failure preserves
the previous valid checkpoint. These fixture checks do not establish performance
on measured RADDet or CARRADA data.

`detect_bev` returns physical metre-coordinate boxes without clipping predicted
extents to the observation grid, as in the released RADDet evaluator. RADDet
ranking and score cutoffs use sigmoid objectness alone, matching the reference
`yoloheadToPredictions2D`/`yoloheadToPredictions` helpers; class logits retain
their separate training loss. Axis-aligned suppression uses Cartesian area IoU.
Oriented suppression uses exact convex polygon intersection of physical rotated
rectangles. Evaluation preserves original physical targets. Different annotation domains require
explicit acknowledgment before cross-domain evaluation. A CARRADA Cartesian
enclosure of native radar annotations remains a distinct label definition from
a true vehicle ground footprint, even though both use Cartesian coordinates.

The default v2 AP protocol uses `--matching-policy best_overall`: each prediction
chooses the maximum-IoU GT over all GT in its frame, then counts as a duplicate
false positive if that GT is already matched. This reproduces the released
RADDet matching rule. `best_unmatched` instead searches only unmatched GT and
is available as an explicitly recorded alternative; the legacy v1 RA evaluator
retains that old default. For example, with two overlapping GT and two identical
predictions both closest to the first GT, `best_overall` can give AP0.5 while
`best_unmatched` gives AP1.0 at the same IoU cutoff. Stable input order resolves
equal scores in this port; exact TensorFlow tie-order parity is not claimed.

`--ap-aggregation global_pr` (default) ranks predictions over the whole evaluation
set and integrates the monotonic precision envelope at every recall change. It
includes false detections in GT-empty frames; AP is undefined if the entire set
has no GT. `--ap-aggregation frame_mean` implements the released RADDet per-class
aggregation: calculate AP separately in each GT-positive car frame, then average.
GT-empty car frames, including their false detections, are excluded; if no frame
contains car GT, class AP is zero. Thus one perfectly detected positive frame
plus a higher-scored false detection in an empty frame gives global AP0.5 and
frame-mean AP1.0. The output records the choice, excluded-frame/prediction counts
and zero-GT rule. It provides no single PR curve for a mean of per-frame AP.

Reference rules were checked against pinned 2021 [`metrics/mAP.py`](https://github.com/ZhangAoCanada/RADDet/blob/e18b4cfaf3823ffd9983b48591d2d30a1082f56d/metrics/mAP.py),
[`evaluate.py`](https://github.com/ZhangAoCanada/RADDet/blob/e18b4cfaf3823ffd9983b48591d2d30a1082f56d/evaluate.py) and
[`util/helper.py`](https://github.com/ZhangAoCanada/RADDet/blob/e18b4cfaf3823ffd9983b48591d2d30a1082f56d/util/helper.py),
whose hashes are recorded in `DETECTOR_SOURCE_LOCK.json`. Exact RadSimReal
evaluator parity remains unverified because its experiment-specific evaluation
code is unreleased. Select the protocol before inspecting evaluation outcomes;
different definitions do not establish reproduction of published scores.

Probabilistic inference also retains its six predicted localization variances,
their offset/log-size/cos/sin parameter names, and a supervision mask. It exposes
the decoded physical `(cx,cy,width,height)` point estimate and
the corresponding diagonal variances in square metres under an explicitly
first-order delta-method propagation. Center variances multiply the offset
variances by squared physical anchor dimensions; size variances multiply the
log-size variances by squared decoded dimensions. Yaw variance uses the atan2
Jacobian and predicted sine/cosine variances, assuming independent regression
parameters. Near-zero sine/cosine norm gives undefined yaw uncertainty (`null`
plus an invalid flag), and axis-aligned training marks yaw uncertainty as
unsupervised. This propagation does not assert a calibrated Gaussian posterior,
does not propagate prediction clipping, and is not empirical real-data
uncertainty calibration. Numerical variance bounds are retained in the encoding.

Evaluation compares the checkpoint's declared radar sampling with the actual
dataset configuration: R/D/A counts, range origin/spacing, velocity origin/
spacing, angular coordinate and angular endpoints. Noise, material gain and
artifact paths do not affect this coordinate comparison. BEV-input models can
use their explicit physical reprojection across different polar samplings, but
the changes are reported. A native-tensor RADDet model rejects different or
missing coordinate provenance unless `allow_sensor_change=True` explicitly
acknowledges transfer; its fixed D/R/A architecture shape must still match.
Resume and train/validation selection apply the same native-input safeguard.
Matching declared sampling does not prove matching sensor PSF or distribution.

## Upstream RADDet license notice

MIT License

Copyright (c) 2021 Aoooooooo~

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
