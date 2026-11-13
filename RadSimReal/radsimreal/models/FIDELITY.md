# Legacy v1 RA-image detector implementation boundary

This document applies only to the retained `radsimreal.models` v1 interface.
The v2 Cartesian stack lives in `radsimreal.bev` and implements U-Net, RADDet
and Probabilistic families, native-coordinate handling and physical IoU.
See [the v2 detector guide](../../docs/DETECTOR_IMPLEMENTATION.md) and
[native HPC workflow](../../docs/HPC_GUIDE.md). Historical choices below do
not describe the capabilities of the complete v2 package.

`CoordinateUNet` is an independently written, executable reconstruction. It is
not an authors' implementation or a recovered checkpoint.

## What the publications establish

[RadSimReal, Section 4](https://arxiv.org/html/2404.18150v1) evaluates three
detector families, including a U-Net augmented with Cartesian input coordinates.
It does not provide the adapted detection heads, layer widths, normalization,
loss weights, training schedule or optimizer settings. Its cited
[Zhang et al. network, Section 4](https://arxiv.org/pdf/1902.05394) uses
complex range-Doppler input with phase normalization and predicts presence and
camera-image coordinates. That task differs from range-azimuth box detection.
Copying its original input and output semantics would therefore not establish
the missing RadSimReal adaptation.

## Explicit reconstruction choices

The executable baseline accepts three channels: fixed-normalized log intensity,
lateral Cartesian position and forward Cartesian position. It has four pooling
levels, transpose-convolution upsampling and skip connections. Its default width
is 16 channels, doubled at successive encoder levels. Each block has two 3x3
convolutions, GroupNorm with one group and ReLU. Odd image dimensions are aligned
to skip tensors by bilinear interpolation. All dimensions must be at least 16.

The three heads predict a class-car center heatmap, two subpixel center offsets
and two normalized box dimensions. Offsets are ordered column then row; box
dimensions are width then height. Heatmap output is a logit; offset and size
outputs use sigmoid. Decoded boxes are normalized image XYXY. These heads support
one center per output pixel. Collisions retain the smaller object and are counted
in training history; increase image resolution to reduce this limitation.

Each center receives a Gaussian heatmap target with standard deviation equal to
one sixth of the box extent, bounded below by one pixel. Focal exponents are 2
and 4. L1 offset loss has weight 1 and L1 normalized size loss weight 5; both are
masked to labeled center pixels. Empty examples contribute negative heatmap loss
and zero box loss.

The chosen optimizer is AdamW, learning rate 0.001, weight decay 0.0001, gradient
norm clipping 10. The training seed controls Python, NumPy and Torch generators.
The best checkpoint minimizes validation loss. Test data never select weights
or fit preprocessing. Fixed dB limits are stored with each checkpoint. Resuming
restores the optimizer and update count, but reseeds shuffle order and proceeds
to the next epoch; it is not bitwise continuation of an interrupted epoch.
The prior best model is retained in the same output directory and copied when
resuming into a new one, provided its history matches the resumed checkpoint.
If only a non-best last checkpoint is supplied and its earlier best weights are
missing, selection covers the resumed segment; the checkpoint explicitly records
this scope and the earlier best loss. Optimizer metadata always reports restored
effective settings, including when they differ from the new command's arguments.

Inference extracts local heatmap maxima, applies a configurable score cutoff and
candidate limit, then greedy IoU suppression. AP ranks detections globally,
matches each ground truth at most once and integrates the monotonic precision
envelope at every recall change. Zero-ground-truth AP is undefined (`null` in
JSON), not a perfect score. The built-in box domain is the range-azimuth image;
Cartesian BEV evaluation requires original compatible labels and a corresponding
detector head; enclosing RA rectangles lose footprint geometry.

## Comparison models absent from the legacy interface

The legacy `radsimreal.models` interface contains only the v1 U-Net. V2 includes
a source-derived RADDet port and a disclosed reconstruction of the Probabilistic
architecture. Their guide separates released source from unreported RadSimReal
adaptations. The separate
[official RADDet repository](https://github.com/ZhangAoCanada/RADDet) provides
TensorFlow code and pretrained-checkpoint links; its published configuration is
not evidence of the exact RadSimReal experiment configuration. Validating the
v2 implementations with matched real-data preprocessing and verified experiment
conventions remains required for reproducing the complete experimental matrix.

No detector settings above require radar hardware design details, but transferring
to a real sensor requires calibrated simulation, compatible image units and grid,
validated box conventions, and training/validation scenes representative of the
intended data distribution. A short smoke test establishes execution, not accuracy.
