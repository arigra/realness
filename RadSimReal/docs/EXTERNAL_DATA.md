# Supplying calibrated sensors and real datasets

The basic demo runs without these resources. Reproducing the published study requires external data and information that are not bundled. V2 supplies native RADDet/CARRADA readers, calibration, synthetic native exports and Cartesian model/evaluation workflows; follow [HPC_GUIDE.md](HPC_GUIDE.md) and [NATIVE_FORMATS.md](NATIVE_FORMATS.md) for those interfaces. The [data-format specification](DATA_FORMAT.md) below primarily documents the retained legacy v1 generic-array schema. Do not infer input semantics from an array shape alone.

## Canonical axes and annotations

Store a linear, nonnegative radar image with axes `(range_row, azimuth_column)`. Optional full cubes use `(range, Doppler, azimuth)`. Preserve original file/frame IDs, source sequence, sensor identity and split provenance. Record whether samples represent power, amplitude or another quantity, and whether source values were logarithmic. Convert dB only when its definition is known: `10**(dB/10)` is appropriate for a power ratio; amplitude dB uses a different relation. Unknown sensor gain or normalization cannot be reconstructed by guessing this conversion.

The legacy v1 detector's boxes are normalized RA `xyxy`:

```text
[azimuth_left / width, range_top / height,
 azimuth_right / width, range_bottom / height]
```

Coordinates describe image edges. Pixel-center index `(row, column)` corresponds to normalized coordinates `((column+0.5)/width, (row+0.5)/height)`. All box coordinates are clipped to `[0,1]` after a documented transformation, with positive width/height and car labels as specified by the schema. Physical vehicle extents, radar reflection extents and a rectangular enclosure of a polar projection are distinct annotation choices. Use the one required by the target experiment and retain its provenance.

**V1 RA-image IoU and v2 Cartesian/BEV IoU are separate interfaces.** V2 preserves original RADDet Cartesian rectangles and supports physical axis-aligned IoU/AP; optional oriented IoU/AP requires genuine orientation targets. CARRADA's converted native radar-support enclosures retain that distinct label provenance. A projected RA rectangle cannot reconstruct an original vehicle footprint. RadSimReal's exact adaptations/evaluator remain unavailable, so matching coordinate domains and thresholds does not alone establish published evaluator parity. See [DETECTOR_IMPLEMENTATION.md](DETECTOR_IMPLEMENTATION.md).

A uniform `sin(theta)` grid and a uniform `theta` grid assign different columns to the same physical bearing. Convert source angles using the destination grid explicitly. Do not apply a fixed linear degrees-to-column conversion to the default grid. Plot a known target and verify left/right, near/far, zero-Doppler index and box alignment before training. Fixed preprocessing limits are stored with the model; do not fit normalization on the test set.

## Measured PSF and noise

Use a narrow stationary isolated target, preferably a corner reflector, and a stationary sensor. Capture multiple full cubes on a common grid. Keep metadata for axis order, range/velocity/angular sampling, target index, linear/log domain, magnitude/power/complex convention, sensor settings and calibration-target properties. Average repeated measurements consistently with that domain. Complex averaging additionally requires phase stability or alignment.

Crop a PSF centered on the target. The publication uses 99% energy retention and response below the unaveraged noise level as motivation for truncation. Inspect the retained/discarded kernel energy and sidelobe level; an arbitrary crop does not become calibrated merely because its size looks plausible. Supply a kernel whose sampling matches the simulator's configured grid. A changed bin spacing requires remeasurement or justified resampling.

Estimate noise from object-free **tensor cells before Doppler maximum**. Name the statistic explicitly: complex total variance `E|n|²`, real-component variance, amplitude variance and power variance have different meanings. The default demo's `noise_std` is the standard deviation of signed additive real Gaussian tensor noise, not a manufacturer noise figure or a universally valid power-noise parameter. Power-only PSFs cannot provide coherent phase information. Peak normalization removes absolute gain, so absolute reflectivity requires separate target/gain calibration.

For application-specific scattering, supply dense RF reflectors with position, radial velocity/velocity convention and calibrated weights, using the README interfaces. V2 also supports triangle-mesh visibility, moving OBJ templates and explicit RF materials. Segmented Lidar can provide geometry, but Lidar intensity is not radar reflectivity; material assignment and normals still require an RF model. Multiple-bounce propagation, diffraction, weather and the authors' CARLA assets remain outside the single-bounce backend; [SCATTERING.md](SCATTERING.md) records that boundary.

## RADDet

Obtain data through [the original RADDet repository](https://github.com/ZhangAoCanada/RADDet), checking its current access notes and licenses. The original code is TensorFlow based, with its own requirements/configuration and advertised pretrained checkpoint link. Its inspected [requirements](https://raw.githubusercontent.com/ZhangAoCanada/RADDet/master/requirements.txt) pin TensorFlow 2.3.0, NumPy 1.18.5, OpenCV 4.1.2.30, Matplotlib 3.3.1, scikit-learn 0.23.2 and tqdm 4.50.2. Those historical packages may require an older compatible Python environment; they are not dependencies of this package. Use an isolated environment if running that external implementation. This project's PyTorch environment and checkpoint format are not drop-in replacements. An original RADDet pretrained checkpoint was not trained under this project's disclosed baseline recipe or necessarily the RadSimReal scene split.

Original RADDet cube shape is described as `(256,256,64)` in range/azimuth/Doppler order. Convert it to the package's `(range,Doppler,azimuth)` order before tensor operations. Confirm the actual source file's complex/amplitude/power convention. Preserve original integer frame IDs when merging the source's train and test directories; original directory membership does not implement RadSimReal's replacement split.

Use:

```python
from radsimreal.datasets.protocol import scene_id_for_frame, split_for_frame

scene = scene_id_for_frame(original_frame_id)
split = split_for_frame(original_frame_id)
if split is None:
    raise ValueError("Original frame ID has no published scene assignment")
```

Scenes 9 and 11 are held out. Every interval endpoint is inclusive. IDs 556–558 and 725–730 have no assignment. Do not renumber files and then apply the table, create synthetic replacements for absent IDs, or randomly split neighboring real frames. The table covers 10,284 possible IDs while 10,158 available images are reported: only an observed-file manifest identifies the additional missing IDs. Inspect the resulting counts against the reference 8,196 train and 1,962 test; explain discrepancies instead of silently discarding data.

## CARRADA

Use [the authors' dataset page](https://arthurouaknine.github.io/codeanddata/carrada) and [official code](https://github.com/valeoai/carrada_dataset). It supplies multiple radar views and annotations. Prefer a known full-tensor export when reproducing Doppler maximum; a precomputed RA view might use another reduction or normalization. Preserve sequence IDs, transforms and label semantics during conversion.

The RadSimReal publication reports 2,208 train and 276 test images but does not supply their exact manifests. Do not substitute a convenient CARRADA split and claim the resulting AP reproduces the paper. Use a declared scene-disjoint split for exploratory work, or obtain the authors' original manifests for a direct comparison. Respect separate data/code licenses.

## CRUW

Consult [CRUW resources](https://www.cruwdataset.org/resources) and [annotation information](https://www.cruwdataset.org/introduction). CRUW and the ROD2021 subset/annotation formats are not interchangeable without checking their exact definitions. Native radar class/center annotations do not directly supply the paper's box evaluation labels.

RadSimReal reports 9,623 training and 2,226 test images and says it extended centers to boxes with CFAR. Its exact frame list, generated boxes and CFAR settings are unavailable in the audited resources. Obtain those artifacts for a direct comparison. If you generate new CFAR-derived boxes, record detector threshold/window/guard rules, component grouping and conversion to RA rectangles; report the result as a new annotation protocol. The package does not claim to recover the unpublished labels automatically.

## Comparison study checklist

Keep a frozen manifest, original IDs, calibration hashes, configuration, annotation conversion version and training checkpoint for each run. Evaluate on real held-out scenes and explicitly state the box domain, AP interpolation and matching conventions; retain full precision-recall data. V2 evaluates physical Cartesian boxes at 0.1/0.3/0.5 IoU and implements U-Net, RADDet and Probabilistic families with documented adaptations. The retained v1 evaluator measures RA-image rectangles. A direct published-result comparison also needs verified author-equivalent labels, splits, training and evaluator conventions; CARRADA support enclosures and absent yaw must remain explicit. Implemented architectures, synthetic validation AP and small training runs do not establish sim-to-real performance or reproduce the published experimental matrix.
