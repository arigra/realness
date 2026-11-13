# Native RADDet and CARRADA adapters

These readers implement the official public layouts and explicit coordinate
conversions. They have been tested with on-disk format fixtures. No access to
the user's HPC recordings is implied. A source-derived reference profile is
not a measured calibration or a locally validated coordinate overlay.

## Declared native profile

`NativeProfile` stores shape in native `(range,azimuth,Doppler)` order, response
representation, sampling, reversals, Doppler sign and the angular coordinate.
`radar_config()` returns the corresponding canonical `(range,Doppler,azimuth)`
grid with increasing range/angle and **positive-receding** velocity. If a
positive-approaching even-length shifted FFT axis is reversed, its canonical
minimum is `-(D/2-1)*dv`, not `-D/2*dv`. The one-bin asymmetry is preserved.

Complex dtype identifies a complex field representation. Real values do not
identify power, magnitude or logged response: the reader refuses to proceed
without `representation=power|amplitude|db`. `db` specifically means power dB,
converted as `10**(value/10)`. Other log conventions need an explicit conversion
before loading. Amplitudes are converted to float64 before squared magnitude,
avoiding integer overflow. The reader does not resize, fit normalization or
collapse Doppler. The returned tensor is complex field or linear power.

Profiles can be saved as JSON and loaded with strict unknown-key rejection.
Reference defaults record evidence and unresolved assumptions. Local release
metadata and diagnostic overlays must resolve those assumptions before a
sensor-validation claim. A shape or similar radar hardware does not establish
identical processing, PSF, noise or physical angular resolution.

`NativeReader.from_manifest` opens an existing index without a recursive raw-data
rescan. It checks unique original IDs, contained tensor/annotation paths and the
first tensor header; subsequent tensors are checked at loading. Files added
after index creation cannot enter a split. A CARRADA annotation archive outside
the tensor root must be explicitly supplied through `annotation_path`.

## RADDet

Supported tensor layout is `train|test/RAD/part*/frame.npy` (also a directly
selected RAD subtree), matching `gt/part*/frame.pickle`. Original numeric frame
IDs are preserved. Duplicate original IDs across copied roots are rejected.
The reference angular grid is uniform in array spatial frequency,
`s=(2*j/A-1)*(76.8/77)`, with physical angle `arcsin(s)`, following the official
projection helper. The range reversal follows that helper and the direct
inference plotting path; confirm the local release rather than treating a
plot-label scalar as a guarantee.

Native `cart_boxes` are retained as Cartesian rectangles. Public center-size
values are row-centre, column-centre, row-size, column-size; the reader returns
`boxes_cartesian_native` as **column,row XYXY**. Native `boxes` are retained
and converted separately into canonical RA bin XYXY. Missing Cartesian labels
raise an error rather than substituting a polar enclosure.

**No metre origin is silently selected.** `cartesian_affine` defaults to null.
Consequently `boxes_bev` is empty and `requires_cartesian_affine=true` until an
explicit affine is supplied. Native Cartesian-bin IoU is valid and unchanged
by a common invertible axis scaling/reflection/translation; physical placement
in the simulator is a separate validation issue.

`raddet_reference_cartesian_affine(profile)` returns an explicit inverse of
the official projection helper:

```text
lateral_m = (native_column + 1)*range_step - 50
forward_m = (R - 1 - native_row)*range_step
```

This source-derived inverse is not enabled automatically. The helper has
integer rounding and a hard-coded 50 m lateral shift. Using the affine does
not undo rounding or prove a modified release uses the same origin. Set
`cartesian_mapping_validated` only after checking actual frame overlays. Native
labels are axis-aligned; no vehicle yaw is invented.

Python pickle requires `trusted_labels=True`, intended for the user's trusted
original public dataset. Pickle may execute code. Header-only inventory never
loads those labels.

## CARRADA

Full tensors live in `sequence/RAD_numpy/frame.npy`. This RAD archive is a
separate distribution from `range_angle_numpy` / `range_doppler_numpy` views.
Processed 2D views cannot recreate missing full tensors or coherent phase.
If tensor and label archives are separate, pass `annotation_path` to
`NativeReader` with `annotations_frame_oriented.json` from the original archive.
JSON is decoded one top-level sequence at a time and only the current sequence
is cached, avoiding retention of the complete dense annotation file.

The reference profile uses the published positive-approaching annotation
Doppler sign and a uniform physical-angle annotation grid. The full tensor's
actual processing/angular coordinate and response convention require local
verification; a folder name does not establish those facts. Its published
annotation mapping uses range row `R-int(range/dr)` and angle column
`A/2+int(theta/(pi/256))`. The resulting one-bin range-origin distinction from
tensor reversal `R-1-row` is recorded, not silently hidden.

Native labels are **radar annotation regions**, principally RA/RD sparse
points, boxes and dense support. `instance_oriented` in a filename means
grouped by object instance; it does not mean oriented physical boxes.
`boxes_bev` is explicitly the Cartesian axis-aligned enclosure of the native
dense/sparse annotation points. If only an RA box is available, the reader
encloses the full annular sector, including interior angle extrema. Neither
conversion recovers physical vehicle footprint or yaw.

All results therefore carry
`label_domain=Cartesian_enclosure_of_CARRADA_radar_annotation_support`,
`geometry_approximation=true` and per-instance provenance. These labels may be
used for a declared converted-region experiment. They are not asserted to
equal RadSimReal's unpublished CARRADA label conversion or oriented-car IoU.
Missing frames/labels raise errors; they are not relabeled as empty scenes.
The reference car IDs `car` and `3` must be checked against the local release's
`data_seq_ref.json`; `car_labels` is configurable.

## Scene-disjoint manifests

`make_native_manifest` assigns whole scenes/sequences. RADDet defaults to the
published test scenes 9 and 11; validation scenes must be explicitly named.
It intersects original frame IDs with actual files, reports uncovered IDs
separately and never fabricates the reported training count. CARRADA requires
explicit validation/test sequence lists or a complete sequence mapping file.
No unpublished author split is invented. Duplicate or conflicting sequence
assignments and original frame IDs are rejected. An explicit split JSON may
contain `train`, `val`, `test` sequence lists or a `sequences` mapping.
Partial RADDet inventories may omit default reference test scenes; the manifest
reports them as `missing_reference_test_scenes`. This permits bounded pilot
inspection but is not a complete benchmark. Explicitly requested validation or
test sequences must exist.

## Primary source anchors

- [RADDet loader](https://raw.githubusercontent.com/ZhangAoCanada/RADDet/main/util/loader.py),
  [configuration](https://raw.githubusercontent.com/ZhangAoCanada/RADDet/main/config.json),
  [projection helper](https://raw.githubusercontent.com/ZhangAoCanada/RADDet/main/util/helper.py),
  [Cartesian annotation drawing](https://raw.githubusercontent.com/ZhangAoCanada/RADDet/main/util/drawer.py),
  [Cartesian label encoding](https://raw.githubusercontent.com/ZhangAoCanada/RADDet/main/dataset/batch_data_generator.py).
- [CARRADA annotation transform](https://raw.githubusercontent.com/valeoai/carrada_dataset/master/carrada_dataset/utils/transform_annotations.py),
  [annotation schema generation](https://raw.githubusercontent.com/valeoai/carrada_dataset/master/carrada_dataset/scripts/generate_annotation_files.py),
  [data-view transforms](https://raw.githubusercontent.com/valeoai/carrada_dataset/master/carrada_dataset/utils/transform_data.py),
  [official downloads](https://arthurouaknine.github.io/codeanddata/carrada).
