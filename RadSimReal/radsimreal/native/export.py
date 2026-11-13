"""Synthetic export into declared RADDet/CARRADA native layouts.

The array conversion inverts NativeReader exactly, apart from documented real
power clipping, log-flooring and storage precision. Native labels have explicit
synthetic policies; matching a schema is not the authors' annotation generator.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import pickle
import re
import numpy as np

from .profiles import NativeProfile


def canonical_to_native(tensor, profile, power_floor=1e-12):
    """Convert canonical R,D,A complex fields or linear powers to native R,A,D.

    Complex output requires actual complex fields: phases are never invented.
    Real output uses squared magnitude for complex input, or nonnegative-clipped
    real intensity-surrogate samples. amplitude=sqrt(power), db=10log10(power).
    All real storage is float32; complex storage is complex64, checked after cast.
    Zero cannot be represented exactly in dB, so power_floor is explicit.
    """
    if not isinstance(profile, NativeProfile):
        raise ValueError("profile must be NativeProfile")
    values = np.asarray(tensor)
    if values.shape != profile.shape or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"Canonical tensor must be numeric R,D,A with shape {profile.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Canonical tensor contains nonfinite values")
    representation = profile.representation
    if representation == "complex":
        if not np.iscomplexobj(values):
            raise ValueError("Native complex export needs complex fields; power values cannot recover phase")
        values = values.astype(np.complex128)
        if max(float(np.max(np.abs(values.real), initial=0)), float(np.max(np.abs(values.imag), initial=0))) > float(np.finfo(np.float32).max):
            raise ValueError("Complex tensor exceeds complex64 storage range")
        output = values.astype(np.complex64)
    else:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                power = np.abs(values.astype(np.complex128)) ** 2 if np.iscomplexobj(values) else np.maximum(values.astype(np.float64), 0)
                if representation == "power":
                    output = power
                elif representation == "amplitude":
                    output = np.sqrt(power)
                elif representation == "db":
                    if not np.isfinite(power_floor) or power_floor <= 0:
                        raise ValueError("power_floor must be finite and positive")
                    output = 10 * np.log10(np.maximum(power, power_floor))
                else:
                    raise ValueError("Real native export requires an explicit power, amplitude or db profile")
            except FloatingPointError as error:
                raise ValueError("Canonical tensor overflows conversion to the declared native response") from error
        if not np.isfinite(output).all() or np.max(np.abs(output), initial=0) > np.finfo(np.float32).max:
            raise ValueError("Native tensor exceeds finite float32 storage range")
        output = output.astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite tensor after native storage conversion")
    output = np.transpose(output, (0, 2, 1))
    if profile.range_reversed:
        output = output[::-1, :, :]
    if profile.azimuth_reversed:
        output = output[:, ::-1, :]
    if profile.canonical_doppler_flip:
        output = output[:, :, ::-1]
    return np.ascontiguousarray(output)


def _exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _exclusive_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)


def _physical_cartesian_box(obj, affine):
    """Invert a declared native-grid affine using the original physical footprint."""
    corners = np.asarray(obj.corners(), dtype=float)
    if corners.ndim != 2 or corners.shape[1] != 2 or not np.isfinite(corners).all():
        raise ValueError("Scene object corners must be finite Nx2 physical lateral/forward coordinates")
    columns = (corners[:, 0] - affine["x_from_col_offset"]) / affine["x_from_col_scale"]
    rows = (corners[:, 1] - affine["y_from_row_offset"]) / affine["y_from_row_scale"]
    lower, upper = np.array([rows.min(), columns.min()]), np.array([rows.max(), columns.max()])
    return np.r_[(lower + upper) / 2, upper - lower], np.r_[corners.min(axis=0), corners.max(axis=0)]


def _doppler_support_crosses_boundary(indices, psf_radius, shape):
    lower = np.floor(indices[:, 1].min()) - psf_radius[1] - .5
    upper = np.ceil(indices[:, 1].max()) + psf_radius[1] + .5
    return bool(lower < -.5 or upper > shape[1] - .5)


def _canonical_support_box(indices, psf_radius, shape, doppler_wrap=False):
    """Return canonical R,D,A center/size of sampled deposits plus truncated PSF."""
    # Half-cell edges represent at least one complete bin, including delta PSFs.
    lower = np.floor(indices.min(axis=0)) - psf_radius - .5
    upper = np.ceil(indices.max(axis=0)) + psf_radius + .5
    lower = np.maximum(lower, -.5)
    upper = np.minimum(upper, np.asarray(shape) - .5)
    if doppler_wrap and _doppler_support_crosses_boundary(indices, psf_radius, shape):
        # Public native GT uses one linear box, not a toroidal box. A wrapped
        # lobe can occupy both ends; retain it with a conservative full-D box.
        lower[1], upper[1] = -.5, shape[1] - .5
    return (lower + upper) / 2, np.maximum(upper - lower, 1.)


def _raddet_rad_box(indices, psf_radius, profile, doppler_wrap=False):
    center, size = _canonical_support_box(indices, psf_radius, profile.shape, doppler_wrap)
    if profile.range_reversed:
        center[0] = profile.shape[0] - 1 - center[0]
    if profile.canonical_doppler_flip:
        center[1] = profile.shape[1] - 1 - center[1]
    if profile.azimuth_reversed:
        center[2] = profile.shape[2] - 1 - center[2]
    center, size = center[[0, 2, 1]], size[[0, 2, 1]]
    center[2] = np.mod(center[2], profile.shape_native[2])  # periodic Doppler, released-label convention
    return np.r_[center, size]


def _expanded_view(points, shape, radius, periodic_columns=False):
    """Rasterized union of local truncated-PSF rectangles, not an invented footprint."""
    points = np.asarray(points, dtype=np.int64).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0], 0, shape[0] - 1)
    points[:, 1] = np.mod(points[:, 1], shape[1]) if periodic_columns else np.clip(points[:, 1], 0, shape[1] - 1)
    sparse = np.unique(points, axis=0)
    mask = np.zeros(shape, dtype=bool)
    radius = np.maximum(np.asarray(radius, dtype=int), 1)
    for point in sparse:
        lower = np.maximum(point - radius, 0)
        upper = np.minimum(point + radius + 1, shape)
        if periodic_columns:
            columns = np.mod(np.arange(point[1] - radius[1], point[1] + radius[1] + 1), shape[1])
            rows = np.arange(lower[0], upper[0])
            mask[np.ix_(rows, columns)] = True
        else:
            mask[lower[0]:upper[0], lower[1]:upper[1]] = True
    dense = np.argwhere(mask)
    lower, upper = dense.min(axis=0), dense.max(axis=0)
    return {"sparse": sparse.tolist(), "box": [lower.tolist(), upper.tolist()],
            "dense": dense.tolist(), "label": 3,
            "synthetic_label_policy": "union of sampled reflector support expanded by truncated PSF halfwidth, at least one bin",
            "periodic_doppler_columns": bool(periodic_columns),
            "box_boundary_policy": "linear enclosure of periodic support; crossing support can span full width" if periodic_columns else "clipped linear enclosure"}


def _carrada_object_views(xyz, velocity, psf_radius, profile, doppler_wrap=False):
    """Reproduce declared CARRADA annotation transforms (distinct from tensor axes)."""
    planar_range = np.hypot(xyz[:, 0], xyz[:, 1])
    angle = np.arctan2(xyz[:, 0], xyz[:, 1])
    r_index = np.trunc((planar_range - profile.range_min) / profile.range_step).astype(int)
    range_origin = profile.shape_native[0] if profile.annotation_range_reverse_origin == "R" else profile.shape_native[0] - 1
    ra_row = range_origin - r_index
    if profile.annotation_angle_step is not None:
        # Public AnnotationTransformer uses midpoint + int(theta/ANGLE_RES).
        # This declared annotation convention is independent of tensor flips.
        ra_col = profile.shape_native[1] // 2 + np.trunc(angle / profile.annotation_angle_step).astype(int)
    else:
        angular = np.sin(angle) if profile.azimuth_coordinate == "sin" else angle
        ra_col = np.rint((angular - profile.angular_min) / profile.angular_step).astype(int)
        if profile.azimuth_reversed:
            ra_col = profile.shape_native[1] - 1 - ra_col
    distance = np.linalg.norm(xyz, axis=1)
    receding = np.einsum("ij,ij->i", xyz, velocity) / np.maximum(distance, 1e-30)
    native_velocity = -receding if profile.doppler_positive == "approaching" else receding
    zero_bin = -profile.velocity_min_native / profile.velocity_step
    rd_col = np.rint(zero_bin).astype(int) + np.trunc(native_velocity / profile.velocity_step).astype(int)
    if doppler_wrap:
        rd_col = np.mod(rd_col, profile.shape_native[2])
    if profile.doppler_reversed:
        rd_col = profile.shape_native[2] - 1 - rd_col
    ra = _expanded_view(np.c_[ra_row, ra_col], profile.shape_native[:2], psf_radius[[0, 2]])
    rd = _expanded_view(np.c_[r_index, rd_col], (profile.shape_native[0], profile.shape_native[2]), psf_radius[[0, 1]], periodic_columns=doppler_wrap)
    return {"range_angle": ra, "range_doppler": rd}


class ExportNativeWriter:
    """Write a fresh synthetic native dataset without touching original recordings.

    RADDet IDs must be globally unique original-style nonnegative integers;
    CARRADA IDs are unique within explicitly split-specific sequences. Label
    JSON is spooled per frame and streamed at finalization to bound memory.
    """
    def __init__(self, root, profile, write_views=False):
        self.root = Path(root)
        self.profile = NativeProfile.from_dict(profile) if isinstance(profile, dict) else profile
        if not isinstance(self.profile, NativeProfile) or self.profile.representation is None:
            raise ValueError("Native export requires an explicit NativeProfile and response representation")
        if self.profile.name == "raddet" and self.profile.cartesian_affine is None:
            raise ValueError("RADDet native export requires explicit cartesian_affine; physical boxes cannot be inferred from RA rectangles")
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError("Native export destination must be new or empty; original datasets are never overwritten")
        self.root.mkdir(parents=True, exist_ok=True)
        self.write_views = bool(write_views)
        self.rows = []
        self._seen = set()
        self._sequence_split = {}
        self._sequence_frames = {}
        self._sequence_instances = {}
        self._finalized = False
        _exclusive_json(self.root / "NATIVE_EXPORT.json", {
            "format_version": 1, "dataset": self.profile.name, "profile": self.profile.to_dict(),
            "source": "independent synthetic mesh/procedural simulation, not original real recordings",
            "label_policy": "RADDet Cartesian labels from original physical footprints; RAD/CARRADA radar labels from PSF-expanded reflector support",
            "benchmark_equivalence_claimed": False, "native_views_written": self.write_views,
            "view_representation": "linear power maxima; not claimed to match every original processed-view pipeline"})

    def write_frame(self, record_id, split, scene, reflections, result, *, sequence=None):
        if self._finalized:
            raise RuntimeError("Native writer has already been finalized")
        text = str(record_id)
        if not text.isascii() or not text.isdigit():
            raise ValueError("Native record_id must contain nonnegative integer digits")
        frame = text.zfill(6)
        numeric_id = int(text)
        if split not in ("train", "val", "test"):
            raise ValueError("Native split must be train, val or test")
        if sequence is None:
            sequence = f"synthetic_{split}_000"
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(sequence)):
            raise ValueError("Sequence names must be safe folder names containing letters, digits, underscore or hyphen")
        sequence = str(sequence)
        if sequence in self._sequence_split and self._sequence_split[sequence] != split:
            raise ValueError("A synthetic sequence cannot occur across train/val/test splits")
        key = numeric_id if self.profile.name == "raddet" else (sequence, numeric_id)
        if key in self._seen:
            raise FileExistsError("Duplicate native frame ID; no dataset files were overwritten")
        native = canonical_to_native(result.tensor, self.profile)
        from ..radar import points_to_indices
        radar_config = self.profile.radar_config()
        declared_wrap = result.metadata.get("doppler_wrap")
        if declared_wrap is None:
            declared_wrap = "periodic Doppler" in str(result.metadata.get("boundary", ""))
        if not isinstance(declared_wrap, (bool, np.bool_)):
            raise ValueError("Simulation metadata doppler_wrap must be a boolean")
        doppler_wrap = bool(declared_wrap)
        radar_config.doppler_wrap = doppler_wrap
        indices, valid = points_to_indices(reflections.xyz, reflections.velocity, radar_config)
        zero_coefficients = np.asarray(reflections.coefficients) == 0
        valid &= ~zero_coefficients
        kernel = np.asarray(result.psf)
        if kernel.ndim != 3:
            raise ValueError("Simulation result PSF must be 3D")
        psf_radius = np.asarray(kernel.shape, dtype=int) // 2
        classes, radar_boxes, cart_boxes, physical_boxes, object_meta, carrada_views = [], [], [], [], [], {}
        skipped = []
        for obj in scene.objects:
            if str(obj.label).lower() != "car":
                continue
            selected = valid & (reflections.object_ids == obj.object_id)
            if not selected.any():
                skipped.append(int(obj.object_id))
                continue
            classes.append("car")
            corners = np.asarray(obj.corners())
            physical_box = np.r_[corners.min(axis=0), corners.max(axis=0)]
            physical_boxes.append(physical_box.tolist())
            object_meta.append({"object_id": int(obj.object_id), "in_domain_reflection_count": int(selected.sum()),
                                "physical_footprint_corners_metres": corners.tolist(),
                                "physical_bev_box_xyxy_metres": physical_box.tolist(),
                                "doppler_support_crosses_periodic_boundary": doppler_wrap and _doppler_support_crosses_boundary(indices[selected], psf_radius, self.profile.shape)})
            if self.profile.name == "raddet":
                radar_boxes.append(_raddet_rad_box(indices[selected], psf_radius, self.profile, doppler_wrap))
                cart, _ = _physical_cartesian_box(obj, self.profile.cartesian_affine)
                cart_boxes.append(cart)
            else:
                instance = f"car_{int(obj.object_id)}"
                carrada_views[instance] = _carrada_object_views(reflections.xyz[selected], reflections.velocity[selected], psf_radius, self.profile, doppler_wrap)
        if self.profile.name == "raddet":
            tensor_path = Path(split) / "RAD" / "part1" / f"{frame}.npy"
            label_path = Path(split) / "gt" / "part1" / f"{frame}.pickle"
            label = {"classes": classes, "boxes": np.asarray(radar_boxes, dtype=np.float32).reshape(-1, 6),
                     "cart_boxes": np.asarray(cart_boxes, dtype=np.float32).reshape(-1, 4),
                     "synthetic_label_policy": "Cartesian footprint enclosure; 3D RAD truncated-PSF-expanded reflector enclosure"}
        else:
            tensor_path = Path(sequence) / "RAD_numpy" / f"{frame}.npy"
            label_path = Path("native_metadata") / sequence / f"{frame}.annotations.json"
        metadata_path = Path("native_metadata") / sequence / f"{frame}.json"
        negative_count = int(np.count_nonzero(np.asarray(result.tensor) < 0)) if not np.iscomplexobj(result.tensor) else 0
        row = {"id": frame, "split": split, "sequence": sequence,
               "tensor": tensor_path.as_posix(), "annotation": label_path.as_posix(),
               "metadata": metadata_path.as_posix(), "labeled_objects": len(classes),
               "skipped_objects_without_in_domain_returns": skipped,
               "negative_real_tensor_cells_clipped_for_power_export": negative_count,
               "zero_coefficient_reflectors_ignored": int(np.count_nonzero(zero_coefficients)),
               "doppler_wrap": doppler_wrap,
               "radar_box_wrap_policy": "periodic tensor support; a single linear native box spans the full Doppler period when support crosses its boundary" if doppler_wrap else "zero-extended Doppler; out-of-domain reflectors excluded",
               "label_domain": "native_RADDet_Cartesian_rectangles" if self.profile.name == "raddet" else "CARRADA_native_radar_annotation_support",
               "physical_boxes_bev": physical_boxes}
        # Validate paths and all contents before the first per-frame mutation.
        for relative in (tensor_path, label_path, metadata_path):
            if (self.root / relative).exists():
                raise FileExistsError(f"Native frame output already exists: {relative.as_posix()}")
        _exclusive_npy(self.root / tensor_path, native)
        if self.profile.name == "raddet":
            destination = self.root / label_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                pickle.dump(label, stream, protocol=4)
        else:
            _exclusive_json(self.root / label_path, carrada_views)
            self._sequence_frames.setdefault(sequence, []).append((frame, label_path))
            self._sequence_instances.setdefault(sequence, set()).update(carrada_views)
        _exclusive_json(self.root / metadata_path, {**row, "objects": object_meta,
            "array_representation": self.profile.representation,
            "label_policy": "native schema with explicitly reconstructed synthetic support; no author-label equivalence",
            "noise_clipping": "Only real signed intensity surrogate cells are clipped before nonnegative native response conversion",
            "db_power_floor": 1e-12, "source_simulation_metadata": result.metadata})
        if self.write_views:
            # Export explicitly declared linear-power views; no original CARRADA
            # inverse-FFT/summing/log view preprocessing is silently assumed.
            canonical_power = np.abs(result.tensor) ** 2 if np.iscomplexobj(result.tensor) else np.maximum(result.tensor, 0)
            ra = canonical_power.max(axis=1)
            rd = canonical_power.max(axis=2)
            if self.profile.range_reversed:
                ra, rd = ra[::-1], rd[::-1]
            if self.profile.azimuth_reversed:
                ra = ra[:, ::-1]
            if self.profile.canonical_doppler_flip:
                rd = rd[:, ::-1]
            if self.profile.name == "carrada":
                for folder, view in (("range_angle_numpy", ra), ("range_doppler_numpy", rd)):
                    if not np.isfinite(view).all() or np.max(view, initial=0) > np.finfo(np.float32).max:
                        raise ValueError("Native linear-power view exceeds float32 range")
                    _exclusive_npy(self.root / sequence / folder / f"{frame}.npy", view.astype(np.float32))
        self._seen.add(key)
        self._sequence_split[sequence] = split
        self.rows.append(row)
        return row

    def finalize(self):
        if self._finalized:
            raise RuntimeError("Native writer has already been finalized")
        if self.profile.name == "carrada":
            path = self.root / "annotations_frame_oriented.json"
            with path.open("x", encoding="utf-8") as stream:
                stream.write("{")
                for sequence_i, sequence in enumerate(sorted(self._sequence_frames)):
                    if sequence_i:
                        stream.write(",")
                    stream.write(json.dumps(sequence) + ":{")
                    for frame_i, (frame, label_path) in enumerate(self._sequence_frames[sequence]):
                        if frame_i:
                            stream.write(",")
                        stream.write(json.dumps(frame) + ":")
                        stream.write((self.root / label_path).read_text(encoding="utf-8"))
                    stream.write("}")
                stream.write("}\n")
            references = {sequence: {"instances": sorted(instances), "labels": [3] * len(instances),
                                     "description": "independent synthetic radar-return annotation support"}
                          for sequence, instances in self._sequence_instances.items()}
            _exclusive_json(self.root / "data_seq_ref.json", references)
            with (self.root / "validated_seqs.txt").open("x", encoding="utf-8") as stream:
                stream.write("\n".join(sorted(self._sequence_frames)) + "\n")
        splits = {split: sorted(sequence for sequence, value in self._sequence_split.items() if value == split)
                  for split in ("train", "val", "test")}
        _exclusive_json(self.root / "splits.json", splits)
        manifest = {"format_version": 1, "dataset": self.profile.name, "profile": self.profile.to_dict(),
                    "source": "synthetic native-layout export", "counts": {k: sum(r["split"] == k for r in self.rows) for k in splits},
                    "rows": self.rows, "benchmark_equivalence_claimed": False,
                    "label_policy": "RADDet physical Cartesian footprints; native RF labels are reconstructed PSF-expanded reflector supports"}
        _exclusive_json(self.root / "native_manifest.json", manifest)
        self._finalized = True
        return manifest
