"""Executable native-data preparation, measurement selection and simulation.

These workflows run beside a user's datasets. They never pretend fixtures are
real measurements and never silently estimate calibration from a test split.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import hashlib
import shutil
import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter
from .profiles import NativeProfile, reference_profile, raddet_reference_cartesian_affine
from .readers import NativeReader, discover_records
from .manifest import make_native_manifest
from .calibration import json_write, record_for_id
from ..config import load_config, save_config


def make_reader(root, dataset, *, profile=None, representation=None, trusted_labels=False,
                annotation_path=None, reference_cartesian=False):
    if profile is None and representation is not None:
        records = discover_records(root, dataset)
        probe = np.load(records[0].path, mmap_mode="r", allow_pickle=False)
        profile = reference_profile(dataset, tuple(probe.shape), representation)
    elif isinstance(profile, (str, Path)):
        profile = NativeProfile.load(profile)
    elif isinstance(profile, dict):
        profile = NativeProfile.from_dict(profile)
    if profile is not None and representation is not None:
        profile = replace(profile, representation=representation)
    reader = NativeReader(root, dataset, profile=profile, trusted_labels=trusted_labels,
                          annotation_path=annotation_path)
    if reference_cartesian:
        if dataset.lower() != "raddet":
            raise ValueError("Reference Cartesian affine applies to RADDet only")
        reader.profile.cartesian_affine = raddet_reference_cartesian_affine(reader.profile)
    return reader


def assert_sensor_sampling(radar, profile):
    expected = profile.radar_config()
    for field in ("ranges", "velocities", "angular_grid"):
        actual_values, expected_values = np.asarray(getattr(radar, field)), np.asarray(getattr(expected, field))
        if actual_values.shape != expected_values.shape or not np.isfinite(actual_values).all() or not np.allclose(actual_values, expected_values, rtol=1e-7, atol=1e-10):
            raise ValueError(f"Simulation/native profile mismatch for {field}; spacing and coordinate origins must both match")
    if radar.azimuth_coordinate != expected.azimuth_coordinate or radar.representation != expected.representation:
        raise ValueError("Simulation/native profile angular coordinate or signal representation differs")


def default_bev_grid(profile, height=None, width=None):
    from ..bev.geometry import BEVGrid
    extent = profile.shape_native[0] * profile.range_step + profile.range_min
    xmin, xmax, ymin, ymax = -extent, extent, 0., extent
    affine = profile.cartesian_affine
    if affine is not None:
        xx = [affine["x_from_col_offset"], affine["x_from_col_offset"] + 2 * profile.shape_native[0] * affine["x_from_col_scale"]]
        yy = [affine["y_from_row_offset"], affine["y_from_row_offset"] + profile.shape_native[0] * affine["y_from_row_scale"]]
        xmin, xmax, ymin, ymax = min(xmin, *xx), max(xmax, *xx), min(ymin, *yy), max(ymax, *yy)
    return BEVGrid(x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax,
                   height=profile.shape_native[0] if height is None else height,
                   width=2 * profile.shape_native[0] if width is None else width)


def automatic_sequence_split(reader):
    """Disclosed pilot partition, not an unpublished author CARRADA manifest."""
    actual = sorted({r.sequence for r in reader.records if not r.sequence.startswith("unassigned/")})
    if reader.dataset == "raddet":
        test = [s for s in ("scene_9", "scene_11") if s in actual]
        remaining = [s for s in actual if s not in test]
        if len(remaining) < 2:
            raise ValueError("Need at least two non-test RADDet scenes for disjoint training/validation")
        return [remaining[0]], test
    if len(actual) < 3:
        raise ValueError("Need at least three CARRADA sequences for a disjoint pilot train/val/test partition")
    count = max(1, int(round(len(actual) * .15)))
    if 2 * count >= len(actual):
        count = 1
    return actual[-2 * count:-count], actual[-count:]


def prepare_native(reader, output_dir, *, val_sequences=None, test_sequences=None,
                   split_file=None, auto_split=False, grid=None, fit_normalization_frames=16,
                   verify_all_labels=True):
    """Create a lazy Cartesian dataset, explicitly excluding unannotated frames.

    CARRADA's full RAD archive can contain warm-up/boundary frames absent from
    its annotation release. Those frames remain in the raw index and may be
    used for TRAIN-sequence calibration, but never become empty GT examples.
    Missing files/sequences and malformed labels remain errors. Quick checking
    still inventories all CARRADA annotation frame keys, while only a bounded
    subset receives full label validation/caching.
    """
    from ..config import SimulationConfig
    from ..datasets.bev_dataset import BEVDataset, DEFAULT_BEV_NORMALIZATION, fit_training_normalization
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Native prepared-data output must be an empty directory")
    if auto_split:
        if split_file is not None or val_sequences is not None or test_sequences is not None:
            raise ValueError("Choose either explicit splits or --auto-split")
        val_sequences, test_sequences = automatic_sequence_split(reader)
    plan = make_native_manifest(reader, val_sequences, test_sequences, split_file)
    if not plan["splits"]["train"] or not plan["splits"]["val"]:
        raise ValueError("Prepared data requires nonempty training and validation partitions")
    if reader.dataset == "raddet" and reader.profile.cartesian_affine is None:
        raise ValueError("Supply an explicit Cartesian label affine or select the source-derived reference Cartesian mapping")
    grid = default_bev_grid(reader.profile) if grid is None else grid
    label_domain = "native_RADDet_Cartesian_rectangles" if reader.dataset == "raddet" else "Cartesian_enclosure_of_CARRADA_radar_annotation_support"
    plan.update(source="native_reference", native_root=str(reader.root), native_profile=reader.profile.to_dict(),
                bev_grid=grid.to_dict(), trusted_labels=reader.trusted_labels,
                label_domain=label_domain, metric_domain="cartesian_bev_metres_axis_aligned",
                normalization=dict(DEFAULT_BEV_NORMALIZATION),
                native_records=[r.to_dict(reader.root) for r in reader.records],
                boundary_policy="reject_outside_grid_no_silent_label_crop",
                native_cartesian_iou_preserved_by_affine=reader.dataset == "raddet",
                benchmark_annotation_conversion_verified=False)
    paths = {str(r.annotation_path) for r in reader.records if r.annotation_path is not None}
    if reader.dataset == "carrada" and len(paths) == 1:
        plan["annotation_path"] = next(iter(paths))
    if auto_split:
        plan["split_source"] += "; automatic deterministic sequence-level pilot validation/test choices; not author CARRADA manifest"
    # Calibration needs a declared train scene/sequence, not a car annotation.
    # Keep these IDs before removing frames unavailable to supervised learning.
    plan["source_counts_before_annotation_filter"] = {split: len(values) for split, values in plan["splits"].items()}
    plan["calibration_record_ids"] = [row["id"] for row in plan["splits"]["train"]]
    plan["calibration_partition_policy"] = "all indexed raw frames in declared TRAIN scenes/sequences; object labels are not required"
    plan["excluded_records"] = []
    out.mkdir(parents=True, exist_ok=True)
    # Cache decoded per-frame labels beside the manifest. Random minibatches
    # must not repeatedly decode CARRADA's entire sequence annotation JSON.
    # Validate labels without loading the radar arrays or changing native files.
    lookup = {r.id: r for r in reader.records}
    label_report = {"frames_checked": 0, "cars": 0, "frames_with_missing_yaw": 0,
                    "native_geometry_preserved": reader.dataset == "raddet", "sequences": {},
                    "full_label_check": bool(verify_all_labels),
                    "excluded_unannotated_frames": 0}
    rows = [r for values in plan["splits"].values() for r in values]
    excluded_ids = set()

    def exclude_unannotated(row, message):
        if row["id"] in excluded_ids:
            return
        excluded_ids.add(row["id"])
        plan["excluded_records"].append({**row,
            "exclusion_reason": "native_frame_absent_from_annotation_release",
            "detail": str(message), "treated_as_empty_ground_truth": False,
            "eligible_for_training_sequence_calibration": row["split"] == "train"})
        label_report["excluded_unannotated_frames"] += 1

    def is_missing_carrada_frame(error):
        # Deliberately narrow: missing sequence/schema/key errors are not an
        # excuse to silently discard otherwise invalid annotations.
        message = error.args[0] if error.args else ""
        return (reader.dataset == "carrada" and isinstance(message, str)
                and message.startswith("CARRADA frame ")
                and message.endswith(" missing from annotations; missing is not an empty scene"))

    if reader.dataset == "carrada" and not verify_all_labels:
        # A first-12 pilot check alone could miss unannotated archive tails.
        # Inspect every sequence's key set without decoding every frame's boxes.
        from .annotations import load_json_sequence
        availability = {}
        for sequence in sorted({row["sequence"] for row in rows}):
            record = next(lookup[row["id"]] for row in rows if row["sequence"] == sequence)
            if record.annotation_path is None:
                raise FileNotFoundError("CARRADA annotations_frame_oriented.json was not found; supply the matching annotation archive")
            frames = load_json_sequence(record.annotation_path, sequence)
            availability[sequence] = set(frames)
        for row in rows:
            record = lookup[row["id"]]
            keys = availability[row["sequence"]]
            if record.path.stem not in keys and (record.frame_id is None or str(record.frame_id) not in keys):
                exclude_unannotated(row, f"CARRADA frame {record.id!r} missing from annotations; missing is not an empty scene")
        rows = [row for row in rows if row["id"] not in excluded_ids]
    checked_rows = rows if verify_all_labels else rows[:min(12, len(rows))]
    for row in checked_rows:
        try:
            ann = reader.load_annotations(lookup[row["id"]])
        except KeyError as error:
            if not is_missing_carrada_frame(error):
                raise
            exclude_unannotated(row, error.args[0])
            continue
        boxes = np.asarray(ann["boxes_bev"]).reshape(-1, 4)
        normalized = grid.normalize_boxes(boxes)
        if len(boxes) and (np.any(normalized < -1e-6) or np.any(normalized > 1 + 1e-6)):
            raise ValueError(f"BEV grid does not contain original Cartesian labels for {row['id']}; supply a larger --grid JSON")
        label_report["frames_checked"] += 1
        label_report["cars"] += len(boxes)
        label_report["frames_with_missing_yaw"] += int(len(boxes) > len(ann.get("oriented_boxes_bev", [])))
        label_report["sequences"][row["sequence"]] = label_report["sequences"].get(row["sequence"], 0) + 1
        label_path = "prepared_labels/" + hashlib.sha256(row["id"].encode()).hexdigest()[:20] + ".json"
        serializable = {k: value.tolist() if isinstance(value, np.ndarray) else value for k, value in ann.items()}
        json_write(out / label_path, serializable)
        row["prepared_annotation"] = label_path
    for split in plan["splits"]:
        plan["splits"][split] = [row for row in plan["splits"][split] if row["id"] not in excluded_ids]
    plan["counts"] = {split: len(values) for split, values in plan["splits"].items()}
    plan["excluded_record_counts_by_split"] = {split: sum(row["split"] == split for row in plan["excluded_records"])
                                                for split in plan["splits"]}
    plan["annotation_validation"] = {
        "all_carrada_frame_availability_checked": reader.dataset == "carrada",
        "full_label_check": bool(verify_all_labels),
        "checked_annotated_frames": label_report["frames_checked"],
        "scope": "all retained annotated native frames" if verify_all_labels else "pilot first12 retained labels; all CARRADA frame availability inventoried",
        "quick_mode_is_complete_label_validation": False if not verify_all_labels else None}
    label_report["source_counts_before_annotation_filter"] = plan["source_counts_before_annotation_filter"]
    label_report["retained_counts"] = plan["counts"]
    label_report["excluded_records"] = plan["excluded_records"]
    label_report["calibration_record_count"] = len(plan["calibration_record_ids"])
    if not plan["splits"]["train"] or not plan["splits"]["val"]:
        raise ValueError("Native annotation filtering left an empty training or validation partition; supply matching annotations or choose different scenes/sequences")
    json_write(out / "manifest.json", plan)
    save_config(SimulationConfig(radar=reader.profile.radar_config()), out / "config.json")
    if fit_normalization_frames:
        dataset = BEVDataset(out, "train", include_radar=False)
        plan["normalization"] = fit_training_normalization(dataset, fit_normalization_frames)
        json_write(out / "manifest.json", plan)
    json_write(out / "label_audit.json", label_report)
    reader.profile.save(out / "native_profile.json")
    return out


def reader_from_prepared(root):
    manifest = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source") != "native_reference":
        raise ValueError("This workflow requires a prepared native-reference dataset")
    options = dict(profile=manifest["native_profile"], trusted_labels=manifest.get("trusted_labels", False),
                   annotation_path=manifest.get("annotation_path"))
    if hasattr(NativeReader, "from_manifest"):
        reader = NativeReader.from_manifest(manifest["native_root"], manifest["dataset"], manifest["native_records"], **options)
    else:
        reader = NativeReader(manifest["native_root"], manifest["dataset"], **options)
    return reader, manifest


def inspect_native(reader, output_dir, max_frames=4, grid=None):
    """Inspect actual tensors and generate physical-coordinate diagnostic figures."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid = default_bev_grid(reader.profile, 128, 256) if grid is None else grid
    report = {"dataset": reader.dataset, "profile": reader.profile.to_dict(), "record_count": len(reader.records),
              "sequences": sorted({r.sequence for r in reader.records}), "frames": [],
              "source_access": "local native files", "sensor_calibration_completed": False}
    indices = np.linspace(0, len(reader.records) - 1, min(max_frames, len(reader.records))).astype(int)
    rng = np.random.default_rng(2024)
    for i in indices:
        record = reader.records[int(i)]
        cube = reader.load_tensor(record)
        power = np.abs(cube) ** 2 if np.iscomplexobj(cube) else cube
        sampled = power.ravel()[rng.choice(power.size, min(50000, power.size), replace=False)]
        ann, error = None, None
        try:
            ann = reader.load_annotations(record)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            error = str(exc)
        report["frames"].append({"id": record.id, "native_path": str(record.path), "canonical_shape": list(cube.shape),
                                 "canonical_dtype": str(cube.dtype), "linear_power_quantiles": np.quantile(sampled, [0, .5, .9, .99, .999, 1]).tolist(),
                                 "car_count": len(ann["boxes_bev"]) if ann is not None else None,
                                 "label_error": error, "label_domain": ann["label_domain"] if ann is not None else None})
        _plot_native(power, reader.profile, grid, ann, out / f"frame_{int(i):06d}.png", record.id)
    json_write(out / "inspection.json", report)
    reader.profile.save(out / "profile_reference.json")
    return report


def _plot_native(power, profile, grid, annotations, output, title, centers=(), noise_regions=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from ..bev.projection import PolarToBEV
    image = np.max(power, axis=1)
    project = PolarToBEV(profile.ranges, profile.angular_grid, grid, profile.azimuth_coordinate)
    with torch.no_grad():
        bev = project(torch.as_tensor(image, dtype=torch.float64)).numpy()
    positive = image[image > 0]
    bounds = np.quantile(10 * np.log10(positive), [.01, .999]) if len(positive) else [-30, 0]
    if bounds[1] <= bounds[0]:
        bounds[1] = bounds[0] + 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].pcolormesh(np.rad2deg(profile.azimuths_rad), profile.ranges, 10 * np.log10(np.maximum(image, 1e-12)),
                       shading="nearest", cmap="magma", vmin=bounds[0], vmax=bounds[1])
    axes[0].set(xlabel="Azimuth (degrees)", ylabel="Range (m)", title="Native power: Doppler maximum")
    for center in centers:
        r, d, a = map(int, center)
        axes[0].plot(np.rad2deg(profile.azimuths_rad[a]), profile.ranges[r], "c+", markersize=12)
    for region in noise_regions:
        r0, r1 = region[0]
        a0, a1 = region[2]
        xx = np.rad2deg(profile.azimuths_rad[[a0, min(a1 - 1, len(profile.azimuths_rad) - 1)]])
        yy = profile.ranges[[r0, min(r1 - 1, len(profile.ranges) - 1)]]
        axes[0].plot([xx[0], xx[1], xx[1], xx[0], xx[0]], [yy[0], yy[0], yy[1], yy[1], yy[0]], "g-")
    axes[1].imshow(10 * np.log10(np.maximum(bev, 1e-12)), origin="lower", extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
                   cmap="magma", vmin=bounds[0], vmax=bounds[1], aspect="equal")
    axes[1].set(xlabel="Lateral x (m)", ylabel="Forward y (m)", title="Cartesian sampling + original labels")
    if annotations is not None:
        for x1, y1, x2, y2 in annotations["boxes_bev"]:
            axes[1].plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "c-", lw=1)
    axes[2].pcolormesh(profile.velocities, profile.ranges, 10 * np.log10(np.maximum(power.max(axis=2), 1e-12)),
                       shading="nearest", cmap="magma", vmin=bounds[0], vmax=bounds[1])
    axes[2].set(xlabel="Radial speed (m/s; positive receding)", ylabel="Range (m)", title="Canonical range–Doppler")
    fig.suptitle(title)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def propose_calibration(prepared_root, output_dir, max_frames=16, half_widths=(4, 2, 10), max_candidates=8):
    """Find bright isolated near-zero-speed candidates; never certify pole identity.

    Lowest-energy blocks are candidate noise regions and can be biased. A user
    must verify real target/noise isolation before calling this a sensor calibration.
    """
    reader, manifest = reader_from_prepared(prepared_root)
    calibration_ids = manifest.get("calibration_record_ids")
    rows = [{"id": value} for value in calibration_ids] if calibration_ids is not None else manifest["splits"]["train"]
    if not rows:
        raise ValueError("No training frames available for calibration candidates")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    half = np.asarray(half_widths, dtype=int)
    if half.shape != (3,) or np.any(half < 0):
        raise ValueError("half_widths must contain three nonnegative integers")
    shape = np.array(reader.profile.shape)
    if np.any(2 * half + 1 > shape):
        raise ValueError("Target support does not fit the tensor dimensions; choose smaller --half-widths")
    centers = np.array(shape // 2)
    zero = int(np.argmin(np.abs(reader.profile.velocities)))
    d_start, d_stop = max(half[1], zero - 1), min(shape[1] - half[1], zero + 2)
    candidates = []
    for i in np.linspace(0, len(rows) - 1, min(max_frames, len(rows))).astype(int):
        record = record_for_id(reader, rows[int(i)]["id"])
        power = reader.load_power(record)
        mask = np.zeros(power.shape, dtype=bool)
        mask[half[0]:shape[0] - half[0] or None, d_start:d_stop,
             half[2]:shape[2] - half[2] or None] = True
        # A narrow local maximum plus distance from the noise floor is a
        # candidate score, not a proof that the object is a point reflector.
        local = maximum_filter(power, size=(max(3, half[0] * 2 + 1), 3, max(3, half[2] * 2 + 1)))
        eligible = np.argwhere(mask & (power == local))
        if not len(eligible):
            continue
        values = power[tuple(eligible.T)]
        order = np.argsort(values)[-3:][::-1]
        noise_shape = np.minimum(shape, np.array([5, 5, 12]))
        energy = uniform_filter(power.astype(float), size=tuple(noise_shape), mode="constant", cval=float(power.max()))
        energy_mask = np.ones(power.shape, dtype=bool)
        for axis in range(3):
            width = int(noise_shape[axis] // 2)
            idx = [slice(None)] * 3
            if width:
                idx[axis] = slice(0, width)
                energy_mask[tuple(idx)] = False
                idx[axis] = slice(shape[axis] - width, None)
                energy_mask[tuple(idx)] = False
        where = np.unravel_index(np.argmin(np.where(energy_mask, energy, np.inf)), power.shape)
        lower = np.maximum(np.array(where) - noise_shape // 2, 0)
        upper = np.minimum(lower + noise_shape, shape)
        noise_bounds = np.column_stack([lower, upper]).astype(int).tolist()
        floor = max(float(np.median(power)), 1e-20)
        for index in order:
            center = eligible[index]
            candidates.append({"record_id": record.id, "center": center.tolist(), "half_widths": half.tolist(),
                               "peak_to_global_median_power": float(values[index] / floor),
                               "noise_bounds_candidate": noise_bounds})
    candidates.sort(key=lambda v: v["peak_to_global_median_power"], reverse=True)
    candidates = candidates[:max_candidates]
    if not candidates:
        raise ValueError("No in-bounds near-zero-Doppler target candidates found; select a verified target manually")
    for i, candidate in enumerate(candidates):
        record = record_for_id(reader, candidate["record_id"])
        power = reader.load_power(record)
        _plot_native(power, reader.profile, default_bev_grid(reader.profile, 128, 256), None,
                     out / f"candidate_{i:02d}.png", f"UNCONFIRMED target candidate {i}: {record.id}",
                     [candidate["center"]], [candidate["noise_bounds_candidate"]])
    best = candidates[0]
    selection = {"coordinate_order": ["range", "doppler", "azimuth"],
        "targets": [{k: best[k] for k in ("record_id", "center", "half_widths")}],
        "noise_regions": [{"record_id": best["record_id"], "bounds": best["noise_bounds_candidate"]}],
        "target_is_isolated": False, "sensor_target_stationary": False, "domain_verified": False,
        "energy_fraction": .99, "note": "Candidate only: verify target identity, add repeated stationary frames and select a genuine reflection-free ROI. Lowest-energy selection can bias noise variance downward."}
    json_write(out / "candidates.json", candidates)
    json_write(out / "selection_template.json", selection)
    return out / "selection_template.json"
