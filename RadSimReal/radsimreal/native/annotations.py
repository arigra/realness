"""Native labels with explicit geometric domain and no invented object yaw.

RADDet Cartesian labels are retained before any physical-coordinate affine.
CARRADA converted regions describe radar annotation support, not car footprints.
"""
from __future__ import annotations

import json
import pickle
import numpy as np


def _boxes(array, columns):
    array = np.asarray(array, dtype=np.float64).reshape(-1, columns)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite native annotation coordinates")
    return array


def _is_car(value, allowed):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return value.lower() in {str(x).lower() for x in allowed}
    return value in allowed


def _base(profile, source):
    return {"boxes_bev": np.empty((0, 4), dtype=np.float64),
            "boxes_cartesian_native": np.empty((0, 4), dtype=np.float64),
            "boxes_ra": np.empty((0, 4), dtype=np.float64),
            "oriented_boxes_bev": np.empty((0, 5), dtype=np.float64),
            "labels": [], "source": source,
            "coordinates_bev": "x lateral right, y forward, metres; axis-aligned XYXY",
            "coordinates_ra": "canonical azimuth-column/range-row bin-centre XYXY, not normalized",
            "orientation_labels_available": False,
            "cartesian_mapping_validated": bool(profile.cartesian_mapping_validated)}


def native_ra_boxes_to_canonical(boxes_row_col_xywh, profile):
    """Native [range-centre,angle-centre,range-size,angle-size] to canonical XYXY."""
    raw = _boxes(boxes_row_col_xywh, 4)
    if len(raw) and np.any(raw[:, 2:] <= 0):
        raise ValueError("Native RA sizes must be positive")
    row1, row2 = raw[:, 0] - raw[:, 2] / 2, raw[:, 0] + raw[:, 2] / 2
    col1, col2 = raw[:, 1] - raw[:, 3] / 2, raw[:, 1] + raw[:, 3] / 2
    if profile.range_reversed:
        row1, row2 = profile.shape_native[0] - 1 - row2, profile.shape_native[0] - 1 - row1
    if profile.azimuth_reversed:
        col1, col2 = profile.shape_native[1] - 1 - col2, profile.shape_native[1] - 1 - col1
    return np.stack([col1, row1, col2, row2], axis=-1)


def _cartesian_affine(boxes, affine):
    if affine is None:
        return np.empty((0, 4), dtype=float)
    xy1 = boxes[:, :2] * [affine["x_from_col_scale"], affine["y_from_row_scale"]] + [affine["x_from_col_offset"], affine["y_from_row_offset"]]
    xy2 = boxes[:, 2:] * [affine["x_from_col_scale"], affine["y_from_row_scale"]] + [affine["x_from_col_offset"], affine["y_from_row_offset"]]
    return np.concatenate([np.minimum(xy1, xy2), np.maximum(xy1, xy2)], axis=1)


def load_raddet_annotations(record, profile, trusted_labels=False):
    """Decode public trusted RADDet pickle and preserve its Cartesian rectangles.

    The opt-in is required because Python pickle can execute arbitrary code.
    It should only be enabled for the user's trusted original dataset files.
    No arbitrary pickle content is copied into project metadata/checkpoints.
    """
    if not trusted_labels:
        raise ValueError("RADDet annotations are pickle files; set trusted_labels=True only for your trusted original dataset")
    if record.annotation_path is None or not record.annotation_path.is_file():
        raise FileNotFoundError(f"Missing native RADDet label file for {record.id}")
    with record.annotation_path.open("rb") as stream:
        native = pickle.load(stream)
    if not isinstance(native, dict) or "classes" not in native:
        raise ValueError("RADDet annotation must be a dict containing classes")
    classes = list(native["classes"])
    if "cart_boxes" not in native:
        raise ValueError("RADDet cart_boxes are missing; a polar enclosing rectangle is not a replacement for native Cartesian ground truth")
    cart = _boxes(native["cart_boxes"], 4)
    rad = _boxes(native.get("boxes", []), 6)
    if len(cart) != len(classes) or (len(rad) and len(rad) != len(classes)):
        raise ValueError("Native RADDet classes and annotation arrays have inconsistent lengths")
    if len(cart) and np.any(cart[:, 2:] <= 0):
        raise ValueError("RADDet Cartesian dimensions must be positive")
    keep = np.asarray([_is_car(label, profile.car_labels) for label in classes], dtype=bool)
    selected = cart[keep]
    # Public native format is [row-centre,col-centre,row-size,col-size].
    # XYXY below explicitly uses lateral-grid col as x and forward-grid row as y.
    center = selected[:, [1, 0]]
    size = selected[:, [3, 2]]
    native_xyxy = np.concatenate([center - size / 2, center + size / 2], axis=1)
    result = _base(profile, str(record.annotation_path))
    result.update(boxes_cartesian_native=native_xyxy,
                  boxes_bev=_cartesian_affine(native_xyxy, profile.cartesian_affine),
                  labels=["car"] * len(selected),
                  label_domain="native_RADDet_Cartesian_rectangles",
                  native_cart_boxes_row_col_xywh=selected.tolist(),
                  physical_affine_enabled=profile.cartesian_affine is not None,
                  requires_cartesian_affine=profile.cartesian_affine is None,
                  geometry_approximation=False,
                  note="Cartesian-bin IoU is preserved by any shared invertible per-axis affine. Physical placement must be validated separately.")
    if len(rad):
        selected_rad = rad[keep]
        result["boxes_ra"] = native_ra_boxes_to_canonical(selected_rad[:, [0, 1, 3, 4]], profile)
        result["native_rad_boxes_center_size"] = selected_rad.tolist()
    return result


def polar_region_enclosure(r_min, r_max, theta_min, theta_max):
    """Exact axis-aligned enclosure of a continuous annular sector in metres.

    Include trigonometric stationary angles, not just the four sector corners.
    This computes a radar-region enclosure and does not infer vehicle shape.
    """
    if not np.isfinite([r_min, r_max, theta_min, theta_max]).all() or not 0 <= r_min <= r_max or theta_min > theta_max:
        raise ValueError("Invalid polar annotation region")
    angles = [theta_min, theta_max]
    for k in range(-4, 5):
        stationary = k * np.pi / 2
        if theta_min <= stationary <= theta_max:
            angles.append(stationary)
    angles = np.asarray(angles)
    points = np.concatenate([np.stack([r * np.sin(angles), r * np.cos(angles)], axis=1) for r in (r_min, r_max)])
    return np.r_[points.min(axis=0), points.max(axis=0)]


def _carrada_angles_from_columns(col, profile):
    """Map annotation columns consistently for point and box-only support."""
    if profile.annotation_angle_step is not None:
        return (col - profile.shape_native[1] // 2) * profile.annotation_angle_step
    canonical_col = profile.shape_native[1] - 1 - col if profile.azimuth_reversed else col
    angular = profile.angular_min + canonical_col * profile.angular_step
    return np.arcsin(np.clip(angular, -1, 1)) if profile.azimuth_coordinate == "sin" else angular


def _carrada_points_to_metres(points, profile):
    points = _boxes(points, 2)
    row, col = points[:, 0], points[:, 1]
    r = profile.shape_native[0] - row if profile.annotation_range_reverse_origin == "R" else profile.shape_native[0] - 1 - row
    r = profile.range_min + np.maximum(r, 0) * profile.range_step
    angle = _carrada_angles_from_columns(col, profile)
    return np.stack([r * np.sin(angle), r * np.cos(angle)], axis=1)


def load_carrada_annotations(record, profile, frame_oriented):
    """Use native CARRADA per-frame radar annotation support; no yaw invented.

    Complete dense/sparse RA point support is transformed where supplied. If
    only a box exists, the continuous annular-sector enclosure is explicit.
    Native row inversion is reproduced from transform_annotations.py (R-row),
    including its one-bin distinction from the tensor's R-1-row reversal.
    """
    if record.sequence not in frame_oriented:
        raise KeyError(f"CARRADA sequence {record.sequence!r} not present in frame-oriented annotations")
    sequence = frame_oriented[record.sequence]
    if not isinstance(sequence, dict):
        raise ValueError("CARRADA sequence annotations must map frames to instance dictionaries")
    key = record.path.stem
    if key not in sequence and record.frame_id is not None and str(record.frame_id) in sequence:
        key = str(record.frame_id)
    if key not in sequence:
        raise KeyError(f"CARRADA frame {record.id!r} missing from annotations; missing is not an empty scene")
    frame = sequence[key]
    if not isinstance(frame, dict):
        raise ValueError("CARRADA frame annotations must map instance IDs to views")
    result = _base(profile, str(record.annotation_path))
    bev, ra_boxes, instances, provenance = [], [], [], []
    for instance, views in frame.items():
        if not isinstance(views, dict):
            raise ValueError(f"CARRADA instance {instance!r} must map radar views to annotations")
        if "range_angle" not in views:
            raise ValueError(f"CARRADA instance {instance!r} lacks range_angle annotations required for Cartesian conversion")
        ra = views["range_angle"]
        if not isinstance(ra, dict):
            raise ValueError(f"CARRADA range_angle annotation for {instance!r} must be an object")
        label = ra.get("label", views.get("label"))
        if label is None:
            raise ValueError(f"CARRADA instance {instance!r} lacks an explicit label")
        if not _is_car(label, profile.car_labels):
            continue
        raw_box = _boxes(ra.get("box", []), 2)
        if raw_box.shape != (2, 2):
            raise ValueError("CARRADA range_angle box must contain two [row,col] corners")
        lower, upper = raw_box.min(axis=0), raw_box.max(axis=0)
        if np.any(upper <= lower):
            raise ValueError("CARRADA native radar box must have positive range/angle extents")
        row_col_center_size = np.r_[(lower + upper) / 2, upper - lower][None]
        ra_boxes.append(native_ra_boxes_to_canonical(row_col_center_size, profile)[0])
        support = ra.get("dense") or ra.get("sparse")
        if support is not None and len(support):
            points = _carrada_points_to_metres(support, profile)
            enclosure = np.r_[points.min(axis=0), points.max(axis=0)]
            method = "Cartesian enclosure of native dense/sparse RA annotation support"
        else:
            # The four transformed box corners can miss a sector's forward
            # extremum near zero azimuth. Use the exact sector enclosure.
            origin = profile.shape_native[0] if profile.annotation_range_reverse_origin == "R" else profile.shape_native[0] - 1
            radii = profile.range_min + np.maximum(origin - raw_box[:, 0], 0) * profile.range_step
            theta = _carrada_angles_from_columns(raw_box[:, 1], profile)
            enclosure = polar_region_enclosure(float(radii.min()), float(radii.max()), float(theta.min()), float(theta.max()))
            method = "Cartesian enclosure of complete annular sector from native RA box"
        if np.any(enclosure[2:] <= enclosure[:2]):
            raise ValueError("CARRADA annotation support gives a degenerate Cartesian enclosure")
        bev.append(enclosure)
        instances.append(str(instance))
        provenance.append({"instance_id": str(instance), "native_label": label, "conversion": method})
    result.update(boxes_bev=np.asarray(bev, dtype=float).reshape(-1, 4),
                  boxes_ra=np.asarray(ra_boxes, dtype=float).reshape(-1, 4), labels=["car"] * len(bev),
                  instance_ids=instances, conversion_provenance=provenance,
                  label_domain="Cartesian_enclosure_of_CARRADA_radar_annotation_support",
                  geometry_approximation=True,
                  note="Enclosures are radar-return regions, not physical vehicle footprints or oriented boxes; RadSimReal exact annotation conversion was not supplied.")
    return result


def load_json_sequence(path, target):
    """Read one top-level sequence without retaining the entire dense JSON file.

    Values are decoded one sequence at a time using the standard JSON decoder.
    Memory is bounded by the largest decoded sequence, not the complete dataset.
    """
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as stream:
        buffer, position, eof = "", 0, False

        def fill():
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = stream.read(1024 * 1024)
            eof = not chunk
            buffer += chunk

        def char():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    return buffer[position]
                if eof:
                    raise ValueError("Unexpected end of CARRADA annotation JSON")
                fill()

        def decode():
            nonlocal position
            char()
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    return value
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()

        if char() != "{":
            raise ValueError("Frame-oriented CARRADA JSON must be an object")
        position += 1
        if char() == "}":
            raise KeyError(f"CARRADA sequence {target!r} not present in annotations")
        while True:
            key = decode()
            if not isinstance(key, str) or char() != ":":
                raise ValueError("Invalid top-level CARRADA sequence mapping")
            position += 1
            value = decode()
            if key == target:
                if not isinstance(value, dict):
                    raise ValueError("CARRADA sequence annotations must be an object")
                return value
            separator = char()
            if separator == "}":
                raise KeyError(f"CARRADA sequence {target!r} not present in annotations")
            if separator != ",":
                raise ValueError("Invalid separator in CARRADA annotation JSON")
            position += 1
