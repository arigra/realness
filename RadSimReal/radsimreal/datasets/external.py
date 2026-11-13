"""Explicit import of real arrays, with source transforms and split provenance."""
import hashlib
import json
from pathlib import Path
import numpy as np
from ..config import save_config


def _convert(record, source, radar):
    path = (source.parent / record["image_path"]).resolve()
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        with data:
            array = np.array(data[record.get("array_key", "image")])
    else:
        array = data
    source_shape, source_dtype = list(array.shape), str(array.dtype)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("Radar arrays must have a numerical dtype")
    # Convert BEFORE squaring: integer products otherwise silently overflow.
    array = array.astype(np.complex128 if np.iscomplexobj(array) else np.float64)
    representation = record["representation"]
    with np.errstate(over="raise", invalid="raise"):
        try:
            if representation == "amplitude":
                array = np.abs(array) ** 2
            elif representation == "db":
                if np.iscomplexobj(array):
                    raise ValueError("dB arrays must be real")
                array = 10.0 ** (array / 10.0)
            elif representation != "power":
                raise ValueError("representation must be power, amplitude or db")
        except FloatingPointError as exc:
            raise ValueError("Input overflows during conversion to linear power") from exc
    if np.iscomplexobj(array) or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("Power array must be finite, real and nonnegative")
    if array.ndim == 3:
        if array.shape != radar.shape:
            raise ValueError("Tensor shape must match range,doppler,azimuth configuration")
        array = np.max(array, axis=1)
    if array.shape != (radar.range_bins, radar.azimuth_bins):
        raise ValueError("Image shape does not match declared sensor grid")
    if np.any(array > np.finfo(np.float32).max):
        raise ValueError("Power values exceed float32 storage range; supply explicit calibrated rescaling")
    array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite values after float32 conversion")
    checksum = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    provenance = {"source_file": str(path), "array_key": record.get("array_key", "image"),
                  "source_shape": source_shape, "source_dtype": source_dtype,
                  "source_representation": representation,
                  "transform": "float64 conversion; amplitude squared or 10**(dB/10) if declared; Doppler max if 3D; float32",
                  "linear_image_sha256": checksum, "user_notes": record.get("provenance", "")}
    return array, provenance


def import_manifest(source_jsonl, output_dir, config):
    """Import docs/DATA_FORMAT.md records without guessed sensor conventions.

    Preflight checks all records/arrays, scene separation and identical-image
    leakage before writing. Arrays are read twice to avoid retaining a complete
    large dataset in RAM. Source changes between those reads are rejected.
    """
    source = Path(source_jsonl).resolve()
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Dataset destination must be empty: {out}")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    scenes, seen_ids, contents = {}, set(), {}
    prepared = []
    for record in records:
        sample_id = str(record["id"])
        if sample_id in seen_ids or not sample_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in sample_id):
            raise ValueError(f"Duplicate or unsafe sample id {sample_id!r}")
        seen_ids.add(sample_id)
        split = record["split"]
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        scene = str(record["scene_id"])
        if scene in scenes and scenes[scene] != split:
            raise ValueError(f"Scene {scene} occurs across splits")
        scenes[scene] = split
        boxes = np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 4)
        if len(boxes) and (not np.isfinite(boxes).all() or np.any(boxes < 0) or np.any(boxes > 1) or np.any(boxes[:, 2:] <= boxes[:, :2])):
            raise ValueError("Invalid normalized RA XYXY boxes")
        _, provenance = _convert(record, source, config.radar)
        checksum = provenance["linear_image_sha256"]
        if checksum in contents and contents[checksum] != split:
            raise ValueError("Identical radar image occurs across splits, even if scene IDs differ")
        contents[checksum] = split
        prepared.append((record, boxes, provenance))
    out.mkdir(parents=True, exist_ok=True)
    save_config(config, out / "config.json")
    manifest = {"format_version": 1, "source": "user_supplied_real_data",
                "axis_order": ["range", "doppler", "azimuth"], "box_space": "normalized_ra_xyxy",
                "metric_domain": "range_and_declared_azimuth_bin_area_not_cartesian_BEV",
                "splits": {k: [] for k in ("train", "val", "test")}}
    for record, boxes, provenance in prepared:
        array, current = _convert(record, source, config.radar)
        if current["linear_image_sha256"] != provenance["linear_image_sha256"]:
            raise ValueError("Source array changed during import")
        split, name = record["split"], str(record["id"])
        (out / split).mkdir(exist_ok=True)
        image_path, annotation_path = f"{split}/{name}.npz", f"{split}/{name}.json"
        np.savez_compressed(out / image_path, image=array)
        annotation = {"id": name, "scene_id": str(record["scene_id"]),
                      "objects": [{"label": "car", "box": box.tolist()} for box in boxes],
                      "provenance": provenance}
        (out / annotation_path).write_text(json.dumps(annotation, indent=2), encoding="utf-8")
        manifest["splits"][split].append({"id": name, "image": image_path, "annotation": annotation_path})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out
