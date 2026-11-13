"""Reusable synthetic native-format fixtures, never real RADDet/CARRADA data.

The controlled radar return is an isolated 3D PSF, not a physically rendered
vehicle. A co-located car annotation exercises file/coordinate interfaces only.
CARRADA annotations remain radar-support regions, not true car footprints.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import pickle
import numpy as np

from radsimreal.native import reference_profile, raddet_reference_cartesian_affine


FIXTURE_NOTICE = "SYNTHETIC NATIVE-FORMAT FIXTURE: not a real radar recording or sensor calibration."


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _to_native(canonical, profile):
    value = np.transpose(canonical, (0, 2, 1))
    if profile.range_reversed:
        value = value[::-1]
    if profile.azimuth_reversed:
        value = value[:, ::-1]
    if profile.canonical_doppler_flip:
        value = value[:, :, ::-1]
    return np.ascontiguousarray(value, dtype=np.complex64)


def create_fixture(base, dataset="raddet", R=32, A=32, D=8):
    """Return (root, profile, training_selection, explicit_split_arguments).

    Writes controlled full tensors, native labels, a declared profile, split and
    selection JSON, and fixture provenance under base/dataset. Source file IDs
    match the published RADDet scene table. Default car centre is (0,31.25)m,
    width 2m and length 4m. CARRADA cannot encode that physical footprint: its
    label file contains declared dense RA support, whose Cartesian enclosure is
    the target used by the adapter. This distinction is preserved in metadata.
    """
    dataset = str(dataset).lower()
    if dataset not in ("raddet", "carrada"):
        raise ValueError("fixture dataset must be raddet or carrada")
    if any(type(n) is not int for n in (R, A, D)) or R < 16 or A < 16 or D < 6:
        raise ValueError("Fixture dimensions require integer R,A>=16 and D>=6")
    root = Path(base) / dataset
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Fixture destination must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    profile = reference_profile(dataset, (R, A, D), "complex")
    profile.range_step = 50.0 / R
    if dataset == "raddet":
        profile.cartesian_affine = raddet_reference_cartesian_affine(profile)
        profile.cartesian_mapping_validated = True  # true for this fixture by construction only
    else:
        profile.annotation_angle_step = np.pi / A  # small-fixture grid, not a changed claim about public A=256
    profile.evidence.append(FIXTURE_NOTICE)
    profile.unresolved.append("These convention checks use controlled fixtures; no real recording was validated")
    zero_velocity = int(np.argmin(np.abs(profile.velocities)))
    center = np.array([min(max(int(round(.625 * R)), 3), R - 4), zero_velocity, A // 2])
    half = np.array([2, 1, 3])
    rr, dd, aa = np.meshgrid(np.arange(-2, 3), np.arange(-1, 2), np.arange(-3, 4), indexing="ij")
    psf = np.exp(-.5 * ((rr / .65) ** 2 + (dd / .55) ** 2 + (aa / 1.0) ** 2))
    covariance = np.array([[.0009, .0003], [.0003, .0016]])
    mean = np.array([.005, .01])
    truth = {"center_x_m": 0.0, "center_y_m": float(profile.ranges[center[0]]),
             "width_m": 2.0, "length_m": 4.0}
    physical_box = [-1.0, truth["center_y_m"] - 2.0, 1.0, truth["center_y_m"] + 2.0]
    if dataset == "raddet":
        plan = [("scene_0", 0, "val"), ("scene_0", 1, "val"),
                ("scene_1", 440, "train"), ("scene_1", 441, "train"), ("scene_1", 442, "train"),
                ("scene_9", 6609, "test"), ("scene_11", 8635, "test")]
        split_args = {"val_sequences": ["scene_0"]}
        split_json = {"train": ["scene_1"], "val": ["scene_0"], "test": ["scene_9", "scene_11"]}
    else:
        plan = [(sequence, frame, split) for sequence, split in
                (("train_seq", "train"), ("val_seq", "val"), ("test_seq", "test")) for frame in range(3)]
        split_args = {"val_sequences": ["val_seq"], "test_sequences": ["test_seq"]}
        split_json = {"train": ["train_seq"], "val": ["val_seq"], "test": ["test_seq"]}
    rng = np.random.default_rng(20241110)
    selection = {"coordinate_order": ["range", "doppler", "azimuth"], "targets": [], "noise_regions": [],
                 "target_is_isolated": True, "sensor_target_stationary": True,
                 "domain_verified": True, "energy_fraction": .99, "fixture_notice": FIXTURE_NOTICE}
    carrada_annotations = {}
    record_table = []
    for sample_number, (sequence, frame, split) in enumerate(plan):
        quadratures = rng.multivariate_normal(mean, covariance, size=profile.shape)
        canonical = quadratures[..., 0] + 1j * quadratures[..., 1]
        crop = tuple(slice(int(c - h), int(c + h + 1)) for c, h in zip(center, half))
        canonical[crop] += 8.0 * np.exp(1j * .01 * sample_number) * psf
        if dataset == "raddet":
            original_subset = "test" if split == "test" else "train"
            folder = root / original_subset / "RAD/part1"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{frame:06d}.npy"
            identifier = f"raddet/{frame:06d}"
            affine = profile.cartesian_affine
            native_row = (truth["center_y_m"] - affine["y_from_row_offset"]) / affine["y_from_row_scale"]
            native_col = (truth["center_x_m"] - affine["x_from_col_offset"]) / affine["x_from_col_scale"]
            cart = [float(native_row), float(native_col), truth["length_m"] / profile.range_step,
                    truth["width_m"] / profile.range_step]
            native_r = R - 1 - center[0] if profile.range_reversed else center[0]
            native_a = A - 1 - center[2] if profile.azimuth_reversed else center[2]
            native_d = D - 1 - center[1] if profile.canonical_doppler_flip else center[1]
            rad = [int(native_r), int(native_a), int(native_d),
                   truth["length_m"] / profile.range_step,
                   max(1.0, truth["width_m"] / max(truth["center_y_m"] * profile.angular_step, 1e-6)), 1.5]
            label_path = root / original_subset / "gt/part1" / f"{frame:06d}.pickle"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            with label_path.open("wb") as stream:
                pickle.dump({"classes": ["car"], "boxes": [rad], "cart_boxes": [cart]}, stream,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            folder = root / sequence / "RAD_numpy"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{frame:06d}.npy"
            identifier = f"carrada/{sequence}/{frame:06d}"
            annotation_row = R - int(center[0])
            annotation_col = int(center[2])
            dense = [[annotation_row + r, annotation_col + a] for r in (-1, 0, 1) for a in (-1, 0, 1)]
            carrada_annotations.setdefault(sequence, {})[f"{frame:06d}"] = {
                "car_0": {"range_angle": {"label": 3, "box": [[annotation_row - 1, annotation_col - 1],
                                                             [annotation_row + 1, annotation_col + 1]],
                                             "sparse": [[annotation_row, annotation_col]], "dense": dense},
                          "range_doppler": {"label": 3,
                            "box": [[int(center[0]) - 1, D // 2 - 1], [int(center[0]) + 1, D // 2 + 1]],
                            "sparse": [[int(center[0]), D // 2]],
                            "dense": [[int(center[0]) + r, D // 2 + d] for r in (-1, 0, 1) for d in (-1, 0, 1)]}}}
        np.save(path, _to_native(canonical, profile), allow_pickle=False)
        if split == "train":
            selection["targets"].append({"record_id": identifier, "center": center.tolist(),
                                          "half_widths": half.tolist()})
            selection["noise_regions"].append({"record_id": identifier,
                 "bounds": [[2, 6], [0, 2], [0, 5]]})
        record_table.append({"id": identifier, "sequence": sequence, "split": split,
                             "canonical_reflector_center": center.tolist()})
    if dataset == "carrada":
        _write_json(root / "annotations_frame_oriented.json", carrada_annotations)
        _write_json(root / "data_seq_ref.json", {sequence: {"instances": ["car_0"], "labels": [3]}
                                                     for sequence in carrada_annotations})
    profile.save(root / "fixture_profile.json")
    _write_json(root / "fixture_selection.json", selection)
    _write_json(root / "fixture_split.json", split_json)
    _write_json(root / "FIXTURE_PROVENANCE.json", {"notice": FIXTURE_NOTICE,
        "radar_return": "single isolated Gaussian 3D PSF plus non-circular complex Gaussian IID noise",
        "canonical_order": ["range", "doppler", "azimuth"], "shape_native": [R, A, D],
        "reflector_center": center.tolist(), "noise_mean": mean.tolist(), "noise_covariance": covariance.tolist(),
        "reference_car_geometry": truth, "reference_car_box_bev": physical_box,
        "annotation_caveat": "CARRADA stores radar annotation regions; reference car geometry is not recoverable from those labels",
        "car_and_reflector_caveat": "Car annotation co-location exercises IO only; return is not a physically rendered car",
        "records": record_table})
    (root / "SYNTHETIC_NATIVE_FIXTURE_README.txt").write_text(FIXTURE_NOTICE + "\n"
        "Controlled file-format/calibration/detection IO fixture. No recording, measured PSF,\n"
        "real annotation, author synthetic scene or published benchmark result is included.\n",
        encoding="utf-8")
    return root, profile, selection, split_args


# Convenient spelling for interactive use while preserving the documented API.
createfixture = create_fixture
