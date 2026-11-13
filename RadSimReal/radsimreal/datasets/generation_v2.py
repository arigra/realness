"""Calibrated tensor generation with original Cartesian geometric annotations."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import numpy as np
from ..config import load_config, save_config
from ..radar import simulate
from ..radar.coordinates import points_to_indices
from ..simulation import sample_scene, generate_reflections
from ..simulation.mesh import mesh_from_scene, trace_reflections, merge_meshes, load_mesh, import_obj
from .synthetic import calibrated_psf, scene_annotations
from .bev_dataset import DEFAULT_BEV_NORMALIZATION
from ..native.calibration import json_write, record_for_id


def _scene_reflections(scene, config, rng, scene_backend="mesh", samples_per_square_meter=4,
                       max_samples=20000, static_mesh=None, vehicle_assets=None):
    if scene_backend == "procedural":
        return generate_reflections(scene, config.radar, config.scene, rng)
    if scene_backend != "mesh":
        raise ValueError("scene_backend must be mesh or procedural")
    if vehicle_assets is None:
        mesh = mesh_from_scene(scene, sensor_position=(0, 0, config.radar.sensor_height))
    else:
        from ..simulation.assets import mesh_from_assets
        mesh = mesh_from_assets(scene, vehicle_assets, rng, sensor_position=(0, 0, config.radar.sensor_height))
    if static_mesh is not None:
        mesh = merge_meshes([mesh, static_mesh])
    return trace_reflections(mesh, config.radar, rng=rng,
                              samples_per_square_meter=samples_per_square_meter, max_samples=max_samples,
                              frequency_hz=config.radar.frequency_hz, polarization=config.radar.polarization,
                              phase_mode=config.radar.phase_mode)


def _geometry_annotations(scene, reflections, radar, grid, visible_only=True, psf=None):
    # Keep v1 radar-image labels for inspection, but all BEV targets originate
    # from the original ground-plane geometry, never the radar rectangle.
    ra = {obj["object_id"]: obj for obj in scene_annotations(scene, radar, reflections)}
    indices, valid = points_to_indices(reflections.xyz, reflections.velocity, radar)
    valid &= np.abs(reflections.coefficients) > 0
    annotations = []
    for obj in scene.objects:
        count = int(np.sum(valid & (reflections.object_ids == obj.object_id)))
        if visible_only and count == 0:
            continue
        corners = np.asarray(obj.corners())[:, :2]
        box = np.r_[corners.min(0), corners.max(0)]
        normalized = grid.normalize_boxes(box)
        if np.any(normalized < -1e-7) or np.any(normalized > 1 + 1e-7):
            raise ValueError("Scene footprint leaves the BEV grid; enlarge grid or narrow scene extent")
        record = dict(ra.get(obj.object_id, {}))
        record.update(object_id=int(obj.object_id), label="car", box_bev=box.tolist(),
                      oriented_box_bev=[obj.x, obj.y, obj.width, obj.length, -obj.yaw],
                      oriented_convention="CCW rotation of width axis from +x; metres",
                      label_scope="original_vehicle_ground_footprint",
                      in_domain_reflection_count=count, has_simulated_return=bool(count))
        object_cells = indices[valid & (reflections.object_ids == obj.object_id)]
        if len(object_cells):
            # Canonical R,A,D normalized center+extent targets for an optional
            # synthetic RADDet pretraining stage; retain this chosen RF support
            # policy instead of presenting it as the authors' label generator.
            from ..native.export import _canonical_support_box
            radius = np.array(psf.shape) // 2 if psf is not None else np.zeros(3, dtype=int)
            center, size = _canonical_support_box(object_cells, radius, radar.shape, radar.doppler_wrap)
            norm = np.asarray(radar.shape, dtype=float)
            record["box_rad"] = np.r_[center[[0, 2, 1]] / norm[[0, 2, 1]], size[[0, 2, 1]] / norm[[0, 2, 1]]].tolist()
            record["box_rad_provenance"] = "canonical truncated-PSF-expanded RF support; normalized R,A,D center/extent; full-D box for periodic boundary crossings"
        annotations.append(record)
    return annotations


def materialize_sensor(config, output_dir):
    """Copy PSF/noise products so a generated dataset remains self-contained."""
    config = deepcopy(config)
    out = Path(output_dir)
    for field, filename in (("psf_path", "calibration.npz"), ("noise_bank_path", "noise_bank.npz"),
                            ("noise_spectrum_path", "noise_spectrum.npz")):
        original = getattr(config.radar, field)
        if original:
            src = Path(original).resolve()
            dest = out / filename
            if src != dest.resolve():
                shutil.copy2(src, dest)
            setattr(config.radar, field, filename)
    save_config(config, out / "config.json")
    return config


def generate_bev_dataset(output_dir, config, counts=None, *, grid=None, normalization=None,
                         scene_backend="mesh", visible_only=True, samples_per_square_meter=4,
                         max_samples=20000, static_mesh_path=None, native_profile=None,
                         save_tensors=True, seed=None, vehicle_assets_path=None):
    from ..native.workflows import default_bev_grid
    from ..native.profiles import reference_profile
    from ..bev.geometry import BEVGrid
    if isinstance(config, (str, Path)):
        config = load_config(config)
    config = deepcopy(config)
    if native_profile is not None:
        from ..native.workflows import assert_sensor_sampling
        assert_sensor_sampling(config.radar, native_profile)
    if seed is not None:
        config.seed = int(seed)
    counts = {"train": 64, "val": 16, "test": 16} if counts is None else dict(counts)
    if set(counts) != {"train", "val", "test"} or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("counts must contain nonnegative integer train, val and test sizes")
    if grid is None:
        extent = float(config.radar.ranges[-1]) + config.radar.range_step
        grid = BEVGrid(x_min=-extent, x_max=extent, y_min=0, y_max=extent,
                       height=config.radar.range_bins, width=2 * config.radar.range_bins)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Generated BEV dataset output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    psf = calibrated_psf(config.radar)
    materialize_sensor(config, out)
    static = None
    library = None
    if vehicle_assets_path:
        if scene_backend != "mesh":
            raise ValueError("Vehicle CAD assets require the mesh backend")
        from ..simulation.assets import load_vehicle_assets
        library = load_vehicle_assets(vehicle_assets_path)
    if static_mesh_path:
        static_mesh_path = Path(static_mesh_path)
        static = import_obj(static_mesh_path) if static_mesh_path.suffix.lower() == ".obj" else load_mesh(static_mesh_path)
        original_ids = np.unique(static.object_ids)
        id_map = {int(value): -i - 1 for i, value in enumerate(original_ids)}
        static.object_ids = np.asarray([id_map[int(value)] for value in static.object_ids], dtype=np.int64)
        static.metadata["static_object_id_mapping"] = {str(k): v for k, v in id_map.items()}
        static.metadata["object_id_policy"] = "negative static IDs cannot collide with nonnegative generated vehicle IDs"
    writer = None
    if native_profile is not None:
        from ..native.export import ExportNativeWriter
        writer = ExportNativeWriter(out / "native_export", native_profile)
    manifest = {"format_version": 2, "source": "generated_original_Cartesian_geometry",
                "axis_order": ["range", "doppler", "azimuth"], "bev_grid": grid.to_dict(),
                "normalization": dict(DEFAULT_BEV_NORMALIZATION if normalization is None else normalization),
                "label_domain": "synthetic_original_vehicle_ground_footprints",
                "metric_domain": "cartesian_bev_metres_axis_aligned",
                "oriented_labels": "true procedural geometry; yaw CCW of width axis from +x",
                "boundary_policy": "reject_outside_grid_no_silent_label_crop",
                "visibility_policy": "only objects with in-domain simulated returns" if visible_only else "all geometric objects",
                "scene_backend": scene_backend, "published_result_reproduced": False, "splits": {}}
    if vehicle_assets_path:
        manifest["vehicle_asset_manifest"] = str(Path(vehicle_assets_path).resolve())
    if native_profile is not None:
        manifest["native_profile"] = native_profile.to_dict()
    number = 0
    for split_number, (split, count) in enumerate(counts.items()):
        (out / split).mkdir()
        rows = []
        for i in range(count):
            rng = np.random.default_rng(np.random.SeedSequence([config.seed, split_number, i]))
            scene = sample_scene(config.radar, config.scene, rng)
            reflections = _scene_reflections(scene, config, rng, scene_backend,
                                              samples_per_square_meter, max_samples, static, library)
            result = simulate(reflections, config.radar, psf, rng)
            objects = _geometry_annotations(scene, reflections, config.radar, grid, visible_only, result.psf)
            name = f"{split}_{i:06d}"
            image_path, annotation_path = f"{split}/{name}.npz", f"{split}/{name}.json"
            arrays = {"image": result.image}
            if save_tensors:
                cube = result.tensor
                max_value = max(float(np.max(np.abs(cube.real))), float(np.max(np.abs(cube.imag))) if np.iscomplexobj(cube) else 0.)
                if max_value > np.finfo(np.float32).max:
                    raise ValueError("Generated tensor exceeds float32/complex64 storage range")
                arrays["tensor"] = cube.astype(np.complex64 if np.iscomplexobj(cube) else np.float32)
            np.savez_compressed(out / image_path, **arrays)
            json_write(out / annotation_path, {"id": name, "objects": objects, "scene": scene.to_dict(),
                                              "simulation": result.metadata, "reflection_metadata": reflections.metadata,
                                              "seed_sequence": [config.seed, split_number, i]})
            row = {"id": name, "image": image_path, "annotation": annotation_path}
            if writer is not None:
                row["native_export"] = writer.write_frame(f"{number:06d}", split, scene, reflections, result)
            rows.append(row)
            number += 1
        manifest["splits"][split] = rows
    if writer is not None:
        writer.finalize()
        manifest["synthetic_label_origin"] = manifest["label_domain"]
        if native_profile.name == "raddet":
            # The exported Cartesian labels are an invertible affine of these
            # same original footprints; no polar-box inverse is involved.
            manifest["label_domain"] = "native_RADDet_Cartesian_rectangles"
        else:
            # Match the public CARRADA annotation-region convention through the
            # same native parser used for real data. Keep original geometry as
            # separate provenance; radar support has no invented yaw target.
            from ..native.readers import NativeReader
            reader = NativeReader(out / "native_export", "carrada", profile=native_profile)
            by_path = {str(record.path.resolve()): record for record in reader.records}
            for rows in manifest["splits"].values():
                for row in rows:
                    record = by_path[str((out / "native_export" / row["native_export"]["tensor"]).resolve())]
                    native_labels = reader.load_annotations(record)
                    annotation_path = out / row["annotation"]
                    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                    originals = annotation["objects"]
                    original_by_id = {int(o["object_id"]): o for o in originals}
                    objects = []
                    for i, (box, instance) in enumerate(zip(native_labels["boxes_bev"], native_labels["instance_ids"])):
                        object_id = int(instance.removeprefix("car_"))
                        original = original_by_id.get(object_id, {})
                        obj = {"object_id": object_id, "label": "car", "box_bev": box.tolist(),
                               "label_scope": "PSF-expanded synthetic RF support; Cartesian enclosure via CARRADA native parser",
                               "has_simulated_return": True}
                        if "box_rad" in original:
                            obj["box_rad"] = original["box_rad"]
                            obj["box_rad_provenance"] = original["box_rad_provenance"]
                        objects.append(obj)
                    annotation["original_geometry_objects"] = originals
                    annotation["objects"] = objects
                    annotation["label_conversion"] = native_labels["conversion_provenance"]
                    json_write(annotation_path, annotation)
            manifest["label_domain"] = "Cartesian_enclosure_of_CARRADA_radar_annotation_support"
            manifest["oriented_labels"] = "unavailable for radar-support targets; original geometry retained separately"
    json_write(out / "manifest.json", manifest)
    return out


def fit_relative_scattering_gain(prepared_root, sensor_config, output_path, *, real_frames=16,
                                  synthetic_frames=4, scene_backend="mesh", samples_per_square_meter=4,
                                  max_samples=10000, quantile=.995, vehicle_assets_path=None):
    """Unlabeled TRAIN-only empirical power-quantile gain fit, not absolute RCS.

    A global gain cannot repair incorrect scene/material distributions. This is
    an explicit optional calibration choice beyond the paper's disclosed details.
    """
    from ..native.workflows import reader_from_prepared
    reader, manifest = reader_from_prepared(prepared_root)
    rows = manifest["splits"]["train"]
    if not rows or real_frames < 1 or synthetic_frames < 1 or not 0 < quantile < 1:
        raise ValueError("Gain fitting requires training frames, positive sample counts and a valid quantile")
    config = load_config(sensor_config)
    from ..native.workflows import assert_sensor_sampling
    assert_sensor_sampling(config.radar, reader.profile)
    cfg = deepcopy(config)
    cfg.radar.noise_model = "none"
    cfg.radar.noise_std = 0
    cfg.radar.scattering_gain = 1
    psf = calibrated_psf(cfg.radar)
    library = None
    if vehicle_assets_path:
        from ..simulation.assets import load_vehicle_assets
        library = load_vehicle_assets(vehicle_assets_path)
    real_statistics = []
    for i in np.linspace(0, len(rows) - 1, min(real_frames, len(rows))).astype(int):
        power = reader.load_power(record_for_id(reader, rows[int(i)]["id"]))
        real_statistics.append(float(np.quantile(power.max(axis=1), quantile)))
    synthetic_statistics = []
    for i in range(synthetic_frames):
        rng = np.random.default_rng([2024, 991, i])
        scene = sample_scene(cfg.radar, cfg.scene, rng)
        reflections = _scene_reflections(scene, cfg, rng, scene_backend, samples_per_square_meter, max_samples, vehicle_assets=library)
        result = simulate(reflections, cfg.radar, psf, rng)
        synthetic_statistics.append(float(np.quantile(result.image, quantile)))
    reference = float(np.median(synthetic_statistics))
    target = float(np.median(real_statistics))
    from ..radar.noise import sample_noise
    noise_draws = sample_noise(config.radar, (100000,), np.random.default_rng(731))
    noise_power = np.abs(noise_draws) ** 2 if np.iscomplexobj(noise_draws) else np.maximum(noise_draws, 0)
    noise_floor = float(np.quantile(noise_power, quantile ** (1 / config.radar.doppler_bins)))
    if noise_floor > 0 and target <= noise_floor * 1.05:
        raise ValueError("Chosen gain statistic is dominated by the estimated Doppler-max background; inspect signal regions or calibrate gain explicitly")
    if not np.isfinite([reference, target]).all() or min(reference, target) <= 0:
        raise ValueError("Gain statistic has no positive simulated/measured response; inspect scenes or choose another declared quantile")
    gain = target / reference
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("Gain fit was nonfinite")
    config.radar.scattering_gain = gain
    # Absolute sensor paths are resolved by load_config and kept when exporting
    # a separate gain-fit config; generation materializes its sensor artifacts.
    save_config(config, output_path)
    json_write(Path(output_path).with_suffix(".gain_fit.json"), {
        "method": "unlabeled_training_RA_power_quantile_ratio", "quantile": quantile,
        "real_training_frame_count": len(real_statistics), "synthetic_frame_count": synthetic_frames,
        "real_statistics": real_statistics, "synthetic_unit_gain_statistics": synthetic_statistics,
        "relative_power_scattering_gain": gain, "absolute_RCS_calibration": False,
        "noise_Doppler_max_quantile_IID_estimate": noise_floor,
        "test_frames_used": 0, "note": "Global empirical relative gain; cannot establish material or scene fidelity."})
    return Path(output_path)
