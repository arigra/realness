"""Seeded generation with disjoint, independently sampled scene splits."""
import json
from pathlib import Path
import numpy as np
from ..config import save_config


def scene_annotations(scene, radar, reflections=None):
    """Project full object footprints to normalized RA pixel-edge XYXY boxes.

    Edge sampling handles the closest point on a long face occurring between
    corners. Physical footprint labels are a documented alternative to the
    unavailable author-generated RF extent/CFAR labels.
    """
    from ..radar import cartesian_to_polar
    records = []
    for obj in scene.objects:
        corners = np.asarray(obj.corners())[:, :2]
        pts = np.concatenate([a[None] + np.linspace(0, 1, 65)[:, None] * (b - a)[None]
                              for a, b in zip(corners, np.roll(corners, -1, axis=0))])
        ranges, angles = cartesian_to_polar(pts[:, 0], pts[:, 1])
        angular = np.sin(angles) if radar.azimuth_coordinate == "sin" else angles
        row = (ranges - radar.range_min) / radar.range_step
        col = (angular - radar.angular_grid[0]) / radar.angular_step
        box = np.array([(col.min() + 0.5) / radar.azimuth_bins,
                        (row.min() + 0.5) / radar.range_bins,
                        (col.max() + 0.5) / radar.azimuth_bins,
                        (row.max() + 0.5) / radar.range_bins])
        box = np.clip(box, 0, 1)
        if np.all(box[2:] > box[:2]):
            record = {"object_id": int(obj.object_id), "label": obj.label,
                      "box": box.tolist(), "box_space": "normalized_ra_xyxy",
                      "label_scope": "amodal_full_ground_footprint"}
            if reflections is not None:
                from ..radar import points_to_indices
                _, in_domain = points_to_indices(reflections.xyz, reflections.velocity, radar)
                n = int(np.sum((reflections.object_ids == obj.object_id) & in_domain))
                record.update(in_domain_reflection_count=n, has_simulated_return=bool(n))
            records.append(record)
    return records


def calibrated_psf(radar):
    if radar.psf_path is None:
        return None
    from ..radar import load_psf
    return load_psf(radar.psf_path, config=radar, strict_metadata=True)


def generate_dataset(output_dir, config, counts=None, save_tensors=False):
    from ..simulation import sample_scene, generate_reflections
    from ..radar import simulate
    counts = {"train": 64, "val": 16, "test": 16} if counts is None else dict(counts)
    if set(counts) != {"train", "val", "test"} or any(not isinstance(n, int) or n < 0 for n in counts.values()):
        raise ValueError("counts requires nonnegative integer train, val and test sizes")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Dataset destination must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    save_config(config, out / "config.json")
    psf = calibrated_psf(config.radar)
    # Materialize calibration with data: paths remain portable when copied.
    if psf is not None:
        from ..radar import save_psf, load_psf
        _, calibration_metadata = load_psf(config.radar.psf_path, return_metadata=True)
        calibration_metadata["source_path_at_generation"] = str(config.radar.psf_path)
        save_psf(out / "calibration.npz", psf, calibration_metadata)
        saved = config.to_dict()
        saved["radar"]["psf_path"] = "calibration.npz"
        (out / "config.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
    manifest = {"format_version": 1, "source": "independent_approximate_box_surface_scenes",
                "axis_order": ["range", "doppler", "azimuth"],
                "box_space": "normalized_ra_xyxy", "splits": {}}
    for split_number, (split, count) in enumerate(counts.items()):
        (out / split).mkdir()
        rows = []
        for i in range(count):
            seed = [config.seed, split_number, i]
            rng = np.random.default_rng(np.random.SeedSequence(seed))
            scene = sample_scene(config.radar, config.scene, rng)
            reflections = generate_reflections(scene, config.radar, config.scene, rng)
            result = simulate(reflections, config.radar, psf=psf, rng=rng)
            name = f"{split}_{i:06d}"
            array_path = f"{split}/{name}.npz"
            annotation_path = f"{split}/{name}.json"
            arrays = {"image": result.image.astype(np.float32)}
            if save_tensors:
                arrays["tensor"] = result.tensor
            np.savez_compressed(out / array_path, **arrays)
            annotation = {"id": name, "seed_sequence": seed, "scene": scene.to_dict(),
                          "objects": scene_annotations(scene, config.radar, reflections),
                          "simulation": result.metadata}
            (out / annotation_path).write_text(json.dumps(annotation, indent=2), encoding="utf-8")
            rows.append({"id": name, "image": array_path, "annotation": annotation_path})
        manifest["splits"][split] = rows
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out

