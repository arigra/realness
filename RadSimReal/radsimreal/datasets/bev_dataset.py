"""Dataset bridge to true Cartesian targets, with native labels kept intact.

Native mode reads original arrays lazily and never copies/rewrites the source
dataset. Generated mode uses physical mesh/vehicle geometry, not RA enclosures.
"""
import json
from pathlib import Path
import numpy as np
import torch
from ..config import load_config
from ..bev.geometry import BEVGrid
from ..bev.projection import PolarToBEV


DEFAULT_BEV_NORMALIZATION = {"db_min": -30.0, "db_max": 30.0, "power_floor": 1e-12,
                             "radar_log_mean": 3.2438383, "radar_log_divisor": 6.8367246}


class BEVDataset:
    def __init__(self, root, split="train", normalization=None, include_radar=True, grid=None):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != 2:
            raise ValueError("BEV training requires a version-2 dataset with original Cartesian labels")
        self.samples = self.manifest["splits"][split]
        self.split = split
        self.include_radar = bool(include_radar)
        self.grid = BEVGrid.from_dict(self.manifest["bev_grid"]) if grid is None else grid
        self.config = load_config(self.root / "config.json")
        self.normalization = dict(normalization if normalization is not None else self.manifest.get("normalization", DEFAULT_BEV_NORMALIZATION))
        self.label_domain = self.manifest["label_domain"]
        self.metric_domain = self.manifest.get("metric_domain", "cartesian_bev_metres_axis_aligned")
        self.native_reader = None
        if self.manifest.get("source") == "native_reference":
            from ..native.readers import NativeReader
            options = dict(profile=self.manifest["native_profile"], trusted_labels=self.manifest.get("trusted_labels", False),
                           annotation_path=self.manifest.get("annotation_path"))
            if hasattr(NativeReader, "from_manifest") and "native_records" in self.manifest:
                self.native_reader = NativeReader.from_manifest(self.manifest["native_root"], self.manifest["dataset"],
                                                                 self.manifest["native_records"], **options)
            else:
                self.native_reader = NativeReader(self.manifest["native_root"], self.manifest["dataset"], **options)
            # A manifest is authoritative: a later-added file cannot enter a split.
            self._records = {record.id: record for record in self.native_reader.records}
        cfg = self.config.radar
        self.project = PolarToBEV(cfg.ranges, cfg.angular_grid, self.grid, cfg.azimuth_coordinate)
        self.radar_shape = (cfg.doppler_bins, cfg.range_bins, cfg.azimuth_bins)
        self._coordinates = self._coordinate_channels()
        self._check_normalization()

    def _check_normalization(self):
        norm = self.normalization
        for name in DEFAULT_BEV_NORMALIZATION:
            if name not in norm or not np.isfinite(norm[name]):
                raise ValueError(f"Missing/nonfinite normalization parameter {name}")
        if norm["db_max"] <= norm["db_min"] or norm["power_floor"] <= 0 or norm["radar_log_divisor"] <= 0:
            raise ValueError("Invalid fixed BEV/native-radar normalization")

    def _coordinate_channels(self):
        x, y = np.meshgrid(self.grid.x_centers, self.grid.y_centers)
        scale = max(abs(self.grid.x_min), abs(self.grid.x_max), abs(self.grid.y_min), abs(self.grid.y_max), 1e-6)
        return np.stack([x / scale, y / scale]).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def ra_cache_name(record_id):
        return record_id.replace("/", "__") + ".npy"

    def _cached_image(self, record_id):
        """Doppler-maximum linear power cached by scripts/cache_ra_images.py (prepared native data only).

        Used only when the cache index was built for this manifest's exact native profile.
        """
        if not hasattr(self, "_ra_cache_dir"):
            self._ra_cache_dir = None
            index = self.root / "ra_cache" / "INDEX.json"
            if index.is_file():
                info = json.loads(index.read_text(encoding="utf-8"))
                if info.get("native_profile") == self.manifest.get("native_profile") and info.get("quantity") == "max_doppler_linear_power":
                    self._ra_cache_dir = index.parent
        if self._ra_cache_dir is None:
            return None
        path = self._ra_cache_dir / self.ra_cache_name(record_id)
        return np.load(path, allow_pickle=False).astype(np.float64) if path.is_file() else None

    def raw(self, index):
        row = self.samples[index]
        if self.native_reader is not None:
            record = self._records.get(row["id"])
            if record is None:
                raise ValueError(f"Manifest frame disappeared from source dataset: {row['id']}")
            cached = self._cached_image(row["id"]) if not self.include_radar else None
            cube = None if cached is not None else self.native_reader.load_tensor(record)
            labels = json.loads((self.root / row["prepared_annotation"]).read_text(encoding="utf-8")) if row.get("prepared_annotation") else self.native_reader.load_annotations(record)
            if labels.get("requires_cartesian_affine"):
                raise ValueError("RADDet BEV training needs an explicit native Cartesian-label affine in the profile")
            boxes = np.asarray(labels["boxes_bev"], dtype=np.float64).reshape(-1, 4)
            oriented = np.asarray(labels.get("oriented_boxes_bev", []), dtype=np.float64).reshape(-1, 5)
            label_domain = labels["label_domain"]
            rad_boxes = self._canonical_rad_boxes(labels.get("native_rad_boxes_center_size", []), self.native_reader.profile)
            image = cached
        else:
            with np.load(self.root / row["image"], allow_pickle=False) as data:
                # The stored image is the Doppler maximum of the stored tensor (generation contract);
                # skip decompressing the full cube when the model does not consume it.
                cube = np.array(data["tensor"]) if "tensor" in data and self.include_radar else None
                image = np.array(data["image"])
            annotation = json.loads((self.root / row["annotation"]).read_text(encoding="utf-8"))
            objects = [o for o in annotation["objects"] if o["label"] == "car"]
            if any("box_bev" not in obj for obj in objects):
                raise ValueError("Generated annotations lack original Cartesian geometry; RA box conversion is forbidden")
            boxes = np.asarray([obj["box_bev"] for obj in objects], dtype=np.float64).reshape(-1, 4)
            oriented = np.asarray([obj["oriented_box_bev"] for obj in objects if "oriented_box_bev" in obj], dtype=np.float64).reshape(-1, 5)
            label_domain = self.label_domain
            rad_boxes = np.asarray([o["box_rad"] for o in objects if "box_rad" in o], dtype=np.float32).reshape(-1, 6)
        if cube is not None:
            power = np.abs(cube) ** 2 if np.iscomplexobj(cube) else np.maximum(cube, 0)
            if not np.isfinite(power).all():
                raise ValueError("Power conversion yielded nonfinite values")
            image = np.max(power, axis=1)
        else:
            power = None
        if image.shape != (self.config.radar.range_bins, self.config.radar.azimuth_bins) or not np.isfinite(image).all() or np.any(image < 0):
            raise ValueError("BEV source image must be finite nonnegative linear power on the declared radar grid")
        return {"id": row["id"], "power": power, "image_ra": image,
                "boxes_physical": boxes, "oriented_boxes": oriented, "label_domain": label_domain,
                "boxes_rad": rad_boxes}

    @staticmethod
    def _canonical_rad_boxes(raw_boxes, profile):
        boxes = np.asarray(raw_boxes, dtype=np.float64).reshape(-1, 6).copy()
        if not len(boxes):
            return boxes.astype(np.float32)
        r, a, d = profile.shape_native
        # RADDet gt 'boxes' store absolute native Doppler indices in [0, D);
        # verified on all 28,401 boxes of the public release (min 0.0, median 31.5, max 63.5).
        if profile.range_reversed:
            boxes[:, 0] = r - 1 - boxes[:, 0]
        if profile.azimuth_reversed:
            boxes[:, 1] = a - 1 - boxes[:, 1]
        if profile.canonical_doppler_flip:
            boxes[:, 2] = d - 1 - boxes[:, 2]
        # Doppler is periodic and RADDet boxes use inclusive bin centres: a centre of
        # D-0.5 lies between bin D-1 and wrapped bin 0 (released labels span -32..99).
        boxes[:, 2] = np.mod(boxes[:, 2], d)
        boxes[:, :3] /= [r, a, d]
        boxes[:, 3:] /= [r, a, d]
        if not np.isfinite(boxes).all() or np.any(boxes[:, :3] < 0) or np.any(boxes[:, :3] > 1) or np.any(boxes[:, 3:] <= 0):
            raise ValueError("Native RAD boxes are invalid after the declared axis conversion")
        return boxes.astype(np.float32)

    def __getitem__(self, index):
        raw = self.raw(index)
        norm = self.normalization
        # CPU float32 interpolation; reject overflow before converting.
        if np.any(raw["image_ra"] > np.finfo(np.float32).max):
            raise ValueError("Radar power exceeds float32 interpolation range; supply an explicit unit rescaling")
        with torch.no_grad():
            bev = self.project(torch.as_tensor(raw["image_ra"], dtype=torch.float32)).numpy()
        db = 10 * np.log10(np.maximum(bev, norm["power_floor"]))
        intensity = np.clip((db - norm["db_min"]) / (norm["db_max"] - norm["db_min"]), 0, 1).astype(np.float32)
        # Outside valid radar support is zero intensity, not a fabricated noise floor.
        intensity[~self.project.valid_mask.numpy()] = 0
        image = np.concatenate([intensity[None], self._coordinates], axis=0)
        physical = np.asarray(raw["boxes_physical"], dtype=np.float32).reshape(-1, 4)
        if not np.isfinite(physical).all() or (len(physical) and np.any(physical[:, 2:] <= physical[:, :2])):
            raise ValueError("Invalid original Cartesian target boxes")
        boxes = self.grid.normalize_boxes(physical).astype(np.float32)
        if len(boxes) and (np.any(boxes < -1e-6) or np.any(boxes > 1 + 1e-6)):
            raise ValueError("Original Cartesian labels extend outside the BEV grid. Expand the declared grid; training/evaluation may not silently crop ground truth.")
        boxes = np.clip(boxes, 0, 1)
        original_oriented = torch.as_tensor(raw["oriented_boxes"], dtype=torch.float32)
        train_oriented = original_oriented
        result = {"image": torch.from_numpy(image), "boxes": torch.from_numpy(boxes),
                  "boxes_physical": torch.from_numpy(physical), "id": raw["id"],
                  "oriented_boxes": train_oriented, "oriented_boxes_physical": original_oriented,
                  "boxes_rad": torch.as_tensor(raw["boxes_rad"], dtype=torch.float32),
                  "label_domain": raw["label_domain"], "target_boxes_outside_grid": 0}
        if raw["power"] is not None and self.include_radar:
            encoded = (np.log10(raw["power"] + 1) - norm["radar_log_mean"]) / norm["radar_log_divisor"]
            if not np.isfinite(encoded).all() or np.any(np.abs(encoded) > np.finfo(np.float32).max):
                raise ValueError("Nonfinite native RADDet input encoding")
            result["radar"] = torch.from_numpy(np.transpose(encoded, (1, 0, 2)).astype(np.float32))
        return result

    def get_meta(self):
        return {"format_version": 2, "source": self.manifest.get("source"),
                "bev_grid": self.grid.to_dict(), "normalization": self.normalization,
                "radar_shape": list(self.radar_shape), "label_domain": self.label_domain,
                "metric_domain": self.metric_domain, "split": self.split,
                "native_profile": self.manifest.get("native_profile"),
                "radar_config": self.config.to_dict()["radar"],
                "split_source": self.manifest.get("split_source"),
                "published_result_reproduced": False}


def fit_training_normalization(dataset, max_frames=32):
    """Fit disclosed display/input bounds from TRAIN only, never test/validation."""
    if dataset.split != "train" or len(dataset) == 0:
        raise ValueError("Normalization fitting requires nonempty training data")
    rng = np.random.default_rng(2024)
    selected = np.linspace(0, len(dataset) - 1, min(max_frames, len(dataset))).astype(int)
    db_values, radar_values = [], []
    for i in selected:
        raw = dataset.raw(int(i))
        image = raw["image_ra"].reshape(-1)
        image = image[rng.choice(len(image), min(4096, len(image)), replace=False)]
        db_values.append(10 * np.log10(np.maximum(image, 1e-12)))
        if raw["power"] is not None:
            values = raw["power"].reshape(-1)
            values = values[rng.choice(len(values), min(4096, len(values)), replace=False)]
            radar_values.append(np.log10(values + 1))
    values = np.concatenate(db_values)
    lo, hi = np.quantile(values, [.005, .999])
    if hi <= lo:
        hi = lo + 1
    result = dict(DEFAULT_BEV_NORMALIZATION, db_min=float(lo), db_max=float(hi))
    if radar_values:
        values = np.concatenate(radar_values)
        result.update(radar_log_mean=float(values.mean()), radar_log_divisor=max(float(values.std()), 1e-6))
    result["fit_source"] = "training_frames_only"
    result["frames_used"] = len(selected)
    return result
