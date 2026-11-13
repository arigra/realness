"""Fixed, explicit intensity preprocessing and variable-length car annotations."""
import json
from pathlib import Path
import numpy as np
from ..config import load_config

DEFAULT_NORMALIZATION = {"db_min": -30.0, "db_max": 30.0, "power_floor": 1e-12}


def preprocess_image(image, radar, normalization=None):
    """Return float32 [intensity dB, lateral x, forward y], each scaled explicitly.

    The first channel is clipped to [0,1] using fixed calibration-independent
    bounds. The coordinates are divided by the last range-bin center. This
    recipe is a reconstruction choice; the authors do not publish theirs.
    """
    norm = dict(DEFAULT_NORMALIZATION if normalization is None else normalization)
    lo, hi, floor = norm["db_min"], norm["db_max"], norm["power_floor"]
    if not np.isfinite([lo, hi, floor]).all() or not hi > lo or floor <= 0:
        raise ValueError("normalization requires finite db_max>db_min and positive power_floor")
    image = np.asarray(image)
    if image.shape != (radar.range_bins, radar.azimuth_bins):
        raise ValueError(f"image shape {image.shape} does not match configured radar")
    if np.iscomplexobj(image) or not np.isfinite(image).all() or np.any(image < 0):
        raise ValueError("Expected finite, nonnegative LINEAR intensity/power image")
    db = 10 * np.log10(np.maximum(image, floor))
    normalized = np.clip((db - lo) / (hi - lo), 0, 1)
    rr, aa = np.meshgrid(radar.ranges, radar.azimuths_rad, indexing="ij")
    scale = max(float(radar.ranges[-1]), 1e-6)
    return np.stack([normalized, rr * np.sin(aa) / scale, rr * np.cos(aa) / scale]).astype(np.float32)


class RadarDataset:
    def __init__(self, root, split="train", normalization=None):
        self.root = Path(root)
        self.config = load_config(self.root / "config.json")
        self.normalization = dict(DEFAULT_NORMALIZATION if normalization is None else normalization)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if split not in self.manifest["splits"]:
            raise ValueError(f"Unknown split {split!r}")
        self.samples = self.manifest["splits"][split]
        self.split = split
        self.shape = (self.config.radar.range_bins, self.config.radar.azimuth_bins)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch
        record = self.samples[index]
        with np.load(self.root / record["image"], allow_pickle=False) as data:
            image = data["image"]
        annotation = json.loads((self.root / record["annotation"]).read_text(encoding="utf-8"))
        boxes = np.asarray([obj["box"] for obj in annotation["objects"] if obj["label"] == "car"], dtype=np.float32).reshape(-1, 4)
        if len(boxes) and (not np.isfinite(boxes).all() or np.any(boxes < 0) or np.any(boxes > 1) or np.any(boxes[:, 2:] <= boxes[:, :2])):
            raise ValueError(f"Invalid normalized XYXY boxes in {record['annotation']}")
        return {"image": torch.from_numpy(preprocess_image(image, self.config.radar, self.normalization)),
                "boxes": torch.from_numpy(boxes), "id": record["id"]}


def collate_fn(batch):
    import torch
    return {"image": torch.stack([b["image"] for b in batch]),
            "boxes": [b["boxes"] for b in batch], "id": [b["id"] for b in batch]}

