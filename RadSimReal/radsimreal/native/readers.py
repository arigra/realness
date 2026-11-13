"""Native array discovery and canonical loading without guessed value domains."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from .profiles import NativeProfile, reference_profile


@dataclass
class NativeRecord:
    id: str
    path: Path
    sequence: str
    frame_id: int | None = None
    annotation_path: Path | None = None
    split: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self, root=None):
        def path_text(path):
            if path is None:
                return None
            if root is not None:
                try:
                    return Path(path).relative_to(root).as_posix()
                except ValueError:
                    pass
            return str(path)
        return {"id": self.id, "path": path_text(self.path), "sequence": self.sequence,
                "frame_id": self.frame_id, "annotation_path": path_text(self.annotation_path),
                "split": self.split, "metadata": self.metadata}


def discover_records(root, dataset):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {root}")
    dataset = dataset.lower()
    if dataset not in ("raddet", "carrada"):
        raise ValueError("dataset must be raddet or carrada")
    folder = "RAD" if dataset == "raddet" else "RAD_numpy"
    candidates = []
    if root.name.lower() == folder.lower():
        candidates.extend(root.rglob("*.npy"))
    else:
        for path in root.rglob("*.npy"):
            if any(p.lower() == folder.lower() for p in path.relative_to(root).parts[:-1]):
                candidates.append(path)
    records = []
    seen = set()
    for path in sorted(candidates):
        if path.is_symlink():
            # A linked root is resolved above; linked files must be selected by
            # passing their real root so discovery has explicit provenance.
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        parts = path.parts
        indices = [i for i, p in enumerate(parts[:-1]) if p.lower() == folder.lower()]
        if not indices:
            continue
        marker = indices[-1]
        frame = int(path.stem) if path.stem.isdigit() else None
        split = next((p.lower() for p in parts[:marker] if p.lower() in ("train", "val", "test")), None)
        if dataset == "raddet":
            suffix = Path(*parts[marker + 1:]).with_suffix(".pickle")
            annotation = Path(*parts[:marker]) / "gt" / suffix
            if not annotation.exists() and annotation.with_suffix(".pkl").exists():
                annotation = annotation.with_suffix(".pkl")
            if not annotation.exists() and len(suffix.parts) == 2:
                # Some extracted RADDet copies repeat the part folder only under gt/
                # (gt/part_2/part_2/000031.pickle). Same file name, one extra level.
                nested = annotation.parent / suffix.parts[0] / annotation.name
                if nested.exists():
                    annotation = nested
            from ..datasets.protocol import scene_id_for_frame
            scene = scene_id_for_frame(frame) if frame is not None else None
            sequence = f"scene_{scene}" if scene is not None else f"unassigned/{path.parent.name}"
            record_id = f"raddet/{path.stem}" if frame is not None else f"raddet/{path.relative_to(root).with_suffix('').as_posix()}"
        else:
            sequence = parts[marker - 1] if marker > 0 else root.parent.name
            annotation = next((candidate for candidate in
                (root / "annotations_frame_oriented.json", root.parent / "annotations_frame_oriented.json",
                 Path(*parts[:marker - 1]) / "annotations_frame_oriented.json") if candidate.exists()), None)
            record_id = f"carrada/{sequence}/{path.stem}"
        records.append(NativeRecord(record_id, path, sequence, frame, annotation, split,
                                    {"native_layout": "R,A,D", "native_subset": split,
                                     "relative_path": path.relative_to(root).as_posix()}))
    if not records:
        if dataset == "carrada":
            raise FileNotFoundError("No RAD_numpy/*.npy tensors found. CARRADA full RAD tensors are a separate download; processed 2D views are insufficient.")
        raise FileNotFoundError("No native RAD/**/*.npy files found under the supplied root")
    ids = [r.id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate original frame IDs discovered; choose one dataset copy/root to prevent leakage")
    return records


class NativeReader:
    """Canonical R,D,A reader. Real native tensors need explicit representation.

    Full complex arrays preserve fields. Power/amplitude/dB declarations produce
    linear-power arrays. No network normalization or Doppler maximum occurs here.
    Native annotation decoding is separate and may need trusted pickle consent.
    """
    def __init__(self, root, dataset, profile=None, trusted_labels=False, annotation_path=None):
        self.root = Path(root).expanduser().resolve()
        self.dataset = dataset.lower()
        self.records = discover_records(self.root, self.dataset)
        self._apply_annotation_override(annotation_path)
        self._configure(profile, trusted_labels)

    @classmethod
    def from_manifest(cls, root, dataset, records, profile, trusted_labels=False, annotation_path=None):
        """Open only indexed records, without rediscovering a large raw dataset.

        Raw paths must resolve inside root. Relative annotation paths must also
        stay there. A separately supplied CARRADA annotation_path is the explicit
        authorization to use an external annotation archive. Added raw files are
        never admitted to an existing manifest. The first array header is checked
        against the profile; each later array is checked when loaded.
        """
        reader = cls.__new__(cls)
        reader.root = Path(root).expanduser().resolve()
        reader.dataset = str(dataset).lower()
        if not reader.root.is_dir():
            raise FileNotFoundError(f"Native manifest root is not a directory: {reader.root}")
        if reader.dataset not in ("raddet", "carrada"):
            raise ValueError("dataset must be raddet or carrada")
        reader.records = []
        ids, paths = set(), set()

        def inside(value, kind):
            path = (reader.root / Path(value)).resolve()
            try:
                path.relative_to(reader.root)
            except ValueError as error:
                raise ValueError(f"Native manifest {kind} path escapes raw dataset root") from error
            return path

        for row in records:
            if not isinstance(row, dict) or not {"id", "path", "sequence"} <= set(row):
                raise ValueError("Native manifest record requires id, path and sequence")
            identifier, sequence = str(row["id"]), str(row["sequence"])
            if not identifier or not sequence or identifier in ids:
                raise ValueError("Native manifest record IDs must be nonempty and unique")
            path = inside(row["path"], "tensor")
            if not path.is_file() or path.suffix.lower() != ".npy":
                raise FileNotFoundError(f"Manifest tensor is missing or is not NPY: {path}")
            if path in paths:
                raise ValueError("Native manifest repeats a tensor path under different IDs")
            frame = row.get("frame_id")
            if frame is not None and (isinstance(frame, bool) or not isinstance(frame, int)):
                raise ValueError("Native manifest frame_id must be an integer or null")
            if path.stem.isdigit() and frame != int(path.stem):
                raise ValueError("Native manifest frame_id disagrees with original file stem")
            expected = f"raddet/{path.stem}" if reader.dataset == "raddet" and frame is not None else None
            if reader.dataset == "carrada":
                expected = f"carrada/{sequence}/{path.stem}"
            if expected is not None and identifier != expected:
                raise ValueError("Native manifest id disagrees with original frame/sequence identity")
            split = row.get("split")
            if split not in (None, "train", "val", "test"):
                raise ValueError("Native manifest split must be train, val, test or null")
            annotation = row.get("annotation_path")
            if annotation_path is None and annotation is not None:
                annotation = inside(annotation, "annotation")
            else:
                annotation = None
            metadata = dict(row.get("metadata", {}))
            metadata["relative_path"] = path.relative_to(reader.root).as_posix()
            reader.records.append(NativeRecord(identifier, path, sequence, frame, annotation, split, metadata))
            ids.add(identifier)
            paths.add(path)
        reader._apply_annotation_override(annotation_path)
        reader._configure(profile, trusted_labels)
        return reader

    def _apply_annotation_override(self, annotation_path):
        if annotation_path is not None:
            annotation_path = Path(annotation_path).expanduser().resolve()
            if self.dataset != "carrada" or not annotation_path.is_file():
                raise ValueError("annotation_path override must point to CARRADA annotations_frame_oriented.json")
            for record in self.records:
                record.annotation_path = annotation_path

    def _configure(self, profile, trusted_labels):
        probe = np.load(self.records[0].path, mmap_mode="r", allow_pickle=False) if self.records else None
        if probe is not None and probe.ndim != 3:
            raise ValueError("Native radar file must be a 3D tensor, not a processed view")
        if profile is None:
            if probe is None:
                raise ValueError("An empty native manifest requires an explicit profile")
            representation = "complex" if np.iscomplexobj(probe) else None
            profile = reference_profile(self.dataset, tuple(probe.shape), representation)
        elif isinstance(profile, (str, Path)):
            profile = NativeProfile.load(profile)
        elif isinstance(profile, dict):
            profile = NativeProfile.from_dict(profile)
        if not isinstance(profile, NativeProfile) or profile.name != self.dataset:
            raise ValueError("Profile dataset must match the requested native dataset")
        if probe is not None and probe.shape != profile.shape_native:
            raise ValueError(f"First native tensor shape {probe.shape} differs from profile {profile.shape_native}")
        if profile.representation is None:
            if probe is not None and np.iscomplexobj(probe):
                from dataclasses import replace
                profile = replace(profile, representation="complex")
            else:
                raise ValueError("Real native tensor representation is ambiguous: explicitly declare power, amplitude or db in a NativeProfile")
        if probe is not None and profile.representation == "complex" and not np.iscomplexobj(probe):
            raise ValueError("Complex native profile requires a complex tensor header")
        if probe is not None and np.iscomplexobj(probe) and profile.representation not in ("complex", "amplitude"):
            raise ValueError("Complex native tensors require complex or explicit amplitude representation")
        self.profile = profile
        self.trusted_labels = bool(trusted_labels)
        self._annotation_cache = None

    def __len__(self):
        return len(self.records)

    def get_record(self, record):
        if isinstance(record, NativeRecord):
            return record
        if isinstance(record, (int, np.integer)):
            return self.records[int(record)]
        text = str(record)
        for candidate in self.records:
            if candidate.id == text or candidate.metadata["relative_path"] == text or str(candidate.path) == text:
                return candidate
        raise KeyError(f"Unknown native record: {text}")

    def load_tensor(self, record):
        record = self.get_record(record)
        array = np.load(record.path, mmap_mode="r", allow_pickle=False)
        if array.shape != self.profile.shape_native:
            raise ValueError(f"Native shape {array.shape} differs from profile {self.profile.shape_native}: {record.id}")
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError("Native radar dtype must be numerical")
        rep = self.profile.representation
        if rep == "complex":
            if not np.iscomplexobj(array):
                raise ValueError("complex profile requires a complex field tensor")
            array = np.asarray(array, dtype=np.complex128)
        else:
            if np.iscomplexobj(array) and rep != "amplitude":
                raise ValueError("Complex native tensors require complex or explicit amplitude representation")
            array = np.asarray(array, dtype=np.complex128 if np.iscomplexobj(array) else np.float64)
            with np.errstate(over="raise", invalid="raise"):
                try:
                    if rep == "amplitude":
                        array = np.abs(array) ** 2
                    elif rep == "db":
                        array = 10.0 ** (array / 10.0)
                    elif rep != "power":
                        raise ValueError("Unsupported native response convention")
                except FloatingPointError as error:
                    raise ValueError("Native tensor overflows declared response conversion") from error
            if np.any(array < 0):
                raise ValueError("Power/amplitude tensor must yield nonnegative power")
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite native tensor values: {record.id}")
        if self.profile.range_reversed:
            array = array[::-1, :, :]
        if self.profile.azimuth_reversed:
            array = array[:, ::-1, :]
        if self.profile.canonical_doppler_flip:
            array = array[:, :, ::-1]
        return np.transpose(array, (0, 2, 1)).copy()

    def load_power(self, record):
        array = self.load_tensor(record)
        if np.iscomplexobj(array):
            with np.errstate(over="raise", invalid="raise"):
                try:
                    array = np.abs(array) ** 2
                except FloatingPointError as error:
                    raise ValueError("Complex native tensor overflows squared magnitude") from error
        return array

    def load_annotations(self, record):
        from .annotations import load_raddet_annotations, load_carrada_annotations
        record = self.get_record(record)
        if self.dataset == "raddet":
            return load_raddet_annotations(record, self.profile, trusted_labels=self.trusted_labels)
        if record.annotation_path is None:
            raise FileNotFoundError("CARRADA annotations_frame_oriented.json was not found; provide annotation_path when the RAD archive is separate")
        cache_key = (str(record.annotation_path), record.sequence)
        if self._annotation_cache is None or self._annotation_cache[0] != cache_key:
            from .annotations import load_json_sequence
            sequence = load_json_sequence(record.annotation_path, record.sequence)
            self._annotation_cache = (cache_key, sequence)
        return load_carrada_annotations(record, self.profile, {record.sequence: self._annotation_cache[1]})
