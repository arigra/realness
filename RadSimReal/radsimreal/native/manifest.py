"""Scene-disjoint native manifests, keeping original IDs and explicit gaps."""
from __future__ import annotations

from pathlib import Path
import json


def make_native_manifest(reader, val_sequences=None, test_sequences=None, explicit_split_file=None):
    """Create a portable manifest with deterministic scene-level partitions.

    RADDet defaults to the RadSimReal held-out scenes 9 and 11; validation
    scenes must be supplied explicitly. CARRADA requires explicit validation
    and test sequence lists, or a mapping file; no author manifest is invented.
    No available-file count is inferred from published integer intervals.

    Explicit JSON accepts either {'train':[sequences],...} or
    {'sequences':{'sequence':'train|val|test'}}. All discovered sequences must
    have one assignment. Per-frame assignment is intentionally unsupported.
    """
    actual = {record.sequence for record in reader.records}
    assignments = {}
    source = "explicit scene/sequence split"
    missing_reference_test_scenes = []
    if explicit_split_file is not None:
        path = Path(explicit_split_file)
        data = json.loads(path.read_text(encoding="utf-8"))
        if "sequences" in data:
            assignments = dict(data["sequences"])
        else:
            for split in ("train", "val", "test"):
                for sequence in data.get(split, []):
                    if sequence in assignments:
                        raise ValueError(f"Sequence {sequence!r} assigned more than once")
                    assignments[str(sequence)] = split
        missing = actual - set(assignments)
        if missing:
            raise ValueError(f"Explicit split does not assign observed sequences: {sorted(missing)}")
    else:
        val = set(map(str, val_sequences or []))
        if reader.dataset == "raddet":
            val = {f"scene_{v}" if v.isdigit() else v for v in val}
            if not val:
                raise ValueError("RADDet validation scenes must be explicitly supplied; no random frame split is permitted")
            test = {"scene_9", "scene_11"} if test_sequences is None else {str(v) for v in test_sequences}
            test = {f"scene_{v}" if v.isdigit() else v for v in test}
            if test_sequences is None:
                missing_reference_test_scenes = sorted(test - actual)
            source = "RadSimReal test scenes 9/11 with explicitly chosen validation scenes" if test_sequences is None else source
        else:
            if val_sequences is None or test_sequences is None:
                raise ValueError("CARRADA needs explicit val_sequences and test_sequences or an explicit sequence split JSON")
            test = set(map(str, test_sequences))
        if val & test:
            raise ValueError("Validation and test sequences overlap")
        unused = (val | test) - actual
        if reader.dataset == "raddet" and test_sequences is None:
            unused -= {"scene_9", "scene_11"}
        if unused:
            raise ValueError(f"Requested split sequences are absent from this dataset: {sorted(unused)}")
        assignments = {sequence: "val" if sequence in val else "test" if sequence in test else "train" for sequence in actual}
    if any(split not in ("train", "val", "test") for split in assignments.values()):
        raise ValueError("Split values must be train, val or test")
    splits = {name: [] for name in ("train", "val", "test")}
    unassigned = []
    for record in reader.records:
        if reader.dataset == "raddet" and record.sequence.startswith("unassigned/"):
            unassigned.append(record.to_dict(reader.root))
            continue
        row = record.to_dict(reader.root)
        row["split"] = assignments[record.sequence]
        splits[row["split"]].append(row)
    return {"format_version": 2, "dataset": reader.dataset, "profile": reader.profile.to_dict(),
            "tensor_axis_order": ["range", "doppler", "azimuth"],
            "split_source": source, "sequence_assignments": assignments,
            "splits": splits, "unassigned_records": unassigned,
            "missing_reference_test_scenes": missing_reference_test_scenes,
            "counts": {key: len(value) for key, value in splits.items()},
            "published_result_reproduced": False}
