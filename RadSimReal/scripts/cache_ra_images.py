"""Cache Doppler-maximum linear power (canonical range x azimuth, float32) for a prepared native dataset.

Written inside the prepared folder (ra_cache/), never next to the original recordings.
BEVDataset uses it only when include_radar is False and INDEX.json matches the manifest profile.
Parallel over processes; resumable (existing files are skipped).
"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

_reader = None


def _init(data):
    global _reader
    from radsimreal.native.workflows import reader_from_prepared
    _reader, _ = reader_from_prepared(data)


def _one(args):
    record_id, out = args
    path = Path(out)
    if path.is_file():
        return record_id, "skip"
    rec = next(r for r in _reader.records if r.id == record_id) if not hasattr(_reader, "get_record") else _reader.get_record(record_id)
    cube = _reader.load_tensor(rec)
    power = np.abs(cube) ** 2 if np.iscomplexobj(cube) else cube
    image = np.max(power, axis=1).astype(np.float32)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, image)
    os.replace(tmp, path)
    return record_id, "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from radsimreal.datasets.bev_dataset import BEVDataset
    manifest = json.loads((Path(a.data) / "manifest.json").read_text())
    cache = Path(a.data) / "ra_cache"
    cache.mkdir(exist_ok=True)
    ids = [row["id"] for s in a.splits for row in manifest["splits"][s]]
    jobs = [(i, str(cache / BEVDataset.ra_cache_name(i))) for i in ids]
    done = 0
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.data,)) as pool:
        for i, (rid, status) in enumerate(pool.map(_one, jobs, chunksize=8)):
            done += 1
            if done % 500 == 0:
                print(done, "/", len(jobs), flush=True)
    (cache / "INDEX.json").write_text(json.dumps({
        "quantity": "max_doppler_linear_power", "axes": ["range", "azimuth"], "dtype": "float32",
        "native_profile": manifest["native_profile"], "splits": a.splits, "count": len(jobs)}, indent=1))
    print("cached", len(jobs))


if __name__ == "__main__":
    main()
