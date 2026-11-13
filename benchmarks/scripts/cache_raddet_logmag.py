"""Local cache of RADDet RAD inputs as the exact float32 values the official loader computes.

Official: RAD complex64 -> util/helper.complexTo2Channels -> log10(|x|^2 + 1) (float32).
Reading the 33 MB complex files from /truenas takes ~2 s each, so training is I/O-bound
(~6 s/step). This writes log10(|x|^2 + 1) as float32 .npy to local disk, mirroring the raw
layout; the patched util/loader.readRAD uses the cache file when it exists. Originals untouched.
Writes are atomic (tmp + rename), existing files are skipped, so it can be rerun/resumed.

  python3 scripts/cache_raddet_logmag.py [--workers 6]
"""
import argparse, glob, os
from multiprocessing import Pool
import numpy as np

RAW = "/truenas/home/arigra/permuter/ariGranevich/radar data/RADDet/raw"
CACHE = "/tmp/arigra_raddet_logmag"


def convert(src):
    dst = os.path.join(CACHE, os.path.relpath(src, RAW))
    if os.path.exists(dst):
        return 0
    a = np.load(src)
    assert a.dtype == np.complex64
    out = np.log10(pow(np.abs(a), 2) + 1.)          # identical ops to helper.getMagnitude + getLog
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + f".tmp{os.getpid()}.npy"
    np.save(tmp, out)
    os.replace(tmp, dst)
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()
    files = sorted(glob.glob(f"{RAW}/train/RAD/*/*.npy")) + sorted(glob.glob(f"{RAW}/test/RAD/*/*.npy"))
    done = 0
    with Pool(a.workers) as p:
        for i, n in enumerate(p.imap_unordered(convert, files, chunksize=4)):
            done += n
            if i % 200 == 0:
                print(f"{i + 1}/{len(files)} (new {done})", flush=True)
    print("done", len(files), "new", done, flush=True)
