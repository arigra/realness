"""Candidate bank v2 (docs/notes/2026-11-06-metric-v2-preregistration.md, "Bank v2").

Every candidate is the dataset's best tuned simulator on its scene seeds (SEED0 + i; damage generator i, as
Study 2), so candidates are paired sequence by sequence; what a candidate randomises is drawn per sequence from
torch.Generator(9_000_000 + i) in the fixed order of its spec and saved in the sequence's meta.
- tuned_control: the best tuned simulator's cache, unchanged.
- scnr_narrow / scnr_wide: target and static-scene strength offsets (dB, on top of the tuned offsets).
- shape_range: range blur (x the range response width at half power; Doppler 1) and a one-sided range tail
  (cells); shape_doppler: Doppler blur (range 1) and micro-motion scale.
- label_stochastic: the tuned_control maps; each (object, frame) label moved to its strongest cell within
  +/-4 cells plus an offset drawn from the real training labels' (label - strongest cell) offsets, measured on
  the real training sequences outside the primary reference.
- support_stratified: the scene of a sequence is drawn with weights so that the four reference strata (near /
  far at the primary reference's median range x |Doppler offset| <= 1 bin / moving) are equally likely: a
  scene's weight = mean over its objects of 1 / (stratum share among all training scenes' objects); scenes
  with no object keep their natural share of draws (clarification, fixed before rendering). The main
  generator still makes its own scene draw, so everything else in the sequence is the tuned one's.
- combined_moderate: scnr_narrow + range blur U(1, 1.5) and tail U(0, 2) + support_stratified scenes.
Caches: data/tuned/<dataset>/bank2_<name> (written once).
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.damage import TrackDamage, blur_sigma
from src.data import load, write_cache
from src.tuned import simulator, table
from src.versions import DATASETS, N_SIM, SEED0, best

DRAW_SEED = 9_000_000
K_OFFSET = 4
BANK2 = {
    "tuned_control": {},
    "scnr_narrow": {"target_db": (-3.0, 3.0), "static_minus_target_db": (-2.0, 2.0)},
    "scnr_wide": {"target_db": (-9.0, 6.0), "static_db": (-9.0, 6.0)},
    "shape_range": {"blur_r": (1.0, 2.5), "tail": (0.0, 4.0)},
    "shape_doppler": {"blur_d": (1.0, 2.5), "micro": (0.0, 1.0)},
    "label_stochastic": {"label_offsets": "real training labels outside the primary reference"},
    "support_stratified": {"scenes": "stratified"},
    "combined_moderate": {"target_db": (-3.0, 3.0), "static_minus_target_db": (-2.0, 2.0),
                          "blur_r": (1.0, 1.5), "tail": (0.0, 2.0), "scenes": "stratified"},
}


def cache_dir(dataset, name):
    if name == "tuned_control":
        return Path("data/tuned") / dataset / "best"
    return Path("data/tuned") / dataset / f"bank2_{name}"


def draws(spec, i):
    """Per-sequence draws, in the spec's order, from Generator(DRAW_SEED + i)."""
    g = torch.Generator().manual_seed(DRAW_SEED + i)
    out = {}
    for k, v in spec.items():
        if isinstance(v, tuple):
            out[k] = float(v[0] + (v[1] - v[0]) * torch.rand((), generator=g))
    if "static_minus_target_db" in out:
        out["static_db"] = out["target_db"] + out["static_minus_target_db"]
    return out, g


class BankDamage(TrackDamage):
    """TrackDamage with separate range / Doppler blur factors and weighted real-scene draws."""

    def __init__(self, *args, blur_r=1.0, blur_d=1.0, scene_weights=None, scene_generator=None, **kw):
        super().__init__(*args, **kw)
        self.sigma = (blur_sigma(self.win_r, blur_r), blur_sigma(self.win_d, blur_d))
        self.scene_weights, self.scene_g = scene_weights, scene_generator
        self.scene_index = None

    def _objects(self):
        if self.scene_weights is None or self.scenes is None or self._own_scenario:
            return super()._objects()
        torch.randint(len(self.scenes), (), generator=self.g)          # the tuned draw, kept for pairing
        k = int(torch.multinomial(self.scene_weights, 1, generator=self.scene_g))
        self.scene_index = k
        return [self._from_label(lab, j) for j, lab in enumerate(self.scenes[k][:8])]


def stratum(range_bin, doppler_bin, median_range_bin, zero, n_d):
    near_zero = abs(((doppler_bin - zero + n_d / 2) % n_d) - n_d / 2) <= 1
    return int(range_bin > median_range_bin) * 2 + int(near_zero)


def scene_weights(dataset, scenes, reference_seqs):
    """Weights of the tuned scene table for support_stratified, and the strata definition."""
    ds = DATASETS[dataset]
    n_d, zero = ds.grid[1], 32
    real = load(ds.real, "train")
    ranges = [float(r) for s in reference_seqs for t in range(real[s]["x"].shape[0])
              for r, _ in real[s]["traj"][real[s]["present"][:, t], t].tolist()]
    med = float(np.median(ranges))
    st = [[stratum(o["range_bin"], o["doppler_bin"], med, zero, n_d) for o in sc] for sc in scenes]
    allst = np.asarray([b for s in st for b in s])
    share = {b: float((allst == b).mean()) for b in range(4)}
    w = np.asarray([np.mean([1.0 / share[b] for b in s]) if s else 0.0 for s in st])
    empty = np.asarray([len(s) == 0 for s in st])
    if empty.any():                         # empty scenes keep their natural share of draws
        p_empty = empty.mean()
        w[~empty] = w[~empty] / w[~empty].sum() * (1 - p_empty)
        w[empty] = p_empty / empty.sum()
    w = w / w.sum()
    expected = {b: float(sum(w[k] * np.mean([x == b for x in s]) for k, s in enumerate(st) if s)
                         / max(sum(w[k] for k, s in enumerate(st) if s), 1e-12)) for b in range(4)}
    return torch.tensor(w, dtype=torch.double), {"median_range_bin": med, "scene_strata_share": share,
                                                  "expected_object_strata_share": expected,
                                                  "empty_scene_share": float(empty.mean())}


def real_offsets(dataset, exclude):
    """(label - strongest cell within +/-4) of every real training label outside `exclude`, (n, 2)."""
    out = []
    for s, it in enumerate(load(DATASETS[dataset].real, "train")):
        if s in exclude:
            continue
        for t in range(it["x"].shape[0]):
            x = it["x"][t].float().numpy()
            for rf, df in it["traj"][it["present"][:, t], t].tolist():
                pr, pd = strongest(x, rf, df)
                n_d = x.shape[1]
                out.append((rf - pr, ((df - pd + n_d / 2) % n_d) - n_d / 2))
    return np.asarray(out, dtype=np.float64)


def strongest(x, rf, df, k=K_OFFSET):
    n_r, n_d = x.shape
    ri, di = min(max(int(round(rf)), 0), n_r - 1), int(round(df)) % n_d
    rows = np.arange(max(ri - k, 0), min(ri + k + 1, n_r))
    cols = (di + np.arange(-k, k + 1)) % n_d
    w = x[np.ix_(rows, cols)]
    i = np.unravel_index(int(np.argmax(w)), w.shape)
    return int(rows[i[0]]), int(cols[i[1]])


def _sim_sequence(dataset, tables, p, i, weights=None):
    kw = {"blur_r": p.get("blur_r", 1.0), "blur_d": p.get("blur_d", 1.0), "tail": p.get("tail", 0.0),
          "micro": p.get("micro", 1.0)}
    sg = None
    if weights is not None:
        sg = torch.Generator().manual_seed(DRAW_SEED + 500_000 + i)
        kw |= {"scene_weights": weights, "scene_generator": sg}
    sim = simulator(dataset, best(dataset), torch.Generator().manual_seed(SEED0 + i), None, tables,
                    mixin=BankDamage, damage_generator=torch.Generator().manual_seed(i),
                    target_db=p.get("target_db", 0.0), static_db=p.get("static_db", 0.0), **kw)
    o = sim.gen_sequence()
    meta = {"draws": p}
    if weights is not None:
        meta["scene_index"] = sim.scene_index
    return o["x"].numpy(), o["labels"], meta


def _label_sequence(control, offsets, i):
    """The control's maps; labels at the strongest cell within +/-4 plus a real offset."""
    it = control[i]
    g = np.random.default_rng(DRAW_SEED + i)
    x = it["x"].float().numpy()
    n_r, n_d = x.shape[1:]
    frames, used = [], []
    for t in range(x.shape[0]):
        labs = []
        for m in torch.nonzero(it["present"][:, t]).flatten().tolist():
            rf, df = it["traj"][m, t].tolist()
            pr, pd = strongest(x[t], rf, df)
            k = int(g.integers(len(offsets)))
            used.append(k)
            labs.append({"id": m, "range_bin": float(np.clip(pr + offsets[k, 0], 0, n_r - 1)),
                         "doppler_bin": float((pd + offsets[k, 1]) % n_d)})
        frames.append(labs)
    return it["x"].numpy(), frames, {"offset_indices": used}


def render(dataset, name, reference_seqs):
    """Write a bank-v2 candidate's cache once; returns (cache, info)."""
    out = cache_dir(dataset, name)
    if name == "tuned_control":
        return out, {"note": "the best tuned simulator's cache, unchanged"}
    spec = BANK2[name]
    done = {"name": name, "spec": {k: list(v) if isinstance(v, tuple) else v for k, v in spec.items()},
            "switches": list(best(dataset)), "reference_seqs": list(map(int, reference_seqs)), "version": 1}
    info_path = out / "bank2_info.json"
    if (out / "DONE").exists() and json.loads((out / "DONE").read_text()) == done:
        return out, json.loads(info_path.read_text())
    ds, tables = DATASETS[dataset], table(dataset)
    info = {}
    if name == "label_stochastic":
        offsets = real_offsets(dataset, set(map(int, reference_seqs)))
        control = load(cache_dir(dataset, "tuned_control"), "train", N_SIM)
        info = {"n_real_offsets": len(offsets), "offset_median": np.median(offsets, 0).tolist(),
                "offset_iqr": (np.percentile(offsets, 75, 0) - np.percentile(offsets, 25, 0)).tolist()}
        make = lambda i: _label_sequence(control, offsets, i)
    else:
        weights = None
        if spec.get("scenes") == "stratified":
            weights, info = scene_weights(dataset, tables["scenes"], reference_seqs)
        make = lambda i: _sim_sequence(dataset, tables, draws(spec, i)[0], i, weights)
    write_cache(out, N_SIM, make, ds.grid, ds.stats_every, slot_by_rank=dataset != "radial")
    info["generated"] = generated_strata(dataset, out, reference_seqs)
    info_path.write_text(json.dumps(info, indent=1))
    (out / "DONE").write_text(json.dumps(done))
    return out, info


def generated_strata(dataset, cache, reference_seqs):
    """Share of the generated labels in each reference stratum (saved for every candidate)."""
    ds = DATASETS[dataset]
    real = load(ds.real, "train")
    ranges = [float(r) for s in reference_seqs for t in range(real[s]["x"].shape[0])
              for r, _ in real[s]["traj"][real[s]["present"][:, t], t].tolist()]
    med = float(np.median(ranges))
    labels = torch.load(Path(cache) / "train_labels.pt")
    tr, pr = labels["traj"][:N_SIM], labels["present"][:N_SIM]
    st = [stratum(r, d, med, 32, ds.grid[1]) for r, d in tr[pr].tolist()]
    st = np.asarray(st)
    return {"median_range_bin": med, "n_labels": int(len(st)),
            "share": {b: float((st == b).mean()) if len(st) else math.nan for b in range(4)}}
