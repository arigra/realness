"""New natural candidates for the prospective test (docs/notes/2026-11-06-metric-experiments.md, Phase 6).

Two mechanisms, both on the dataset's best tuned simulator and its scene seeds (SEED0 + i), so a
bank candidate differs from the tuned simulator only in how targets look:
- domain randomisation (dr_*): every sequence draws its own blur, range tail, target strength
  offset, confuser density and micro-motion scale from the ranges in BANK;
- hybrid (hybrid_*): the tuned simulator with every target unrendered (labels kept), and a real
  target template pasted at each label. Templates come from real training sequences that are *not*
  in the metric's reference subset (sequence-disjoint); a template is the real map's excess over its
  range rows' median around a labelled object (columns within 1 bin of zero Doppler removed, so the
  static ridge is not carried along), converted to the simulator's dB scale and pasted as
  max(simulator, simulator row median + scale x excess). The template's own label offset comes with it.
Caches: data/tuned/<dataset>/bank_<name> (written once).
"""
import json
from pathlib import Path

import numpy as np
import torch

from src.damage import TrackDamage
from src.data import load, write_cache
from src.tuned import simulator, table
from src.versions import DATASETS, N_SIM, SEED0, best

K_R, K_D = 12, 8
BANK = {
    "dr_narrow": {"blur": (1.0, 1.5), "tail": (0.0, 2.0), "target_db": (-3.0, 3.0), "confusers": (0.75, 1.5),
                  "micro": (0.75, 1.0)},
    "dr_broad": {"blur": (1.0, 3.0), "tail": (0.0, 6.0), "target_db": (-9.0, 6.0), "confusers": (0.25, 3.0),
                 "micro": (0.0, 1.0)},
    "dr_weak": {"blur": (1.0, 1.5), "tail": (0.0, 2.0), "target_db": (-9.0, -3.0), "confusers": (0.75, 1.5),
                "micro": (0.75, 1.0)},
    "dr_strong": {"blur": (1.0, 1.5), "tail": (0.0, 2.0), "target_db": (3.0, 9.0), "confusers": (0.75, 1.5),
                  "micro": (0.75, 1.0)},
    "hybrid_x1": {"template_scale": 1.0},
    "hybrid_x1.5": {"template_scale": 1.5},
}


def cache_dir(dataset, name):
    return Path("data/tuned") / dataset / f"bank_{name}"


def _draw(ranges, i):
    g = torch.Generator().manual_seed(7_000_000 + i)
    u = lambda lo, hi: float(lo + (hi - lo) * torch.rand((), generator=g))
    return {k: u(*v) for k, v in ranges.items()}


def _dr_sequence(dataset, ranges, tables, i):
    p = _draw(ranges, i)
    sim = simulator(dataset, best(dataset), torch.Generator().manual_seed(SEED0 + i), None, tables,
                    mixin=TrackDamage, damage_generator=torch.Generator().manual_seed(i),
                    target_db=p["target_db"], blur=p["blur"], tail=p["tail"], confusers=p["confusers"],
                    micro=p["micro"])
    o = sim.gen_sequence()
    return o["x"].numpy(), o["labels"], {"draws": p}


def templates(dataset, exclude, zero=32):
    """Real target templates (excess over row median, z units) from the real training sequences
    not in `exclude`, with their label's range (bin) and |Doppler offset| from zero."""
    out = []
    for s, it in enumerate(load(DATASETS[dataset].real, "train")):
        if s in exclude:
            continue
        for t in range(it["x"].shape[0]):
            z = ((it["x"][t].float() - it["stats"]["mean"]) / it["stats"]["std"]).numpy()
            exc = z - np.median(z, axis=1, keepdims=True)
            n_r, n_d = z.shape
            for r, d in it["traj"][it["present"][:, t], t].tolist():
                ri, di = int(round(r)), int(round(d)) % n_d
                if ri < K_R or ri >= n_r - K_R:
                    continue
                cols = (di + np.arange(-K_D, K_D + 1)) % n_d
                patch = np.clip(exc[ri - K_R: ri + K_R + 1][:, cols], 0, None)
                ridge = np.abs((cols - zero + n_d / 2) % n_d - n_d / 2) <= 1
                patch[:, ridge] = 0.0
                out.append({"patch": patch, "dr": r - ri, "dd": ((d - di + n_d / 2) % n_d) - n_d / 2,
                            "range": r, "speed": abs(((d - zero + n_d / 2) % n_d) - n_d / 2), "seq": s})
    return out


def _hybrid_sequence(dataset, tables, temps, scale, sim_std, i):
    sim = simulator(dataset, best(dataset), torch.Generator().manual_seed(SEED0 + i), None, tables,
                    mixin=TrackDamage, damage_generator=torch.Generator().manual_seed(i), unrender_frac=1.0)
    o = sim.gen_sequence()
    x = o["x"].numpy().astype(np.float64)
    g = np.random.default_rng(i)
    feats = np.array([[tp["range"], tp["speed"]] for tp in temps])
    sd = feats.std(0) + 1e-9
    zero = 32
    used = []
    for t, frame in enumerate(o["labels"]):
        n_r, n_d = x[t].shape
        rowmed = np.median(x[t], axis=1)
        for lab in frame:
            r, d = lab["range_bin"], lab["doppler_bin"]
            sp = abs(((d - zero + n_d / 2) % n_d) - n_d / 2)
            near = np.argsort((((feats - [r, sp]) / sd) ** 2).sum(1))[:5]
            tp = temps[int(g.choice(near))]
            # place the template so that its label lands on this label
            ri, di = int(round(r - tp["dr"])), int(round(d - tp["dd"])) % n_d
            if ri < K_R or ri >= n_r - K_R:
                continue
            cols = (di + np.arange(-K_D, K_D + 1)) % n_d
            rows = np.arange(ri - K_R, ri + K_R + 1)
            block = x[t][np.ix_(rows, cols)]
            paste = rowmed[rows][:, None] + scale * sim_std * tp["patch"]
            x[t][np.ix_(rows, cols)] = np.maximum(block, paste)
            used.append(tp["seq"])
    return x.astype(np.float32), o["labels"], {"template_seqs": sorted(set(used))}


def render_bank(dataset, name, reference_seqs):
    """Write a bank candidate's cache (once). reference_seqs: real training sequences the metric's
    reference uses (templates never come from them)."""
    out = cache_dir(dataset, name)
    spec = BANK[name]
    done = {"name": name, "spec": spec, "switches": list(best(dataset)), "reference_seqs": list(map(int, reference_seqs))}
    if (out / "DONE").exists() and json.loads((out / "DONE").read_text()) == done:
        return out
    ds, tables = DATASETS[dataset], table(dataset)
    if name.startswith("dr_"):
        make = lambda i: _dr_sequence(dataset, spec, tables, i)
    else:
        temps = templates(dataset, set(map(int, reference_seqs)))
        sim_std = float(torch.load(Path("data/tuned") / dataset / "best" / "stats.pt")["std"])
        make = lambda i: _hybrid_sequence(dataset, tables, temps, spec["template_scale"], sim_std, i)
    write_cache(out, N_SIM, make, ds.grid, ds.stats_every, slot_by_rank=dataset != "radial")
    (out / "DONE").write_text(json.dumps(done))
    return out
