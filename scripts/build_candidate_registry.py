"""Registry of every synthetic candidate with real-test results (metric plan, Phase 0.3).

One row per distinct synthetic pool (dataset + the data it is made of). The same pool reached by
more than one suite (e.g. the best tuned simulator = sim2's chosen variant = the clean of study2,
study2b and budget) is one candidate; every (suite, version, setting, real budget) result entry is
listed under it in `entries`, so each existing result belongs to exactly one candidate. The first
entry fills the plan's flat fields (suite, version, setting, result_path).

Bases: "spec" = the specification simulator (sim2 clean; v1, fair, smoothall and tune act on it), "tuned" =
the best tuned simulator (results/tuning/best.json; study2 and study2b act on it), "fixes" = the
other tuned variants (sim2: all fixes, and all minus one fix).
`usable` = has synthetic-only results scored on the dataset's current real split (tune_raddet was
scored on RADDet's official, interleaved split and is kept for the record only).
Writes results/metric/candidates.json.

    PYTHONPATH=. python3 scripts/build_candidate_registry.py
"""
import json
import re
from pathlib import Path

from src.provenance import provenance
from src.tolerance import EXTRA, FAMILIES
from src.versions import DATASETS, N_SIM, SUITES, best

OUT = Path("results/metric/candidates.json")
FAIR_FAMILY = {"contrast": "contrast", "unrender": "unrender", "extra": "extra", "smooth": "label_box_smoothing",
               "noise": "noise", "range": "rangeshift", "doppler": "dopplershift", "drop": "label_drop",
               "blink": "blink"}


def _level(name):
    m = re.search(r"([+-]?\d+(?:\.\d+)?)$", name)
    return float(m.group(1)) if m else None


def _family(suite, name):
    if suite in ("study2", "study2b"):
        for f, names in FAMILIES.items():
            if name in names:
                return f
        if name in EXTRA:
            return "sidelobes-"
    if suite in ("fair", "v1"):
        if name.startswith("clutter"):
            return "clutter+" if "+" in name else "clutter-"
        return FAIR_FAMILY[re.match(r"[a-z]+", name).group(0)]
    if suite == "smoothall":
        return "smoothing"
    raise KeyError((suite, name))


def identify(dataset, suite, name):
    """(key of the pool's content, candidate id, base, family, level, cache)."""
    ds = DATASETS[dataset]
    kind, value = SUITES[suite][0][name]
    on_best = suite in ("study2", "study2b", "budget")
    if name == "clean":
        if on_best:
            return ("tuned2", tuple(sorted(best(dataset)))), "tuned", "tuned", "natural", None, f"data/tuned/{dataset}/best"
        return ("spec",), "spec", "spec", "natural", None, ds.sim
    if suite == "tune":                     # RADDet's first tuned simulator (src/raddet_tuned.py)
        return (("spec", kind, json.dumps(value, sort_keys=True)), f"tune:{name}", "spec", "fixes_v0", None,
                f"{ds.damage}/{name}")
    if suite == "sim2":
        switches = tuple(sorted(value))
        if switches == tuple(sorted(best(dataset))):
            return ("tuned2", switches), "tuned", "tuned", "natural", None, f"data/tuned/{dataset}/best"
        return ("tuned2", switches), f"sim2:{name}", "fixes", "fixes", None, f"data/tuned/{dataset}/{name}"
    base = "tuned" if on_best else "spec"
    cache = (f"data/tuned/{dataset}/damage_{name}" if kind == "t2" else
             f"{ds.damage}/{name}" if kind not in ("range", "doppler", "smoothall", "smooth", "drop") else
             f"(on the {base} cache, applied at load: {kind} {value})")
    return (base, kind, json.dumps(value, sort_keys=True)), f"{suite}:{name}", base, _family(suite, name), _level(name), cache


def entries(dataset):
    """Every (suite, version, setting, real budget) with results, and its result file."""
    out = []
    for p in sorted(Path("results").glob(f"*_{dataset}_runs.json")) + \
            sorted(Path("results").glob(f"budget_{dataset}_real*_runs.json")):
        m = re.match(rf"([a-z0-9]+)_{dataset}(?:_real(\d+))?_runs\.json$", p.name)
        if not m or m.group(1) not in SUITES:
            continue                        # reference_real, classify: not synthetic candidates
        suite, real_n = m.group(1), int(m.group(2)) if m.group(2) else None
        r = json.loads(p.read_text())
        for setting, versions in r["runs"].items():
            if setting == "real":
                continue                    # real only: a reference, not a candidate
            for name, seeds in versions.items():
                out.append({"suite": suite, "version": name, "setting": setting,
                            "real_n": real_n if setting == "mix" else None,
                            "real_n_mix": (real_n or 50) if setting == "mix" else None,
                            "result_path": str(p), "n_seeds": len(seeds),
                            "has_val": name in r.get("val", {}).get(setting, {}),
                            "real_cache": r.get("real"),
                            # scored on the dataset's current real split (RADDet: scene-disjoint)
                            "usable": r.get("real") == DATASETS[dataset].real and len(seeds) > 0})
    return out


def main():
    rows, n_entries = [], 0
    for dataset in DATASETS:
        by_key = {}
        for e in entries(dataset):
            n_entries += 1
            key, cid, base, fam, level, cache = identify(dataset, e["suite"], e["version"])
            if key not in by_key:
                by_key[key] = {"dataset": dataset, "candidate": cid, "suite": e["suite"], "version": e["version"],
                               "family": fam, "level": level, "base": base, "setting": e["setting"],
                               "n_sim_sequences": N_SIM, "result_path": e["result_path"],
                               "has_test_ap": e["n_seeds"] > 0, "cache": cache, "entries": []}
            by_key[key]["entries"].append(e)
        rows += list(by_key.values())
    # every result entry exactly once
    seen = [(r["dataset"], e["suite"], e["version"], e["setting"], e["real_n"]) for r in rows for e in r["entries"]]
    assert len(seen) == len(set(seen)) == n_entries
    for r in rows:                      # the "only" entry first, when there is one
        r["entries"].sort(key=lambda e: (e["setting"] != "only", e["suite"] != r["suite"]))
        first = r["entries"][0]
        r.update(suite=first["suite"], version=first["version"], setting=first["setting"],
                 result_path=first["result_path"],
                 usable=any(e["usable"] and e["setting"] == "only" for e in r["entries"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"provenance": provenance(), "n_result_entries": n_entries,
                               "candidates": rows}, indent=1))
    for ds in DATASETS:
        sub = [r for r in rows if r["dataset"] == ds]
        fams = sorted({r["family"] for r in sub})
        print(f"{ds}: {len(sub)} candidates ({sum(r['usable'] for r in sub)} with usable synthetic-only "
              f"test AP), {sum(len(r['entries']) for r in sub)} result entries; families {fams}")


if __name__ == "__main__":
    main()
