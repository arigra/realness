#!/bin/bash
# Restart whatever part of the metric experiments (docs/notes/2026-11-06-metric-experiments.md) is not
# running. Every step resumes from what is on disk (finished candidates / seeds are skipped), and a step
# that is already running is left alone, so this is safe to run at any time, as often as needed.
#   bash scripts/resume_metric.sh          # launch what is missing
#   bash scripts/resume_metric.sh --dry    # only say what it would launch
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
DRY=$1
running() { pgrep -f "$1" > /dev/null; }
# processes paused with SIGSTOP (e.g. before a session change): continue them first
for p in $(pgrep -f "scripts/[p]roxy_detector|scripts/[c]lassify_controls_v2|scripts/[p]rospective|scripts/[a]ugment_study|scripts/[l]earned_features|scripts/[b]uild_metric_features|scripts/[a]rchitecture_check|[m]etric_gpu[23]?.sh"); do
    [ "$(ps -o stat= -p $p | cut -c1)" = "T" ] && { [ "$DRY" = "--dry" ] && echo "would continue $p" || kill -CONT $p; }
done
launch() {        # name, log, command...
    local name=$1 log=$2; shift 2
    if [ "$DRY" = "--dry" ]; then echo "would start: $name"; return; fi
    echo "starting: $name"
    OMP_NUM_THREADS=4 nohup "$@" >> "$log" 2>&1 &
}

# CPU: Phase 1 features (cached candidates are skipped); a dataset whose every usable candidate has a
# current vector is done
features_done() {
    python3 -c "
import json, os, sys
from scripts.build_metric_features import feature_key, candidate_dir
key = feature_key('$1')
rows = [r for r in json.load(open('results/metric/candidates.json'))['candidates'] if r['dataset'] == '$1' and r['usable']]
p = lambda r: candidate_dir('$1', r['candidate']) / 'reference_50_0.json'
sys.exit(0 if all(p(r).exists() and json.load(open(p(r))).get('feature_key') == key for r in rows) else 1)"
}
for d in carrada raddet; do
    features_done $d || running "build_metric_features.py --dataset $d --candidates all" ||
        launch "features $d" logs/metric_features_$d.log python3 scripts/build_metric_features.py --dataset $d --candidates all --check
done
if ! features_done radial && ! running "build_metric_features.py --dataset radial"; then
    python3 -c "
import json
reg=[r for r in json.load(open('results/metric/candidates.json'))['candidates'] if r['dataset']=='radial' and r['usable']]
ids=[r['candidate'] for r in reg]
for k in range(3): open(f'logs/radial_ids_{k}.txt','w').write(','.join(ids[k::3]))"
    for k in 0 1 2; do
        launch "features radial part $k" logs/metric_features_radial_$k.log \
            python3 scripts/build_metric_features.py --dataset radial --candidates "$(cat logs/radial_ids_$k.txt)"
    done
fi

# CPU: Phase 6 bank rendering (until results/metric/bank.json exists)
[ -f results/metric/bank.json ] || running "prospective.py render" ||
    launch "bank render" logs/bank_render.log nice -n 10 python3 scripts/prospective.py render

# GPU: Phase 8 controls, and the Phase 4-5 chain (one after the other)
grep -q small_mask_matched results/metric/classification/controls_v2.json 2>/dev/null || running "[c]lassify_controls_v2.py" ||
    launch "classification controls" logs/classify_controls_v2.log python3 scripts/classify_controls_v2.py
if [ ! -f logs/metric_chain.done ] && ! running "[m]etric_chain.sh" && ! running "[m]etric_gpu.sh"; then
    launch "GPU chain (proxy carrada, raddet; augmentation; proxy radial)" logs/metric_gpu.log bash scripts/metric_gpu.sh
fi
grep -q "gpu2 done" logs/metric_chain.done 2>/dev/null || running "[m]etric_gpu2.sh" ||
    launch "GPU queue 2 (Stage B RADDet; learned features)" logs/metric_gpu2.log bash scripts/metric_gpu2.sh
grep -q "gpu3 done" logs/metric_chain.done 2>/dev/null || running "[m]etric_gpu3.sh" ||
    launch "GPU queue 3 (Phase 7.1 transformer check)" logs/metric_gpu3.log bash scripts/metric_gpu3.sh
