#!/bin/bash
# Restart whatever part of Metric v2 (docs/notes/2026-11-06-metric-v2-preregistration.md) is missing. Every step
# resumes from disk (finished candidates / seeds are skipped); a running step is left alone.
#   bash scripts/resume_metric_v2.sh pre      # features, bank-v2 render, bank-v2 features, proxy (before the freeze)
#   bash scripts/resume_metric_v2.sh post     # full training and augmentation (only after the freeze commit)
#   add --dry to see what would start
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
STAGE=$1; DRY=$2
running() { pgrep -f "$1" > /dev/null; }
launch() {        # name, log, command...
    local name=$1 log=$2; shift 2
    if [ "$DRY" = "--dry" ]; then echo "would start: $name"; return; fi
    echo "starting: $name"
    OMP_NUM_THREADS=1 nohup "$@" >> "$log" 2>&1 &
}
if [ "$STAGE" = "pre" ]; then
    for d in carrada raddet; do
        running "[b]uild_metric_v2_features.py --dataset $d --candidates all" ||
            for k in 0 1 2 3; do launch "features $d $k" logs/metric_v2_features_${d}_$k.log \
                python3 scripts/build_metric_v2_features.py --dataset $d --candidates all --part $k/4; done
        running "[p]rospective_v2.py render --dataset $d" ||
            launch "render $d" logs/metric_v2_render_$d.log nice -n 10 python3 scripts/prospective_v2.py render --dataset $d
    done
    running "[b]uild_metric_v2_features.py --dataset radial" ||
        for k in 0 1 2 3 4; do launch "features radial $k" logs/metric_v2_features_radial_$k.log \
            python3 scripts/build_metric_v2_features.py --dataset radial --candidates all --part $k/5; done
elif [ "$STAGE" = "bank" ]; then
    for d in carrada raddet; do
        running "[b]uild_metric_v2_features.py --dataset $d --candidates bank2" ||
            launch "bank2 features $d" logs/metric_v2_bank2_features_$d.log \
                python3 scripts/build_metric_v2_features.py --dataset $d --candidates bank2
    done
    running "[m]etric_v2_proxy" || launch "proxy (both radars)" logs/metric_v2_proxy.log \
        bash -c 'for d in carrada raddet; do python3 scripts/prospective_v2.py proxy --dataset $d; done # metric_v2_proxy'
elif [ "$STAGE" = "post" ]; then
    git log --format=%s | grep -q "bank-v2 freeze" || { echo "no freeze commit: refusing to train"; exit 1; }
    running "[m]etric_v2_gpu_post" || launch "GPU: train carrada, raddet; augment raddet" logs/metric_v2_gpu_post.log \
        bash -c 'python3 scripts/prospective_v2.py train --dataset carrada; python3 scripts/prospective_v2.py train --dataset raddet; python3 scripts/prospective_v2.py augment --dataset raddet; echo done # metric_v2_gpu_post'
else
    echo "usage: $0 pre|bank|post [--dry]"
fi
