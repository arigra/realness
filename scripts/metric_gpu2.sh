#!/bin/bash
# GPU queue, part 2 (after scripts/metric_gpu.sh / the first chain): Phase 5 Stage B (RADDet, budgets
# 1 and 10), then the Phase 7.2 learned-feature encoders and distances. Each step resumes.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
while pgrep -f "metric_chain.sh|metric_gpu.sh" > /dev/null; do sleep 30; done
for b in 1 10; do python3 scripts/augment_study.py --dataset raddet --stage B --budget $b >> logs/augment_raddet_B.log 2>&1; done
for d in carrada raddet; do python3 scripts/learned_features.py --dataset $d >> logs/learned_$d.log 2>&1; done
echo "gpu2 done $(date)" >> logs/metric_chain.done
