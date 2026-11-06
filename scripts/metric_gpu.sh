#!/bin/bash
# The GPU part of the metric experiments, in order; each step resumes (finished runs are skipped).
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
for d in carrada raddet; do python3 scripts/proxy_detector.py --dataset $d >> logs/proxy_$d.log 2>&1; done
for d in carrada raddet; do python3 scripts/augment_study.py --dataset $d --stage A >> logs/augment_$d.log 2>&1; done
python3 scripts/proxy_detector.py --dataset radial >> logs/proxy_radial.log 2>&1
echo "gpu chain done $(date)" >> logs/metric_chain.done
