#!/bin/bash
# Phase 6 after the selections were committed (03de8c3): train every bank candidate (3 seeds) and run
# the proxy on the bank; then the report. Resumes.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
for d in carrada raddet; do
  python3 scripts/prospective.py train --dataset $d >> logs/bank_train_$d.log 2>&1
  python3 scripts/proxy_detector.py --dataset $d --candidates bank >> logs/bank_proxy_$d.log 2>&1
  python3 scripts/prospective.py report --dataset $d >> logs/bank_report_$d.log 2>&1
done
echo "gpu4 done $(date)" >> logs/metric_chain.done
