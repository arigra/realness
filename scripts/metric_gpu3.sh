#!/bin/bash
# Phase 7.1 architecture check: prereg candidates + M1's leave-one-radar-out pick and the oracle
# (results/metric/scores/evaluation.json, reference 5_0, set "all"). Resumes.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
python3 scripts/architecture_check.py --dataset carrada \
  --candidates spec,tuned,study2:rangeshift1,study2:blur2,study2b:tail2,fair:noise+6,sim2:all-scenes >> logs/architecture_carrada.log 2>&1
python3 scripts/architecture_check.py --dataset raddet \
  --candidates spec,tuned,study2:rangeshift1,study2:blur2,study2b:tail2,study2:noise+6,study2:clutter-12 >> logs/architecture_raddet.log 2>&1
echo "gpu3 done $(date)" >> logs/metric_chain.done
