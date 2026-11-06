#!/usr/bin/env bash
# Step 1 to 2 for one dataset, unattended: the sim2 ablation, the best simulator
# (scripts/choose_best.py), Study 2, Study 2b and the label budget
# (docs/notes/2026-11-02-simulators.md, docs/notes/2026-11-02-study2b.md).
#   bash scripts/pipeline.sh carrada clean,all,all-processing,...
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
d=$1 versions=$2
bash scripts/run_study.sh sim2 "$d" --settings only --versions "$versions"
python3 scripts/choose_best.py --dataset "$d" > "logs/best_$d.log"
bash scripts/run_study.sh study2 "$d" --settings only
bash scripts/run_study.sh study2b "$d" --settings only
for n in 1 5 10 25 50; do
  python3 scripts/damage_study.py --suite budget --dataset "$d" --settings real,mix --versions clean --real-n $n >> "logs/budget_$d.log" 2>&1
done
echo "$(date '+%F %T') pipeline $d done" >> "logs/pipeline_$d.log"
