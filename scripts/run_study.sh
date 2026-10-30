#!/usr/bin/env bash
# Run (or resume) one damage study: bash scripts/run_study.sh <suite> <dataset>,
# e.g. fair radial. Writes a manifest of the code and data first; a heartbeat
# file keeps two copies of the same study apart.
set -euo pipefail
cd "$(dirname "$0")/.."
suite=$1 dataset=$2
PY="${PY:-$(command -v python3)}"
LOG=logs/${suite}_${dataset}.log
LOCK=logs/${suite}_${dataset}.heartbeat
export PYTHONPATH=.
mkdir -p logs results/manifests
say() { echo "$(date '+%F %T') | $*" >> "$LOG"; }
if [ -f "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -lt 300 ]; then
  say "REFUSED: another copy is running ($(cat "$LOCK"))"; exit 1
fi
echo "$(hostname -s) pid $$" > "$LOCK"
( while kill -0 $$ 2>/dev/null; do touch "$LOCK"; sleep 60; done ) &
trap 'rm -f "$LOCK"' EXIT
say "start"
"$PY" scripts/write_manifest.py --allow-dirty --paths src scripts \
    --out "results/manifests/${suite}_${dataset}_$(date +%Y%m%d_%H%M%S).json" >> "$LOG" 2>&1
"$PY" scripts/damage_study.py --suite "$suite" --dataset "$dataset" >> "$LOG" 2>&1
say "DONE"
