#!/usr/bin/env bash
# docs/notes/2026-10-28-damage-spread-fair.md, RADIal: the fair damage suite, both
# settings, 3 seeds. Resumable per run; heartbeat keeps two copies apart.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(command -v python3)}"
LOG=logs/damage_fair_radial.log
LOCK=logs/damage_fair_radial.heartbeat
export PYTHONPATH=.
say() { echo "$(date '+%F %T') | $*" >> "$LOG"; }
if [ -f "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -lt 300 ]; then
  say "REFUSED: another copy is running ($(cat "$LOCK"))"; exit 1
fi
echo "$(hostname -s) pid $$" > "$LOCK"
( while kill -0 $$ 2>/dev/null; do touch "$LOCK"; sleep 60; done ) &
trap 'rm -f "$LOCK"' EXIT
say "start"
mkdir -p samples/damage_manifests
"$PY" scripts/write_manifest.py --allow-dirty --paths configs src scripts \
    --out "samples/damage_manifests/fair_radial_$(date +%Y%m%d_%H%M%S).json" >> "$LOG" 2>&1
"$PY" scripts/damage_study.py --suite fair --dataset radial \
    --out samples/damage_fair_radial_study.json --verdict samples/damage_fair_radial_verdict.json \
    --dets-dir samples/damage_fair_radial_dets >> "$LOG" 2>&1
say "DONE"
