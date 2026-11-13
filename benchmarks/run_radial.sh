#!/usr/bin/env bash
# FFT-RadNet (official valeoai/RADIal code, unmodified) on RADIal; resumes from the newest epoch checkpoint.
set -euo pipefail
B=/truenas/home/arigra/permuter/ariGranevich/realness/benchmarks
cd $B/RADIal/FFTRadNet
last=$(ls $B/runs/radial/*/FFTRadNet*_epoch*.pth 2>/dev/null | sed -E 's/.*_epoch([0-9]+)_.*/\1 &/' | sort -n | tail -1 | cut -d' ' -f2- || true)
if [[ -n "$last" ]]; then echo "resume $last"; exec $B/.venv/bin/python 1-Train.py --config $B/runs/radial/config.json --resume "$last"; fi
exec $B/.venv/bin/python 1-Train.py --config $B/runs/radial/config.json
