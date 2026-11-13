#!/usr/bin/env bash
# Recipe search on the DEV protocol (train scenes only; scenes 0+13 held out; no test frames).
# Usage: dev_run.sh <name> <epochs> [extra bev-train args...]; then evaluates best.pt on the dev split.
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1 epochs=$2; shift 2
out=runs/dev/$name
.venv/bin/python -m radsimreal bev-train --data runs/raddet_prepared_dev --out $out --base-channels 64 --batch-size 8 \
  --device cuda --threads 4 --workers 6 --seed 2024 --epochs $epochs "$@"
for agg in global_pr frame_mean; do
  .venv/bin/python -m radsimreal bev-evaluate --data runs/raddet_prepared_dev --checkpoint $out/best.pt --split val \
    --device cuda --ap-aggregation $agg --out $out/dev_$agg.json > /dev/null
done
.venv/bin/python -c "
import json
for a in ('global_pr','frame_mean'):
    m=json.load(open('$out/dev_'+a+'.json')); print('$name',a,{k:round(v*100,1) for k,v in m.items() if k.startswith('AP')})"
