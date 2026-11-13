#!/usr/bin/env bash
# Step 1: RADDet, U-Net, trained on real vs on RadSimReal(CARLA) synthetic; both tested on real test (scenes 9, 11).
# Hyperparameters fixed BEFORE any test evaluation and identical for both training sets:
#   U-Net base 64 channels (Ronneberger U-Net cited by the paper), batch 8, AdamW lr 1e-3 (package default),
#   40 epochs TOTAL, best checkpoint by validation loss on the run's own validation split
#   (real: scene_4; synthetic: held-out CARLA drives). No test data in selection.
# Evaluation: AP@0.1/0.3/0.5, global PR area (package default), matching best_overall, score>=0.05, NMS 0.3.
# Every stage resumes: rerun the same command. bev-train's --epochs counts ADDITIONAL epochs after a
# resume, so the remaining count is computed from the run's history to keep 40 in total.
# Usage: step1_raddet_unet.sh [real|sim|all]   (SIM_DATA overrides the synthetic dataset)
set -euo pipefail
cd "$(dirname "$0")/.."
P=.venv/bin/python
EPOCHS=40
COMMON="--model unet --base-channels 64 --batch-size 8 --device cuda --threads 8 --seed 2024"
SIM_DATA=${SIM_DATA:-runs/raddet_carla_synthetic_vehicle}

train() {  # train <data> <out>
  local data=$1 out=$2 done_epochs=0
  if [[ -f $out/history.json ]]; then
    done_epochs=$($P -c "import json;h=json.load(open('$out/history.json'));print(h[-1]['epoch']+1 if h else 0)")
  fi
  local remaining=$((EPOCHS - done_epochs))
  if (( remaining <= 0 )); then echo "$out: $done_epochs epochs done (>= $EPOCHS), skipping training"; return; fi
  local resume=(); [[ -f $out/last.pt ]] && resume=(--resume $out/last.pt)
  $P -m radsimreal bev-train --data $data --out $out $COMMON --epochs $remaining "${resume[@]}"
}

stage=${1:-all}

if [[ $stage == real || $stage == all ]]; then
  R=runs/step1/unet_real
  train runs/raddet_prepared_v $R
  $P -m radsimreal bev-evaluate --data runs/raddet_prepared_v --checkpoint $R/best.pt --split test \
     --device cuda --out $R/test_metrics.json
fi

if [[ $stage == sim || $stage == all ]]; then
  S=runs/step1/unet_sim_vehicle
  train $SIM_DATA $S
  $P -m radsimreal bev-evaluate --data runs/raddet_prepared_v --checkpoint $S/best.pt --split test \
     --device cuda --allow-label-domain-change --out $S/test_metrics.json
fi
