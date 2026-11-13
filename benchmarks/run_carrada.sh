#!/usr/bin/env bash
# TMVA-Net (official MVRSS code + resume patch) on CARRADA, paper config (300 epochs max, early stop after 30 epochs without val improvement); resumes per epoch.
set -euo pipefail
B=/truenas/home/arigra/permuter/ariGranevich/realness/benchmarks
cd $B/MVRSS
export MVRSS_RESUME=$B/runs/carrada/resume.pt
export MVRSS_PATIENCE=30   # early stop: 30 epochs without a better val prec
exec $B/.venv/bin/python mvrss/train.py --cfg $B/runs/carrada/tmvanet.json
