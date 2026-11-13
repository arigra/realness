#!/usr/bin/env bash
# RADDet (official ZhangAoCanada/RADDet code, TF 2.15 instead of 2.3) 3D head; restores the latest checkpoint itself.
set -euo pipefail
B=/truenas/home/arigra/permuter/ariGranevich/realness/benchmarks
cd $B/RADDet
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2
export RADDET_RAW_ROOT="/truenas/home/arigra/permuter/ariGranevich/radar data/RADDet/raw" RADDET_LOGMAG_CACHE=/tmp/arigra_raddet_logmag   # local input cache, see scripts/cache_raddet_logmag.py
exec $B/.venv_tf/bin/python train.py
