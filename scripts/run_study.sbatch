#!/bin/bash
#SBATCH --job-name=realness
#SBATCH --partition=shared_3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=24:00:00
#SBATCH --output=/truenas/home/arigra/jobs/realness_%j.out
#SBATCH --error=/truenas/home/arigra/jobs/realness_%j.err
# Run or resume a damage study on the cluster; safe to resubmit. If a copy is
# already running (e.g. in an interactive session), waits for it to stop.
# From ece-hpc:  sbatch scripts/run_study.sbatch fair radial
cd /truenas/home/arigra/permuter/ariGranevich/realness
LOCK=logs/$1_$2.heartbeat
while [ -f "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -lt 300 ]; do sleep 120; done
bash scripts/run_study.sh "$1" "$2"
