#!/bin/bash
#SBATCH --job-name=realness_damage_radial
#SBATCH --partition=shared_3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=24:00:00
#SBATCH --output=/truenas/home/arigra/jobs/realness_damage_radial_%j.out
#SBATCH --error=/truenas/home/arigra/jobs/realness_damage_radial_%j.err
# RADIal fair damage study (docs/notes/2026-10-28-damage-spread-fair.md), resuming
# wherever it stopped. Safe to resubmit. If another copy is still running (e.g. in
# an interactive session), this waits for its heartbeat to go stale.
# Submit from the login host ece-hpc:  sbatch scripts/damage_radial.sbatch
cd /truenas/home/arigra/permuter/ariGranevich/realness
LOCK=logs/damage_fair_radial.heartbeat
while [ -f "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -lt 300 ]; do sleep 120; done
bash scripts/run_damage_radial.sh
