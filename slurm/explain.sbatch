#!/bin/bash
# Grad-CAM evidence maps (bud / flower / fruit) for the final ensemble on val
# examples: confident hits, misses and false alarms per organ (CPU only).
#     sbatch slurm/explain.sbatch
#     sbatch slurm/explain.sbatch --per-group 5
#SBATCH --job-name=myrta_explain
#SBATCH --account=jboyko0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/config.sh
activate_env

python tools/explain.py --device cpu "$@"
