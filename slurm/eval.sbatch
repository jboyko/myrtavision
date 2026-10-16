#!/bin/bash
# Ensembles from slurm/ensembles.tsv, then score every run with one set of
# metrics (thresholds chosen on val).
#     sbatch slurm/eval.sbatch             # val only; choose methods on this
#     sbatch slurm/eval.sbatch --test      # also score held-out test; run once
#SBATCH --job-name=myrta_eval
#SBATCH --account=jboyko0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/config.sh
activate_env

python tools/ensemble.py
python tools/compare.py "$@"
