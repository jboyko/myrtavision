#!/bin/bash
# Score new sheets with the final ensemble (dino_classifier/final_model.json).
#     sbatch slurm/predict.sbatch <out.csv> <image file or folder>...
# e.g. sbatch slurm/predict.sbatch results/new_sheets.csv data/new_sheets
# Roughly 0.3 s per sheet per member on a V100 (estimated from training speed),
# so ~1 s per sheet for the 3-member ensemble; raise --time for big batches.
#SBATCH --job-name=myrta_predict
#SBATCH --account=jboyko0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/config.sh
activate_env

OUT=$1
shift
python -m dino_classifier.predict "$@" --out "$OUT" --workers "$SLURM_CPUS_PER_TASK"
