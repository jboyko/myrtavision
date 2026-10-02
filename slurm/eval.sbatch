#!/bin/bash
# Zero-shot LeafMachine2 detector baseline on val/test, then score every run with
# one set of metrics (thresholds chosen on val).
#     sbatch slurm/eval.sbatch             # val only; choose methods on this
#     sbatch slurm/eval.sbatch --test      # also score held-out test; run once
#SBATCH --job-name=myrta_eval
#SBATCH --account=jboyko0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/config.sh
activate_env

mkdir -p results/predictions/lm2_zeroshot
for SPLIT in val test; do
    OUT="results/predictions/lm2_zeroshot/$SPLIT.csv"
    [ -f "$OUT" ] || python lm2_predict.py "datasets/combo/$SPLIT" --device 0 --out "$OUT"
done
python tools/compare.py "$@"
