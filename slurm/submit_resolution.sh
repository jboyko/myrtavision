#!/bin/bash
# DINOv3-L LoRA at 1536, 2048, 2560 and 3072 px, three seeds each, end to end:
# build the 3072 px cache, train experiments.tsv tasks 17-27 (file lines 18-28;
# dinov3_l_lora seed 0 at 1536 already exists), then the ensembles and val and
# test scores. Run once from the login node:
#     bash slurm/submit_resolution.sh
# Training waits for the cache; scoring waits for every training task, finished
# or failed, so one bad run doesn't block the rest. Longest runs are listed
# first so they start first. 3072 px is estimated at ~3.5 days per run on a
# V100; --time leaves room for that (the gpu partition allows 14 days).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
prep=$(sbatch --parsable slurm/prep.sbatch --maxpx 3072 --cache-only)
train=$(sbatch --parsable --dependency=afterok:"$prep" --time=8-00:00:00 --array=17-27%5 slurm/train.sbatch)
evaluate=$(sbatch --parsable --dependency=afterany:"$train" slurm/eval.sbatch --test)
echo "cache $prep -> training array $train (11 runs, 5 at a time) -> scoring $evaluate"
echo "results: results/compare_val.csv and results/compare_test.csv when $evaluate finishes"
