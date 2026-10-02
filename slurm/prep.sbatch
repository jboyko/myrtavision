#!/bin/bash
# Build the downscaled cache and the YOLO classification datasets (CPU only).
# Images must already be in data/images: download them on the login node first
# (the compute-node proxy blocks some hosts). Fails if any manifest image is
# missing, so a chained training array never starts on a partial set.
#     sbatch slurm/prep.sbatch
#SBATCH --job-name=myrta_prep
#SBATCH --account=jboyko0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/config.sh
activate_env

python tools/build_datasets.py --workers "$SLURM_CPUS_PER_TASK"
