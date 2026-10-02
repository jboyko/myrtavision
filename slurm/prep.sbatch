#!/bin/bash
# Download every manifest image to scratch, then build the downscaled cache and
# the YOLO classification datasets (CPU only).
#     sbatch slurm/prep.sbatch
#SBATCH --job-name=myrta_prep
#SBATCH --account=jboyko0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/config.sh
activate_env

python tools/download_images.py --workers 16
python tools/build_datasets.py --workers "$SLURM_CPUS_PER_TASK"
