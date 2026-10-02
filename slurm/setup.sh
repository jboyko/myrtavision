#!/bin/bash
# One-time setup on a Great Lakes login node, run from the repo root (~/myrtavision):
#     bash slurm/setup.sh
# Creates the conda env, fetches the LeafMachine2 submodule, puts the big folders
# on /scratch (symlinked into the repo), and downloads pretrained YOLO weights.
# The LeafMachine2 checkpoint is not downloadable; scp it from the Mac first.
set -euo pipefail
source slurm/config.sh

module load python3.11-anaconda/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "$ENV_NAME" python=3.13
fi
conda activate "$ENV_NAME"
pip install torch torchvision --index-url "$TORCH_INDEX"
pip install -r requirements.txt

git submodule update --init third_party/LeafMachine2
CKPT=third_party/LeafMachine2/checkpoints/best.pt
if [ ! -f "$CKPT" ]; then
    echo "Missing $CKPT; scp it from the Mac (see README), then rerun." >&2
    exit 1
fi
echo "$LM2_SHA256  $CKPT" | sha256sum -c -

# Rebuildable data lives on scratch (purged after 60 days without access);
# results/ stays in home.
for d in data datasets runs weights; do
    mkdir -p "$SCRATCH_DIR/$d"
    if [ ! -e "$d" ]; then
        ln -s "$SCRATCH_DIR/$d" "$d"
    fi
done
mkdir -p results logs

cd weights
for w in $PRETRAINED; do
    [ -f "$w" ] || python -c "from ultralytics import YOLO; YOLO('$w')"
done
cd ..

python -c "import torch, ultralytics; print('torch', torch.__version__, '| CUDA build', torch.version.cuda, '| ultralytics', ultralytics.__version__)"
echo "Setup done. Scratch: $SCRATCH_DIR"
