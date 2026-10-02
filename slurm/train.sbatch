#!/bin/bash
# Train every run listed in slurm/experiments.tsv, one array task per line.
#     sbatch --array=1-$(($(wc -l < slurm/experiments.tsv) - 1))%5 slurm/train.sbatch
# (%5 = the gpu partition's 5-GPU-per-account limit.) Batch sizes in experiments.tsv
# are sized for 16 GB V100s. To rerun single runs, pass their line numbers, e.g.
# --array=3,5 (line 1 is the header). Each task reads its line when it starts.
#SBATCH --job-name=myrta_train
#SBATCH --account=jboyko0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/config.sh
activate_env

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" slurm/experiments.tsv)
IFS=$'\t' read -r NAME METHOD MODEL IMGSZ BATCH EPOCHS SEED EXTRA <<< "$LINE"
[ "$EXTRA" = "-" ] && EXTRA=""
echo "Run $NAME: $METHOD, model $MODEL, imgsz $IMGSZ, batch $BATCH, epochs $EPOCHS, seed $SEED, extra '$EXTRA' on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

case "$METHOD" in
lm2_head)
    # shellcheck disable=SC2086  # EXTRA holds several arguments
    python -m leafmachine_classifier.train --name "$NAME" --image-size "$IMGSZ" \
        --batch-size "$BATCH" --epochs "$EPOCHS" --seed "$SEED" \
        --workers "$SLURM_CPUS_PER_TASK" --force $EXTRA
    # Keep what matters in home; runs/ on scratch can be purged.
    OUT="results/train/$NAME"
    mkdir -p "$OUT" "results/predictions/$NAME"
    cp "runs/leafmachine_classifier/$NAME/best.pt" "runs/leafmachine_classifier/$NAME/history.csv" "$OUT/"
    for SPLIT in val test; do
        python -m leafmachine_classifier.evaluate "$OUT/best.pt" --split "$SPLIT" \
            --batch-size "$BATCH" --workers "$SLURM_CPUS_PER_TASK" \
            --out "results/predictions/$NAME/$SPLIT.csv"
    done
    ;;
yolo_combo|yolo_binary)
    # shellcheck disable=SC2086
    python yolo_classifier/train.py --task "${METHOD#yolo_}" --model "weights/$MODEL" \
        --name "$NAME" --imgsz "$IMGSZ" --batch "$BATCH" --epochs "$EPOCHS" \
        --seed "$SEED" --workers "$SLURM_CPUS_PER_TASK" $EXTRA
    ;;
*)
    echo "Unknown method $METHOD" >&2
    exit 1
    ;;
esac
echo "Kept weights in results/train/$NAME and val/test probabilities in results/predictions/$NAME"
