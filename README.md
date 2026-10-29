# myrtavision

Phenology scoring of herbarium sheets for three independent targets: bud,
flower, and fruit.

## Current direction

The current approach reuses LeafMachine2's herbarium-trained YOLOv5x backbone
and feature pyramid as a weakly supervised multiscale classifier:

1. Preserve the complete sheet with aspect-aware 1280-pixel preprocessing.
2. Read the stride-8, stride-16, and stride-32 feature maps instead of the
   detector's bounding-box outputs.
3. Produce spatial bud, flower, and fruit evidence at each feature scale.
4. Aggregate sparse evidence with normalized log-sum-exp pooling.
5. Train three independent sigmoid outputs from the existing sheet-level flags.

Raw inputs remain stable and shared:

- `scores.csv`: sheet-level phenology annotations (3,000 images; 135 unavailable, see `splits/unavailable.csv`)
- `splits/phenology_v3.csv`: specimen-grouped train/val/test split used by every method. It keeps every image of `phenology_v2.csv` in its v2 split and allocates only new specimens (`python -m leafmachine_classifier.prepare`).
- `data/images`: native-resolution herbarium images, downloaded by `tools/download_images.py`

LeafMachine2 is vendored at `third_party/LeafMachine2`, pinned to commit
`c5003798dfe716f176a4d87714260449664ec4a6`. The broad `PLANT_GroupAB_200`
checkpoint is `third_party/LeafMachine2/checkpoints/best.pt`.

The classifier implementation is under `leafmachine_classifier`. The split
manifest contains 1,329 training, 280 validation, and 281 held-out test images
(1,830 specimens) with no GBIF specimen crossing partitions.

```bash
.venv/bin/python -m leafmachine_classifier.train --name frozen_1280
.venv/bin/python -m leafmachine_classifier.evaluate \
  runs/leafmachine_classifier/frozen_1280/best.pt --split test
```

See `leafmachine_classifier/README.md` for architecture, staged fine-tuning,
and prediction details.

## Detector baseline

The existing LeafMachine detector can still be evaluated as a zero-shot
baseline or used to generate proposed boxes:

```bash
.venv/bin/python lm2_predict.py data/images --out lm2_predictions.csv \
  --labels runs/lm2/pseudo_labels
```

`lm2_predict.py` reports maximum detector confidence and counts for each organ;
these are not calibrated sheet-level probabilities. `lm2_eval.py` evaluates a
prediction CSV when image paths have combination-class parent directories.

## Training on Great Lakes

Several methods are compared on the same split and scored by the same code:

| method | what it is |
|---|---|
| `yolo_combo` | ultralytics YOLO-cls over the eight combinations; organ probability = summed softmax |
| `dino` | DINOv2 or DINOv3 ViT patch tokens + per-organ attention pooling (`dino_classifier`); backbone `frozen`, `lora`, or `last` N blocks trained |
| `yolo_binary`, `lm2_head` | still supported by `train.sbatch` but dropped after the v2 comparison |

Results from the 1,890-image v2 split are archived in `results/v2/`.

Code lives in the home-directory clone. Rebuildable data (`data/`, `datasets/`, `runs/`, `weights/`) is symlinked to `/scratch` (see `slurm/config.sh`). Durable outputs go to `results/` in home: weights and curves in `results/train/<name>/`, per-image val/test probabilities in `results/predictions/<name>/`.

```bash
git clone --recurse-submodules <repo> ~/myrtavision && cd ~/myrtavision
scp third_party/LeafMachine2/checkpoints/best.pt greatlakes:myrtavision/third_party/LeafMachine2/checkpoints/   # from the Mac
hf auth login                             # once, after accepting the DINOv3 license on Hugging Face
bash slurm/setup.sh                       # once: conda env, checkpoint check, scratch symlinks, YOLO and DINO weights
rsync -av --ignore-existing data/images/ greatlakes:myrtavision/data/images/   # from the Mac: images some hosts no longer serve
source slurm/config.sh && activate_env && python tools/download_images.py      # login node; compute-node proxy blocks some hosts
PREP=$(sbatch --parsable slurm/prep.sbatch)   # build cache + YOLO datasets; fails if any manifest image is missing
sbatch --dependency=afterok:$PREP --array=1-$(($(wc -l < slurm/experiments.tsv) - 1))%5 slurm/train.sbatch
sbatch slurm/eval.sbatch                  # ensembles + results/compare_val.csv
```

| File | Purpose |
|------|---------|
| `slurm/experiments.tsv` | One run per line (name, method, model, imgsz, batch, epochs, seed, extra args); add lines to try more |
| `tools/download_images.py` | Downloads manifest images (login node); failures go to `data/download_failures.csv` |
| `splits/unavailable.csv` | Scored images no host serves any more; `prepare.py` leaves them out of the split |
| `tools/build_datasets.py` | 1536 px cache and `datasets/combo`, `datasets/binary_<organ>` symlink trees |
| `yolo_classifier/train.py` | Trains `yolo_combo` / `yolo_binary` runs and writes val/test probabilities |
| `dino_classifier/train.py` | Trains `dino` runs (DINOv2 or v3) on the 1536 px cache and writes val/test probabilities |
| `dino_classifier/predict.py` | Scores new sheets with the final ensemble (`slurm/predict.sbatch`) |
| `tools/explain.py` | Grad-CAM bud/flower/fruit evidence maps for the final ensemble on val hits, misses, false alarms (`slurm/explain.sbatch`) |
| `tools/ensemble.py` | Averages member runs' probabilities for ensembles in `slurm/ensembles.tsv` |
| `tools/compare.py` | Per-organ F1 (thresholds chosen on val), average precision, exact match for every run |

Choose methods on `val` (`compare_val.csv`); run `sbatch slurm/eval.sbatch --test` only once, for the final choice. Frozen LeafMachine head checkpoints (~1 MB) are committed; fine-tuned LM2 and YOLO weights stay in `results/` but out of git.

## Scoring new sheets

The final model is `ens_dinov2_l_lora_x3`: three DINOv2-L LoRA runs (seeds
0-2) averaged, chosen on val. On the 426-sheet test split it scores macro F1
0.893 and macro AP 0.971 (YOLO y26l ensemble: 0.817 / 0.910). Its members and
val-chosen thresholds are in `dino_classifier/final_model.json`; the member
checkpoints live in `results/train/` on Great Lakes.

```bash
sbatch slurm/predict.sbatch results/new_sheets.csv data/new_sheets
```

The unscored specimens in the GBIF export are listed one image each in
`splits/pool.csv` (`tools/build_pool.py`, run on the Mac where
`training_data/` lives) and downloaded on a login node with
`python tools/download_images.py --manifest splits/pool.csv --images data/pool`.

The CSV has, per sheet, `<organ>_prob` (ensemble mean) and a 0/1 call per
organ at the val thresholds. The fruit threshold (0.94) transferred less well
to test than the ranking did, so keep the probabilities if fruit recall matters.

## Legacy approach

The previous YOLO26 eight-combination whole-sheet classifier, its generated
datasets, checkpoints, results, and scripts are isolated under
`legacy/yolo26_classification`. See its README for reproduction commands.

LeafMachine2 is GPL-3.0 licensed.
