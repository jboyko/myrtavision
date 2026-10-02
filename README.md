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

- `scores.csv`: sheet-level phenology annotations (2,025 scored images)
- `splits/phenology_v2.csv`: specimen-grouped train/val/test split used by every method
- `data/images`: native-resolution herbarium images, downloaded by `tools/download_images.py`

LeafMachine2 is vendored at `third_party/LeafMachine2`, pinned to commit
`c5003798dfe716f176a4d87714260449664ec4a6`. The broad `PLANT_GroupAB_200`
checkpoint is `third_party/LeafMachine2/checkpoints/best.pt`.

The classifier implementation is under `leafmachine_classifier`. The split
manifest contains 1,404 training, 320 validation, and 301 held-out test images
(1,946 specimens) with no GBIF specimen crossing partitions.

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
| `lm2_head` | LeafMachine2 feature pyramid + evidence head (`leafmachine_classifier`) |
| `yolo_combo` | ultralytics YOLO-cls over the eight combinations; organ probability = summed softmax |
| `yolo_binary` | three present/absent YOLO-cls models, one per organ |
| `lm2_zeroshot` | LeafMachine2 detector confidences, no training (run by `eval.sbatch`) |

Code lives in the home-directory clone. Rebuildable data (`data/`, `datasets/`, `runs/`, `weights/`) is symlinked to `/scratch` (see `slurm/config.sh`). Durable outputs go to `results/` in home: weights and curves in `results/train/<name>/`, per-image val/test probabilities in `results/predictions/<name>/`.

```bash
git clone --recurse-submodules <repo> ~/myrtavision && cd ~/myrtavision
scp third_party/LeafMachine2/checkpoints/best.pt greatlakes:myrtavision/third_party/LeafMachine2/checkpoints/   # from the Mac
bash slurm/setup.sh                       # once: conda env, checkpoint check, scratch symlinks, YOLO weights
PREP=$(sbatch --parsable slurm/prep.sbatch)   # download images, build cache + YOLO datasets
sbatch --dependency=afterok:$PREP --array=1-$(($(wc -l < slurm/experiments.tsv) - 1))%5 slurm/train.sbatch
sbatch slurm/eval.sbatch                  # zero-shot baseline + results/compare_val.csv
```

| File | Purpose |
|------|---------|
| `slurm/experiments.tsv` | One run per line (name, method, model, imgsz, batch, epochs, seed, extra args); add lines to try more |
| `tools/download_images.py` | Downloads manifest images; failures go to `data/download_failures.csv` |
| `tools/build_datasets.py` | 1536 px cache and `datasets/combo`, `datasets/binary_<organ>` symlink trees |
| `yolo_classifier/train.py` | Trains `yolo_combo` / `yolo_binary` runs and writes val/test probabilities |
| `tools/compare.py` | Per-organ F1 (thresholds chosen on val), average precision, exact match for every run |

Choose methods on `val` (`compare_val.csv`); run `sbatch slurm/eval.sbatch --test` only once, for the final choice. LeafMachine head checkpoints (~1 MB) are committed; YOLO weights stay in `results/` but out of git.

## Legacy approach

The previous YOLO26 eight-combination whole-sheet classifier, its generated
datasets, checkpoints, results, and scripts are isolated under
`legacy/yolo26_classification`. See its README for reproduction commands.

LeafMachine2 is GPL-3.0 licensed.
