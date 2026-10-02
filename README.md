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

- `scores.csv`: sheet-level phenology annotations
- `data/images`: 998 native-resolution herbarium images

LeafMachine2 is vendored at `third_party/LeafMachine2`, pinned to commit
`c5003798dfe716f176a4d87714260449664ec4a6`. The broad `PLANT_GroupAB_200`
checkpoint is `third_party/LeafMachine2/checkpoints/best.pt`.

The classifier implementation is under `leafmachine_classifier`. Its stable
specimen-grouped manifest contains 698 training, 148 validation, and 152
held-out test images with no GBIF specimen crossing partitions.

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

## Legacy approach

The previous YOLO26 eight-combination whole-sheet classifier, its generated
datasets, checkpoints, results, and scripts are isolated under
`legacy/yolo26_classification`. See its README for reproduction commands.

LeafMachine2 is GPL-3.0 licensed.
