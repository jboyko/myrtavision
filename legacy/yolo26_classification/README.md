# Legacy YOLO26 combination classifier

This directory preserves the original whole-sheet YOLO26 classification
approach. It encodes the three independent phenology flags as eight mutually
exclusive combination classes and recovers per-organ probabilities by summing
softmax probabilities. It is retained for reproducibility and comparison; it
is not the current modeling direction.

Shared inputs remain at the project root:

- `../../scores.csv`: sheet-level bud, flower, and fruit flags
- `../../data/images`: native-resolution herbarium images

Legacy-generated datasets, downsample caches, model weights, and run outputs
live entirely inside this directory.

From the project root:

```bash
.venv/bin/pip install -r legacy/yolo26_classification/requirements.txt
.venv/bin/python legacy/yolo26_classification/prep.py
.venv/bin/python legacy/yolo26_classification/train.py
.venv/bin/python legacy/yolo26_classification/eval.py
.venv/bin/python legacy/yolo26_classification/predict.py \
  legacy/yolo26_classification/data/dataset/val --out legacy_predictions.csv
```

The historical best documented run is
`runs/classify/cls640/weights/best.pt`.
