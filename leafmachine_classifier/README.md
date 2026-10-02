# LeafMachine multiscale classifier

This package converts LeafMachine2's herbarium-native plant-component detector
into a weakly supervised, three-label classifier. It does not use or require
bounding-box annotations.

## Architecture

The downloaded `PLANT_GroupAB_200` checkpoint is an 86.5M-parameter YOLOv5x
model. Its Detect layer consumes three feature maps:

| source layer | stride | 1280px evidence map |
|---:|---:|---:|
| 17 | 8 | 160 × 160 |
| 20 | 16 | 80 × 80 |
| 23 | 32 | 40 × 40 |

The detector head is discarded. Each feature map receives a small spatial head
with three channels: bud, flower, and fruit. Padding-aware normalized
log-sum-exp pooling turns each evidence map into three sheet-level logits, and
learned per-organ scale weights combine the pyramid levels. The final outputs
are three independent sigmoids.

Complete sheets are fitted inside 1280 × 1280 with gray letterboxing. No part
of a sheet is cropped, and padded locations are excluded from pooling.

## Data split

The manifest shared by every method is `../splits/phenology_v2.csv`. Its 2,025 images are
grouped by GBIF specimen ID before splitting, preventing different images of
one specimen from leaking across partitions:

| split | images | specimens |
|---|---:|---:|
| train | 1,404 | 1,360 |
| validation | 320 | 293 |
| test | 301 | 293 |

Regenerate it only deliberately:

```bash
.venv/bin/python -m leafmachine_classifier.prepare --force   # writes splits/phenology_v2.csv
```

## Training

First train only the 288,661-parameter evidence head while keeping all
LeafMachine weights frozen:

```bash
.venv/bin/python -m leafmachine_classifier.train \
  --name frozen_1280
```

The best checkpoint is
`runs/leafmachine_classifier/frozen_1280/best.pt`. Frozen-backbone checkpoints
contain only the learned head (about 1.1 MB) and verify the SHA-256 digest of
the external LeafMachine base checkpoint when loaded.

If the frozen-head result warrants further work, initialize from it and
fine-tune only the final neck modules with a much lower learning rate:

```bash
.venv/bin/python -m leafmachine_classifier.train \
  --init-checkpoint runs/leafmachine_classifier/frozen_1280/best.pt \
  --unfreeze-from 20 \
  --learning-rate 1e-5 \
  --name neck20_1280
```

Do not begin by unfreezing the entire 86.5M-parameter network on 1,404 training
images.

## Evaluation and prediction

Validation selects per-organ thresholds; the test split remains held out:

```bash
.venv/bin/python -m leafmachine_classifier.evaluate \
  runs/leafmachine_classifier/frozen_1280/best.pt \
  --split test

.venv/bin/python -m leafmachine_classifier.predict \
  runs/leafmachine_classifier/frozen_1280/best.pt \
  data/images \
  --out leafmachine_predictions.csv
```

LeafMachine2 and the reused detector code are GPL-3.0 licensed.
