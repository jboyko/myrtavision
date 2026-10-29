"""Check dino_classifier.predict against the ensemble's training-time predictions.

    python3 tools/check_predict.py results/check_val.csv results/predictions/ens_dinov2_l_lora_x3/val.csv

predict.py reads native images and rebuilds the 1536 px thumbnail in memory;
training read the cached JPEG of that thumbnail. Probabilities should agree to
GPU and fp16 noise (a few thousandths), and calls at the thresholds in
dino_classifier/final_model.json should almost never flip. Standard library
only, so it runs on a login node without the conda env.
"""
import csv
import json
import sys
from pathlib import Path

ORGANS = ("bud", "flower", "fruit")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    new = {row["image"]: row for row in csv.DictReader(open(sys.argv[1]))}
    old = {row["image"]: row for row in csv.DictReader(open(sys.argv[2]))}
    config = Path(__file__).resolve().parents[1] / "dino_classifier/final_model.json"
    thresholds = json.loads(config.read_text())["thresholds"]
    shared = sorted(set(new) & set(old))
    print(f"{len(shared)} sheets in both ({len(new)} predicted, {len(old)} reference)")
    for organ in ORGANS:
        a = [float(new[i][f"{organ}_prob"]) for i in shared]
        b = [float(old[i][f"{organ}_prob"]) for i in shared]
        diffs = [abs(x - y) for x, y in zip(a, b)]
        flips = sum((x >= thresholds[organ]) != (y >= thresholds[organ]) for x, y in zip(a, b))
        print(f"{organ:7} max |diff| {max(diffs):.4f}  mean {sum(diffs) / len(diffs):.5f}  "
              f"over 0.05: {sum(d > 0.05 for d in diffs)}  calls flipped: {flips}")


if __name__ == "__main__":
    main()
