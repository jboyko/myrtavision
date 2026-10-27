"""Check dino_classifier.predict against the ensemble's training-time predictions.

    python3 tools/check_predict.py results/check_val.csv results/predictions/ens_dinov2_l_lora_x3/val.csv

predict.py reads native images and rebuilds the 1536 px thumbnail in memory;
training read the cached JPEG of that thumbnail. Probabilities should agree to
within JPEG re-encoding noise, and the 0/1 calls almost everywhere. Standard
library only, so it runs on a login node without the conda env.
"""
import csv
import sys

ORGANS = ("bud", "flower", "fruit")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    new = {row["image"]: row for row in csv.DictReader(open(sys.argv[1]))}
    old = {row["image"]: row for row in csv.DictReader(open(sys.argv[2]))}
    shared = sorted(set(new) & set(old))
    print(f"{len(shared)} sheets in both ({len(new)} predicted, {len(old)} reference)")
    for organ in ORGANS:
        diffs = [abs(float(new[i][f"{organ}_prob"]) - float(old[i][f"{organ}_prob"])) for i in shared]
        print(f"{organ:7} max |diff| {max(diffs):.4f}  mean {sum(diffs) / len(diffs):.5f}  "
              f"over 0.05: {sum(d > 0.05 for d in diffs)}")


if __name__ == "__main__":
    main()
