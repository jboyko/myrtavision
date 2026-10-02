"""Evaluate sheet-level evidence exported by lm2_predict.py."""
import argparse
import csv
from pathlib import Path

from phenology import ORGANS, decode_combination


def metrics(truth, scores, threshold):
    predictions = [score >= threshold for score in scores]
    tp = sum(y and prediction for y, prediction in zip(truth, predictions))
    fp = sum(not y and prediction for y, prediction in zip(truth, predictions))
    fn = sum(y and not prediction for y, prediction in zip(truth, predictions))
    correct = sum(y == prediction for y, prediction in zip(truth, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return correct / len(truth), precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, help="CSV produced by lm2_predict.py")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="select per-organ F1 thresholds on these rows (optimistic; do not report as a test result)",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(args.predictions.open()))
    truth = {organ: [] for organ in ORGANS}
    scores = {organ: [] for organ in ORGANS}
    for row in rows:
        flags = decode_combination(Path(row["image"]).parent.name)
        for organ, flag in zip(ORGANS, flags):
            truth[organ].append(flag)
            scores[organ].append(float(row[f"{organ}_max_conf"]))

    thresholds = {organ: args.threshold for organ in ORGANS}
    if args.optimize:
        for organ in ORGANS:
            candidates = sorted({score for score in scores[organ] if score > 0} | {1.0})
            thresholds[organ] = max(
                candidates,
                key=lambda threshold: metrics(truth[organ], scores[organ], threshold)[3],
            )
        print("Thresholds optimized on evaluation rows; metrics are optimistic.")

    print(f"{'organ':8} {'thresh':>7} {'acc':>6} {'prec':>6} {'rec':>6} {'F1':>6}")
    for organ in ORGANS:
        values = metrics(truth[organ], scores[organ], thresholds[organ])
        print(f"{organ:8} {thresholds[organ]:7.3f} " + " ".join(f"{value:6.3f}" for value in values))

    exact = 0
    for index in range(len(rows)):
        expected = tuple(truth[organ][index] for organ in ORGANS)
        predicted = tuple(scores[organ][index] >= thresholds[organ] for organ in ORGANS)
        exact += expected == predicted
    print(f"\nexact match: {exact / len(rows):.3f}  n={len(rows)}")


if __name__ == "__main__":
    main()
