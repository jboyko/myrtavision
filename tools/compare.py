"""Score every method's predictions with one set of metrics.

Each run writes results/predictions/<name>/{val,test}.csv with an image column
(path or image_id) and per-organ scores (<organ>_prob, or <organ>_max_conf for
the zero-shot detector). Truth comes from the split manifest. Per-organ
thresholds are chosen on val only; val metrics at those thresholds are
therefore optimistic. Average precision is threshold-free.

    python tools/compare.py            # val only; use this to choose methods
    python tools/compare.py --test     # held-out test at val thresholds; run once
"""
import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from leafmachine_classifier.metrics import ORGANS, compute_metrics, optimize_thresholds  # noqa: E402


def load(path, truth):
    rows = list(csv.DictReader(path.open()))
    column = "prob" if f"{ORGANS[0]}_prob" in rows[0] else "max_conf"
    ids = [Path(row["image"]).stem for row in rows]
    unknown = [image_id for image_id in ids if image_id not in truth]
    if unknown:
        raise ValueError(f"{path}: {len(unknown)} images not in the manifest, e.g. {unknown[0]}")
    targets = torch.tensor([truth[image_id] for image_id in ids], dtype=torch.float32)
    scores = torch.tensor([[float(row[f"{organ}_{column}"]) for organ in ORGANS] for row in rows])
    return targets, scores


def average_precision(truth, scores):
    order = scores.argsort(descending=True)
    truth = truth[order].bool()
    if not truth.any():
        return float("nan")
    hits = truth.float().cumsum(0)
    precision = hits / torch.arange(1, len(truth) + 1)
    return float(precision[truth].mean())


def summarize(name, split, targets, scores, thresholds):
    metrics = compute_metrics(targets, scores, thresholds)
    row = {"name": name, "split": split, "n": metrics["n"]}
    for index, organ in enumerate(ORGANS):
        row[f"{organ}_f1"] = metrics[organ]["f1"]
        row[f"{organ}_ap"] = average_precision(targets[:, index], scores[:, index])
        row[f"{organ}_thresh"] = thresholds[index]
    row["macro_f1"] = metrics["macro_f1"]
    row["macro_ap"] = float(torch.tensor([row[f"{organ}_ap"] for organ in ORGANS]).nanmean())
    row["exact"] = metrics["exact_match"]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "splits/phenology_v2.csv")
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "results/predictions")
    parser.add_argument("--test", action="store_true", help="also score the held-out test split")
    parser.add_argument("--out", type=Path, help="defaults to results/compare_{val,test}.csv")
    args = parser.parse_args()

    manifest = list(csv.DictReader(args.manifest.open()))
    truth = {row["image_id"]: [int(row[organ]) for organ in ORGANS] for row in manifest}
    expected = {split: sum(row["split"] == split for row in manifest) for split in ("val", "test")}

    rows = []
    for run in sorted(path for path in args.predictions.iterdir() if (path / "val.csv").is_file()):
        val_targets, val_scores = load(run / "val.csv", truth)
        thresholds = optimize_thresholds(val_targets, val_scores)
        rows.append(summarize(run.name, "val", val_targets, val_scores, thresholds))
        if args.test:
            if not (run / "test.csv").is_file():
                print(f"warning: {run.name} has no test.csv")
                continue
            test_targets, test_scores = load(run / "test.csv", truth)
            rows.append(summarize(run.name, "test", test_targets, test_scores, thresholds))
    if not rows:
        raise SystemExit(f"No runs with val.csv under {args.predictions}")

    out = args.out or args.predictions.parent / f"compare_{'test' if args.test else 'val'}.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'name':24} {'split':5} {'n':>4} {'bud F1':>7} {'flwr F1':>7} {'fruit F1':>8} {'macroF1':>7} {'macroAP':>7} {'exact':>6}")
    for row in sorted(rows, key=lambda row: (row["split"], -row["macro_f1"])):
        short = "" if row["n"] == expected[row["split"]] else f"  (of {expected[row['split']]})"
        print(
            f"{row['name']:24} {row['split']:5} {row['n']:4} {row['bud_f1']:7.3f} {row['flower_f1']:7.3f} "
            f"{row['fruit_f1']:8.3f} {row['macro_f1']:7.3f} {row['macro_ap']:7.3f} {row['exact']:6.3f}{short}"
        )
    print(f"wrote {out}")

    # Runs named <config>_s<N> are seed repeats of <config>; summarize each group.
    groups = {}
    for row in rows:
        if row["split"] == "val":
            groups.setdefault(re.sub(r"_s\d+$", "", row["name"]), []).append(row)
    repeated = {config: members for config, members in groups.items() if len(members) > 1}
    if repeated:
        print(f"\nval seed repeats: mean (min-max)")
        for config, members in sorted(repeated.items(), key=lambda item: -statistics.mean(r["macro_ap"] for r in item[1])):
            parts = []
            for metric in ("macro_f1", "macro_ap", "exact"):
                values = [member[metric] for member in members]
                parts.append(f"{metric} {statistics.mean(values):.3f} ({min(values):.3f}-{max(values):.3f})")
            print(f"{config:24} n_seeds={len(members)}  " + "  ".join(parts))
    if not args.test:
        print("val F1 uses thresholds chosen on val (optimistic); compare methods on macro AP too")


if __name__ == "__main__":
    main()
