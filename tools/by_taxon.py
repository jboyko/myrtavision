"""Break one run's errors down by genus, and export per-sheet errors for modelling.

Thresholds are chosen on val, as in tools/compare.py. Writes, under
results/taxa/:

    <run>_<splits>_sheets.csv   one row per sheet and organ: taxon, truth,
                                probability, call, error, log loss; input to
                                analysis/taxon_effects.R
    <run>_<splits>_genus.csv    per genus and organ: counts, F1, AP, error rate

Val and test are both held out from training, so after the final model is
chosen they can be pooled (--splits val,test) for more sheets per taxon.
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from leafmachine_classifier.metrics import ORGANS, average_precision, optimize_thresholds  # noqa: E402


def load(path, truth):
    rows = list(csv.DictReader(path.open()))
    ids = [Path(row["image"]).stem for row in rows]
    targets = torch.tensor([truth[image_id] for image_id in ids], dtype=torch.float32)
    scores = torch.tensor([[float(row[f"{organ}_prob"]) for organ in ORGANS] for row in rows])
    return ids, targets, scores


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="name under results/predictions, e.g. ens_dino_x6")
    parser.add_argument("--splits", default="val", help="comma-separated: val, or val,test once the model is final")
    parser.add_argument("--min-sheets", type=int, default=20, help="genera with fewer sheets print as 'other'")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "splits/phenology_v3.csv")
    parser.add_argument("--taxonomy", type=Path, default=PROJECT_ROOT / "splits/taxonomy.csv")
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "results/predictions")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results/taxa")
    args = parser.parse_args()

    truth = {row["image_id"]: [int(row[organ]) for organ in ORGANS] for row in csv.DictReader(args.manifest.open())}
    taxa = {row["image_id"]: row for row in csv.DictReader(args.taxonomy.open())}
    run = args.predictions / args.run
    _, val_targets, val_scores = load(run / "val.csv", truth)
    thresholds = optimize_thresholds(val_targets, val_scores)

    sheets = []
    for split in args.splits.split(","):
        ids, targets, scores = load(run / f"{split}.csv", truth)
        for image_id, target, score in zip(ids, targets.tolist(), scores.tolist()):
            taxon = taxa[image_id]
            for index, organ in enumerate(ORGANS):
                p = min(max(score[index], 1e-6), 1 - 1e-6)
                call = int(score[index] >= thresholds[index])
                sheets.append({
                    "image_id": image_id, "gbif_id": taxon["gbif_id"], "split": split,
                    "genus": taxon["genus"], "species": taxon["species"] or f"{taxon['genus']} sp.",
                    "organ": organ, "truth": int(target[index]), "prob": f"{score[index]:.6f}",
                    "call": call, "error": int(call != target[index]),
                    "log_loss": f"{-math.log(p if target[index] else 1 - p):.6f}",
                })

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.run}_{args.splits.replace(',', '+')}"
    with (args.out / f"{stem}_sheets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sheets[0]))
        writer.writeheader()
        writer.writerows(sheets)

    groups = defaultdict(list)
    for row in sheets:
        groups[(row["genus"], row["organ"])].append(row)
    genus_rows = []
    for (genus, organ), rows in sorted(groups.items()):
        tp = sum(r["truth"] and r["call"] for r in rows)
        fp = sum(not r["truth"] and r["call"] for r in rows)
        fn = sum(r["truth"] and not r["call"] for r in rows)
        truths = torch.tensor([r["truth"] for r in rows], dtype=torch.float32)
        probs = torch.tensor([float(r["prob"]) for r in rows])
        genus_rows.append({
            "genus": genus, "organ": organ, "sheets": len(rows), "positives": int(truths.sum()),
            "tp": tp, "fp": fp, "fn": fn, "errors": fp + fn, "error_rate": (fp + fn) / len(rows),
            "f1": f1(tp, fp, fn) if tp + fn else float("nan"), "ap": average_precision(truths, probs),
        })
    with (args.out / f"{stem}_genus.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(genus_rows[0]))
        writer.writeheader()
        writer.writerows(genus_rows)

    # Printed summary: genera with enough sheets, the rest pooled as "other".
    sheet_count = defaultdict(int)
    for row in sheets:
        if row["organ"] == ORGANS[0]:
            sheet_count[row["genus"]] += 1
    print(f"{args.run} on {args.splits}: {sum(sheet_count.values())} sheets, "
          f"thresholds {', '.join(f'{o} {t:.2f}' for o, t in zip(ORGANS, thresholds))}")
    print(f"{'genus':16} {'sheets':>6}  " + "  ".join(f"{o + ' F1':>9} {'err':>5}" for o in ORGANS))
    big = sorted((g for g, n in sheet_count.items() if n >= args.min_sheets), key=lambda g: -sheet_count[g])
    labels = [(g, {g}) for g in big] + [("other", set(sheet_count) - set(big))]
    for label, members in labels:
        if not members:
            continue
        parts = []
        for organ in ORGANS:
            rows = [r for r in sheets if r["genus"] in members and r["organ"] == organ]
            tp = sum(r["truth"] and r["call"] for r in rows)
            fp = sum(not r["truth"] and r["call"] for r in rows)
            fn = sum(r["truth"] and not r["call"] for r in rows)
            parts.append(f"{f1(tp, fp, fn) if tp + fn else float('nan'):9.3f} {fp + fn:5d}")
        print(f"{label:16} {sum(sheet_count[g] for g in members):6}  " + "  ".join(parts))
    print(f"wrote {args.out / f'{stem}_sheets.csv'} and {args.out / f'{stem}_genus.csv'}")


if __name__ == "__main__":
    main()
