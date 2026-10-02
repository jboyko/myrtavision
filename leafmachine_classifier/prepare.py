"""Create a specimen-grouped, combination-stratified split manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORGANS = ("bud", "flower", "fruit")
COMBINATIONS = tuple((bud, flower, fruit) for bud in (0, 1) for flower in (0, 1) for fruit in (0, 1))


def image_key(row: dict) -> str:
    digest = hashlib.md5(row["url"].encode()).hexdigest()[:8]
    return f"{row['gbifID']}_{digest}"


def allocate_bucket(groups, val_fraction, test_fraction, rng):
    groups = list(groups)
    if not groups:
        return {}
    rng.shuffle(groups)
    count = len(groups)
    if count == 1:
        return {groups[0]: "train"}
    validation = max(1, round(count * val_fraction))
    test = max(1, round(count * test_fraction)) if count >= 3 else 0
    while validation + test > count - 1:
        if test >= validation and test > 0:
            test -= 1
        else:
            validation -= 1
    result = {group: "test" for group in groups[:test]}
    result.update({group: "val" for group in groups[test : test + validation]})
    result.update({group: "train" for group in groups[test + validation :]})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=PROJECT_ROOT / "scores.csv")
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "data/images")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "splits/leafmachine_v1.csv")
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.val + args.test >= 1:
        raise ValueError("validation and test fractions must sum to less than one")
    if args.out.exists() and not args.force:
        raise FileExistsError(f"Refusing to replace stable split without --force: {args.out}")

    rows = []
    missing = []
    with args.scores.open() as handle:
        for row in csv.DictReader(handle):
            flags = tuple(int(row[organ]) for organ in ORGANS)
            if not any(flags) and row["none"] != "1":
                continue
            key = image_key(row)
            image = args.images / f"{key}.jpg"
            if not image.is_file():
                missing.append(key)
                continue
            rows.append({
                "image_id": key,
                "gbif_id": row["gbifID"],
                "path": str(image.relative_to(PROJECT_ROOT)),
                **{organ: value for organ, value in zip(ORGANS, flags)},
            })

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["gbif_id"]].append(row)

    # Bucket each specimen by its most frequent image-level combination. A few
    # specimens have inconsistent image annotations, but every image from one
    # specimen still remains in exactly one split.
    buckets = defaultdict(list)
    for gbif_id, group_rows in grouped.items():
        counts = Counter(tuple(int(row[organ]) for organ in ORGANS) for row in group_rows)
        signature = max(counts, key=lambda flags: (counts[flags], flags))
        buckets[signature].append(gbif_id)

    rng = random.Random(args.seed)
    assignments = {}
    for signature in COMBINATIONS:
        assignments.update(
            allocate_bucket(buckets.get(signature, []), args.val, args.test, rng)
        )
    for row in rows:
        row["split"] = assignments[row["gbif_id"]]

    split_groups = defaultdict(set)
    for row in rows:
        split_groups[row["split"]].add(row["gbif_id"])
    assert not (split_groups["train"] & split_groups["val"])
    assert not (split_groups["train"] & split_groups["test"])
    assert not (split_groups["val"] & split_groups["test"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "gbif_id", "path", "split", *ORGANS]
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["split"], row["gbif_id"], row["image_id"])))

    print(f"wrote {len(rows)} images from {len(grouped)} specimens to {args.out}")
    if missing:
        print(f"skipped {len(missing)} missing images")
    print(f"{'split':7} {'images':>7} {'groups':>7} {'bud':>6} {'flower':>7} {'fruit':>6}")
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row["split"] == split]
        totals = [sum(int(row[organ]) for row in selected) for organ in ORGANS]
        print(
            f"{split:7} {len(selected):7} {len(split_groups[split]):7} "
            f"{totals[0]:6} {totals[1]:7} {totals[2]:6}"
        )


if __name__ == "__main__":
    main()
