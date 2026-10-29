"""Join sharded predict outputs with the pool's specimen metadata.

    python3 tools/merge_predictions.py results/pool/pool_*.csv --out results/pool_predictions.csv

Each shard row is matched to splits/pool.csv by image_id (the file stem), so
the output carries gbif_id, host, institution, taxon, country and year beside
the probabilities and calls. Pool sheets with no prediction (never downloaded,
or unreadable) are counted, not written. Standard library only, so it runs on
a login node without the conda env.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORGANS = ("bud", "flower", "fruit")
METADATA = ("gbif_id", "host", "institution", "catalog_number", "genus", "species", "country", "year", "url")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "splits/pool.csv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pool = {row["image_id"]: row for row in csv.DictReader(args.manifest.open())}
    predictions = {}
    for shard in args.shards:
        for row in csv.DictReader(shard.open()):
            predictions[row["image"]] = row
    unknown = set(predictions) - set(pool)
    if unknown:
        raise SystemExit(f"{len(unknown)} predicted images are not in {args.manifest}, e.g. {sorted(unknown)[:3]}")

    columns = ["image_id", *METADATA, *[f"{organ}_prob" for organ in ORGANS], *ORGANS]
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for image_id in sorted(predictions):
            row = {"image_id": image_id, **{key: pool[image_id][key] for key in METADATA}}
            row.update({key: predictions[image_id][key] for key in columns if key in predictions[image_id]})
            writer.writerow(row)

    print(f"wrote {len(predictions)} sheets to {args.out}; {len(pool) - len(predictions)} pool sheets not scored")
    for organ in ORGANS:
        present = sum(row[organ] == "1" for row in predictions.values())
        print(f"{organ:7} called present on {present} ({present / len(predictions):.1%})")
    missing = Counter(pool[i]["host"] for i in set(pool) - set(predictions))
    print("not scored, by host:", missing.most_common(8))


if __name__ == "__main__":
    main()
