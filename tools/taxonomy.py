"""Write genus and species for every manifest image from the raw GBIF export.

training_data/ stays on the Mac (it is not in git), so the result,
splits/taxonomy.csv, is committed for the cluster and the analysis scripts.
"""
import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(sys.maxsize)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--occurrence", type=Path, default=PROJECT_ROOT / "training_data/occurrence.txt")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "splits/phenology_v3.csv")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "splits/taxonomy.csv")
    args = parser.parse_args()

    with args.occurrence.open() as handle:
        occurrence = {
            row["gbifID"]: row
            for row in csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        }
    manifest = list(csv.DictReader(args.manifest.open()))
    missing = [row["gbif_id"] for row in manifest if row["gbif_id"] not in occurrence]
    if missing:
        raise SystemExit(f"{len(missing)} manifest specimens not in {args.occurrence}, e.g. {missing[0]}")

    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "gbif_id", "split", "genus", "species", "scientific_name"])
        for row in manifest:
            taxon = occurrence[row["gbif_id"]]
            writer.writerow([row["image_id"], row["gbif_id"], row["split"], taxon["genus"],
                             taxon["species"], taxon["scientificName"]])
    genera = {occurrence[row["gbif_id"]]["genus"] for row in manifest}
    no_species = sum(not occurrence[row["gbif_id"]]["species"] for row in manifest)
    print(f"wrote {len(manifest)} images, {len(genera)} genera ({no_species} without a species) to {args.out}")


if __name__ == "__main__":
    main()
