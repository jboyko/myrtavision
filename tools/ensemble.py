"""Average member runs' per-organ probabilities into ensemble prediction files.

Each line of slurm/ensembles.tsv names an ensemble and its comma-separated
member runs. For every ensemble whose members all have predictions, this writes
results/predictions/<name>/{val,test}.csv, which tools/compare.py then scores
like any other run. Ensembles with missing members are skipped with a warning.
"""
import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORGANS = ("bud", "flower", "fruit")
SPLITS = ("val", "test")


def read(path):
    rows = list(csv.DictReader(path.open()))
    if f"{ORGANS[0]}_prob" not in rows[0]:
        raise ValueError(f"{path} has no <organ>_prob columns; only probability outputs can be averaged")
    return {Path(row["image"]).stem: [float(row[f"{organ}_prob"]) for organ in ORGANS] for row in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=PROJECT_ROOT / "slurm/ensembles.tsv")
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "results/predictions")
    args = parser.parse_args()

    for spec in csv.DictReader(args.spec.open(), delimiter="\t"):
        name, members = spec["name"], spec["members"].split(",")
        missing = [m for m in members for split in SPLITS if not (args.predictions / m / f"{split}.csv").is_file()]
        if missing:
            print(f"skip {name}: no predictions yet for {sorted(set(missing))}")
            continue
        out_dir = args.predictions / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            tables = [read(args.predictions / member / f"{split}.csv") for member in members]
            images = sorted(tables[0])
            if any(sorted(table) != images for table in tables[1:]):
                raise ValueError(f"{name}: members cover different {split} images")
            with (out_dir / f"{split}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image", *[f"{organ}_prob" for organ in ORGANS]])
                for image in images:
                    mean = [sum(table[image][i] for table in tables) / len(tables) for i in range(len(ORGANS))]
                    writer.writerow([image, *[f"{value:.6f}" for value in mean]])
        print(f"wrote {name} = mean of {len(members)} runs")


if __name__ == "__main__":
    main()
