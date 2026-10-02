"""Build ultralytics classification datasets from the split manifest.

Native sheets (up to ~11k px) are downscaled once into data/derived/<maxpx>,
so YOLO training does not decode full-resolution JPEGs every epoch. Datasets
are folders of symlinks into that cache:

    datasets/combo/{train,val,test}/<combination>/<image_id>.jpg
    datasets/binary_<organ>/{train,val,test}/{absent,present}/<image_id>.jpg

Every method uses the same manifest, so splits match across methods. Exits
non-zero if any manifest image is missing, so training never starts on a
partial set.
"""
import argparse
import csv
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORGANS = ("bud", "flower", "fruit")
SPLITS = ("train", "val", "test")
Image.MAX_IMAGE_PIXELS = None


def combination(row):
    """(bud, flower, fruit) flags -> legacy class name, e.g. bud_flower or none."""
    return "_".join(organ for organ in ORGANS if row[organ] == "1") or "none"


def derive(task):
    source, destination, maxpx = task
    if destination.exists():
        return
    with Image.open(source) as image:
        image.draft("RGB", (maxpx, maxpx))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((maxpx, maxpx), Image.Resampling.LANCZOS)
        tmp = destination.with_suffix(".part.jpg")
        image.save(tmp, quality=95)
        tmp.replace(destination)


def link_tree(rows, root, class_of, cache):
    shutil.rmtree(root, ignore_errors=True)  # drop stale links from earlier builds
    for row in rows:
        destination = root / row["split"] / class_of(row) / f"{row['image_id']}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to((cache / f"{row['image_id']}.jpg").resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "splits/phenology_v2.csv")
    parser.add_argument("--maxpx", type=int, default=1536, help="long side of the derived cache")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open()))
    missing = [row for row in rows if not (PROJECT_ROOT / row["path"]).is_file()]
    if missing:
        by_split = dict(Counter(row["split"] for row in missing))
        examples = ", ".join(row["image_id"] for row in missing[:5])
        raise SystemExit(
            f"{len(missing)}/{len(rows)} manifest images missing (by split: {by_split}), e.g. {examples}. "
            "Download them on the login node (tools/download_images.py) or list unrecoverable ones in "
            "splits/unavailable.csv and regenerate the split."
        )
    print(f"all {len(rows)} manifest images on disk")

    cache = PROJECT_ROOT / "data" / "derived" / str(args.maxpx)
    cache.mkdir(parents=True, exist_ok=True)
    tasks = [(PROJECT_ROOT / row["path"], cache / f"{row['image_id']}.jpg", args.maxpx) for row in rows]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(derive, tasks, chunksize=8))
    print(f"derived cache: {cache}")

    link_tree(rows, args.out / "combo", combination, cache)
    for organ in ORGANS:
        link_tree(
            rows,
            args.out / f"binary_{organ}",
            lambda row, organ=organ: "present" if row[organ] == "1" else "absent",
            cache,
        )

    print(f"{'split':6} {'images':>6}  combination counts")
    for split in SPLITS:
        counts = Counter(combination(row) for row in rows if row["split"] == split)
        print(f"{split:6} {sum(counts.values()):6}  " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
