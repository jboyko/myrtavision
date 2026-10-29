"""List one image per unscored specimen in the GBIF export, for prediction.

Reads training_data/{occurrence,multimedia}.txt (on the Mac only; not in git)
and writes splits/pool.csv, which tools/download_images.py --manifest uses to
fetch the images to data/pool/ and dino_classifier.predict then scores.

Specimens already in scores.csv are left out. Many specimens have several
image links: the same sheet at another size or on a mirror, label close-ups,
extra sheets. As the scorers mostly did, each specimen gets its first usable
image, where unusable means not a single image (web pages, PHP viewers,
deep-zoom tile sets) or a known small rendition (IIIF widths under 1000 px).
An image URL already listed for another specimen is used once. Up to three
further usable images go in the alternates column (|-separated), which the
downloader tries in order if the first fails; image_id stays that of the first.
"""
import argparse
import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(sys.maxsize)
NOT_IMAGES = {"text/html", "dzi"}
IIIF_WIDTH = re.compile(r"/full/(\d+),/")


def usable(media):
    if media["type"] != "StillImage" or not media["identifier"].startswith("http"):
        return False
    url = media["identifier"].lower()
    if media["format"].lower() in NOT_IMAGES or url.endswith(".dzi") or ".php" in url:
        return False
    width = IIIF_WIDTH.search(media["identifier"])
    return not (width and int(width.group(1)) < 1000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--occurrence", type=Path, default=PROJECT_ROOT / "training_data/occurrence.txt")
    parser.add_argument("--multimedia", type=Path, default=PROJECT_ROOT / "training_data/multimedia.txt")
    parser.add_argument("--scores", type=Path, default=PROJECT_ROOT / "scores.csv")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "splits/pool.csv")
    args = parser.parse_args()

    read = lambda path: csv.DictReader(path.open(), delimiter="\t", quoting=csv.QUOTE_NONE)
    occurrence = {row["gbifID"]: row for row in read(args.occurrence)}
    scored = list(csv.DictReader(args.scores.open()))
    scored_specimens = {row["gbifID"] for row in scored}
    used_urls = {row["url"] for row in scored}

    chosen, skipped = {}, Counter()
    for media in read(args.multimedia):
        specimen = media["gbifID"]
        if specimen in scored_specimens:
            continue
        if not usable(media):
            skipped[media["format"] or "unusable"] += 1
            continue
        if media["identifier"] in used_urls:
            skipped["url used by another specimen"] += 1
            continue
        used_urls.add(media["identifier"])
        chosen.setdefault(specimen, []).append(media["identifier"])

    rows = []
    for specimen, (url, *alternates) in chosen.items():
        taxon = occurrence.get(specimen, {})
        image_id = f"{specimen}_{hashlib.md5(url.encode()).hexdigest()[:8]}"  # as leafmachine_classifier/prepare.py
        rows.append({
            "image_id": image_id, "gbif_id": specimen, "path": f"data/pool/{image_id}.jpg", "url": url,
            "host": url.split("/")[2], "institution": taxon.get("institutionCode", ""),
            "catalog_number": taxon.get("catalogNumber", ""), "genus": taxon.get("genus", ""),
            "species": taxon.get("species", ""), "country": taxon.get("countryCode", ""),
            "year": taxon.get("year", ""), "alternates": "|".join(alternates[:3]),
        })
    rows.sort(key=lambda row: row["image_id"])
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} specimens (one image each) to {args.out}")
    print("left out (image links, not specimens):", dict(skipped.most_common()))
    print("by host:", Counter(row["host"] for row in rows).most_common(12))


if __name__ == "__main__":
    main()
