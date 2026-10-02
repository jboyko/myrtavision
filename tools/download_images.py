"""Download every image in the split manifest to data/images at native resolution.

Image keys and URLs come from scores.csv (gbifID plus an MD5 prefix of the URL,
as in leafmachine_classifier/prepare.py). Existing files are skipped, payloads
are verified before they replace anything, and failures are written to
data/download_failures.csv so they can be retried or removed deliberately.
"""
import argparse
import csv
import hashlib
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "myrtavision/1.0"}
Image.MAX_IMAGE_PIXELS = None  # sheets run to 11k px


def image_key(row):
    return f"{row['gbifID']}_{hashlib.md5(row['url'].encode()).hexdigest()[:8]}"


def fetch(key, url, images, retries, timeout):
    dst = images / f"{key}.jpg"
    if dst.exists():
        return key, url, None
    tmp = dst.with_suffix(".part")
    error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                tmp.write_bytes(response.read())
            with Image.open(tmp) as image:
                image.verify()  # reject truncated or HTML-error payloads
            tmp.replace(dst)
            return key, url, None
        except Exception as exc:  # network, HTTP, and decode errors are all retried
            error = f"{type(exc).__name__}: {exc}"
            tmp.unlink(missing_ok=True)
            time.sleep(2 ** attempt)
    return key, url, error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=PROJECT_ROOT / "scores.csv")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "splits/phenology_v2.csv")
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "data/images")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, help="first N manifest rows only (smoke tests)")
    args = parser.parse_args()

    wanted = [row["image_id"] for row in csv.DictReader(args.manifest.open())]
    if args.limit:
        wanted = wanted[: args.limit]
    urls = {image_key(row): row["url"] for row in csv.DictReader(args.scores.open())}
    missing_urls = [key for key in wanted if key not in urls]
    if missing_urls:
        raise SystemExit(f"{len(missing_urls)} manifest images have no URL in {args.scores}, e.g. {missing_urls[0]}")

    args.images.mkdir(parents=True, exist_ok=True)
    before = sum((args.images / f"{key}.jpg").exists() for key in wanted)
    print(f"{before}/{len(wanted)} already downloaded", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(
            lambda key: fetch(key, urls[key], args.images, args.retries, args.timeout), wanted
        ))

    failures = [result for result in results if result[2]]
    failure_log = args.images.parent / "download_failures.csv"
    with failure_log.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "url", "error"])
        writer.writerows(failures)
    print(f"{len(wanted) - len(failures)}/{len(wanted)} images present; {len(failures)} failed -> {failure_log}")
    for key, url, error in failures[:20]:
        print(f"  fail {key} {url}: {error}")


if __name__ == "__main__":
    main()
