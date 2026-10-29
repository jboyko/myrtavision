"""Download every image in a manifest at native resolution.

By default the manifest is the split and images go to data/images, with URLs
looked up in scores.csv by image key (gbifID plus an MD5 prefix of the URL,
as in leafmachine_classifier/prepare.py). A manifest with its own url column,
such as splits/pool.csv from tools/build_pool.py, supplies its URLs directly:

    python tools/download_images.py --manifest splits/pool.csv --images data/pool

If the manifest has an alternates column (|-separated URLs), they are tried in
order when the main URL fails. Images are fetched in a shuffled order so the
threads spread across hosts. A host that fails 25 times in a row is skipped
for the rest of the run (its images are logged as failures); rerunning tries
only images still missing. Existing files are skipped, payloads
are verified before they replace anything, and failures are written to
<images>_download_failures.csv next to the images folder (for data/images,
data/download_failures.csv) so they can be retried or removed deliberately.

Run on a login node: the compute-node proxy blocks some image hosts. Exits
non-zero if any image is still missing.
"""
import argparse
import csv
import hashlib
import random
import sys
import threading
import time
from collections import Counter
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


class HostGuard:
    """Skip hosts that keep failing, so one dead server cannot stall the run."""

    def __init__(self, limit=25):
        self.limit, self.streak, self.lock = limit, Counter(), threading.Lock()

    def fetch(self, key, url, images, retries, timeout):
        host = url.split("/")[2]
        with self.lock:
            if self.streak[host] >= self.limit:
                return key, url, f"skipped: {host} failed {self.limit} times in a row"
        result = fetch(key, url, images, retries, timeout)
        with self.lock:
            self.streak[host] = self.streak[host] + 1 if result[2] else 0
            if result[2] and self.streak[host] == self.limit:
                print(f"  {host}: {self.limit} failures in a row; skipping its remaining images", flush=True)
        return result


def run(pool, task, items, label):
    results = []
    for done, result in enumerate(pool.map(task, items), 1):
        results.append(result)
        if done % 1000 == 0 or done == len(items):
            print(f"  {label}: {done}/{len(items)} done, {sum(1 for r in results if r[2])} failed", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=PROJECT_ROOT / "scores.csv")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "splits/phenology_v3.csv")
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "data/images")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, help="first N manifest rows only (smoke tests)")
    args = parser.parse_args()

    manifest = list(csv.DictReader(args.manifest.open()))
    wanted = [row["image_id"] for row in manifest]
    if args.limit:
        wanted = wanted[: args.limit]
    if "url" in manifest[0]:
        urls = {row["image_id"]: row["url"] for row in manifest}
        alternates = {row["image_id"]: [u for u in row.get("alternates", "").split("|") if u] for row in manifest}
    else:
        urls = {image_key(row): row["url"] for row in csv.DictReader(args.scores.open())}
        alternates = {}
    missing_urls = [key for key in wanted if key not in urls]
    if missing_urls:
        raise SystemExit(f"{len(missing_urls)} manifest images have no URL in {args.scores}, e.g. {missing_urls[0]}")

    args.images.mkdir(parents=True, exist_ok=True)
    before = sum((args.images / f"{key}.jpg").exists() for key in wanted)
    print(f"{before}/{len(wanted)} already downloaded", flush=True)
    random.Random(0).shuffle(wanted)  # neighbouring rows often share a host
    guard = HostGuard()
    with ThreadPoolExecutor(args.workers) as pool:
        results = run(pool, lambda key: guard.fetch(key, urls[key], args.images, args.retries, args.timeout),
                      wanted, "download")
    # Second pass: failed images with backup links try them, one at a time per image.
    fallback = [result for result in results if result[2] and alternates.get(result[0])]
    if fallback:
        print(f"{len(fallback)} failed images have backup links; trying them", flush=True)

        def retry(result):
            for url in alternates[result[0]]:
                outcome = guard.fetch(result[0], url, args.images, args.retries, args.timeout)
                if not outcome[2]:
                    return outcome
            return result

        with ThreadPoolExecutor(args.workers) as pool:
            recovered = {result[0]: result for result in run(pool, retry, fallback, "backup links")}
        results = [recovered.get(result[0], result) for result in results]
        print(f"{sum(not r[2] for r in recovered.values())} recovered from backup links", flush=True)

    failures = [result for result in results if result[2]]
    name = "download_failures.csv" if args.images.name == "images" else f"{args.images.name}_download_failures.csv"
    failure_log = args.images.parent / name
    with failure_log.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "url", "error"])
        writer.writerows(failures)
    print(f"{len(wanted) - len(failures)}/{len(wanted)} images present; {len(failures)} failed -> {failure_log}")
    by_host = Counter(url.split("/")[2] for _, url, _ in failures)
    if by_host:
        print("failures by host:", ", ".join(f"{host} {count}" for host, count in by_host.most_common(10)))
    for key, url, error in failures[:20]:
        print(f"  fail {key} {url}: {error}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
