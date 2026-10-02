"""Download scored images and build a YOLO classification dataset.

Downloads are archived at their native resolution under data/images and are
never modified in place. Any downscaling is a derived, cached artifact under
data/derived/<spec>, so the archive stays the single source of truth and a
resolution sweep costs a re-derive rather than a re-download.
"""
import argparse, csv, hashlib, random, shutil, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from labels import CLASSES, encode

Image.MAX_IMAGE_PIXELS = None  # sheets run to 11k px; the bomb guard trips well below

LEGACY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = LEGACY_ROOT.parents[1]
IMAGES = PROJECT_ROOT / "data" / "images"
DERIVED = LEGACY_ROOT / "data" / "derived"
DATASET = LEGACY_ROOT / "data" / "dataset"
UA = {"User-Agent": "myrtavision/1.0"}


def read_scores(path):
    rows = []
    for r in csv.DictReader(open(path)):
        flags = tuple(int(r[o]) for o in ("bud", "flower", "fruit"))
        if not any(flags) and r["none"] != "1":
            continue  # unscored
        # a gbifID can carry several images, so key on the URL too
        key = f'{r["gbifID"]}_{hashlib.md5(r["url"].encode()).hexdigest()[:8]}'
        rows.append((key, r["url"], encode(flags)))
    return rows


def fetch(row, refetch=False):
    """Download to data/images at native resolution. Never resizes."""
    key, url, _ = row
    dst = IMAGES / f"{key}.jpg"
    if dst.exists() and not refetch:
        return dst
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
        tmp = dst.with_suffix(".part")
        tmp.write_bytes(body)
        with Image.open(tmp) as im:
            im.verify()  # reject truncated/HTML-error payloads before committing
        tmp.replace(dst)
        return dst
    except Exception as e:
        print(f"fail {key}: {e}")
        for p in (dst.with_suffix(".part"),):
            p.unlink(missing_ok=True)
        return None


def spec_dir(maxpx, pad):
    return DERIVED / f"{maxpx}{'_pad' if pad else ''}"


def derive(key, maxpx, pad):
    """Cache a downscaled copy under data/derived/<spec>; return its path.

    With pad, also pad to square. Sheets are ~1.5:1 portrait, so ultralytics
    center-crops the top and bottom off; padding keeps the whole sheet but
    renders the plant 0.625x smaller. Measured worse -- off by default.
    """
    src = IMAGES / f"{key}.jpg"
    dst = spec_dir(maxpx, pad) / f"{key}.jpg"
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.draft("RGB", (maxpx, maxpx))
        im = im.convert("RGB")
        im.thumbnail((maxpx, maxpx), Image.LANCZOS)
        if pad:
            sq = Image.new("RGB", (maxpx, maxpx), (255, 255, 255))
            sq.paste(im, ((maxpx - im.width) // 2, (maxpx - im.height) // 2))
            im = sq
        im.save(dst, quality=95)
    return dst


def split(rows, val_frac, seed):
    by_class = {c: [] for c in CLASSES}
    for row in rows:
        by_class[row[2]].append(row)
    rng = random.Random(seed)
    train, val = [], []
    for items in by_class.values():
        rng.shuffle(items)
        n = min(len(items) - 1, max(1, round(len(items) * val_frac))) if len(items) > 1 else 0
        val += items[:n]
        train += items[n:]
    return train, val


def link(rows, split_name, source):
    for key, _, cls in rows:
        src = (source / f"{key}.jpg").resolve()
        if not src.exists():
            continue
        dst = DATASET / split_name / cls / f"{key}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.unlink(missing_ok=True)
        dst.symlink_to(src)


def report(rows, source):
    """Long-side distribution of what training will actually see."""
    sides = []
    for key, _, _ in rows:
        p = source / f"{key}.jpg"
        if p.exists():
            with Image.open(p) as im:
                sides.append(max(im.size))
    if not sides:
        return
    sides.sort()
    q = lambda f: sides[min(len(sides) - 1, int(f * len(sides)))]
    print(f"long side: min {sides[0]}  p50 {q(.5)}  p90 {q(.9)}  max {sides[-1]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default=str(PROJECT_ROOT / "scores.csv"))
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--maxpx", type=int, default=0,
                   help="downscale long side for the dataset (0 = native)")
    p.add_argument("--pad", action="store_true", help="pad to square (measured worse)")
    p.add_argument("--refetch", action="store_true",
                   help="re-download even if the file is already cached")
    a = p.parse_args()

    rows = read_scores(a.scores)
    IMAGES.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        got = sum(x is not None for x in ex.map(lambda r: fetch(r, a.refetch), rows))
    print(f"{got}/{len(rows)} images")

    if a.maxpx:
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(lambda r: derive(r[0], a.maxpx, a.pad), rows))
        source = spec_dir(a.maxpx, a.pad)
    else:
        source = IMAGES
    print(f"dataset source: {source}")
    report(rows, source)

    train, val = split(rows, a.val, a.seed)
    shutil.rmtree(DATASET, ignore_errors=True)  # stale symlinks from earlier specs
    link(train, "train", source)
    link(val, "val", source)
    print(f"train {len(train)}  val {len(val)}")
    for c in CLASSES:
        print(f"  {c:18} {sum(r[2] == c for r in train):4} {sum(r[2] == c for r in val):4}")


if __name__ == "__main__":
    main()
