"""Score new herbarium sheets with the final ensemble.

    python -m dino_classifier.predict data/new_sheets --out results/new_sheets.csv

Inputs are image files or folders, searched recursively for .jpg, .jpeg, .png
and .tif. Each sheet is reduced exactly as in training: the 1536 px LANCZOS
thumbnail tools/build_datasets.py caches, including its JPEG encoding at
quality 95 (skipping that step moved some probabilities by 0.3), then the
model's own long side. Every member
listed in dino_classifier/final_model.json scores every sheet; probabilities
are averaged and called present at the val-chosen thresholds stored there.

Sheets are scored one at a time. Batching pads sheets to a common size, and
although padding is masked out of the pooling head, every backbone token
attends to it, so a sheet's probabilities depend on its batch-mates. The final members
were trained and evaluated at batch size 1, so this also reproduces their
val and test numbers.

The output has one row per sheet: image (file stem), path, <organ>_prob, and
<organ> (0/1 call). Sheets that cannot be read are listed in
<out>.failures.csv and left out.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from dino_classifier.model import MEAN, ORGANS, STD, DinoPhenology, patch_mask_from_pixels  # noqa: E402
from dino_classifier.train import autocast, collate, resize_to_patches  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
CACHE_SIDE = 1536  # tools/build_datasets.py --maxpx, which every training image passed through
SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class NewSheets(Dataset):
    def __init__(self, paths, image_size, patch):
        self.paths, self.image_size, self.patch = paths, image_size, patch

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            with Image.open(path) as source:
                source.draft("RGB", (CACHE_SIDE, CACHE_SIDE))
                image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((CACHE_SIDE, CACHE_SIDE), Image.Resampling.LANCZOS)
            encoded = io.BytesIO()
            image.save(encoded, "JPEG", quality=95)  # as tools/build_datasets.derive saves the cache
            with Image.open(encoded) as cached:
                cached.draft("RGB", (self.image_size, self.image_size))  # as train.Sheets reads it
                image = cached.convert("RGB")
            image = TF.normalize(resize_to_patches(image, self.image_size, self.patch), MEAN, STD)
        except Exception as error:  # unreadable or truncated files are reported, not fatal
            return None, str(path), f"{type(error).__name__}: {error}"
        valid = torch.ones(image.shape[1:], dtype=torch.bool)
        return (image, valid, torch.zeros(len(ORGANS)), str(path)), str(path), None


def collate_readable(batch):
    failures = [(path, error) for _, path, error in batch if error]
    readable = [item for item, _, error in batch if not error]
    return (collate(readable) if readable else None), failures


def find_images(inputs):
    paths = []
    for item in map(Path, inputs):
        if item.is_dir():
            paths += sorted(p for p in item.rglob("*") if p.suffix.lower() in SUFFIXES)
        elif item.is_file():
            paths.append(item)
        else:
            raise SystemExit(f"not a file or folder: {item}")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="image files or folders")
    parser.add_argument("--out", type=Path, required=True, help="output CSV")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "dino_classifier/final_model.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", choices=("fp16", "bf16", "none"), default="fp16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    checkpoints = [torch.load(PROJECT_ROOT / member, map_location="cpu") for member in config["members"]]
    shared = ("backbone", "mode", "lora_rank", "unfreeze", "image_size")
    settings = {key: checkpoints[0]["args"][key] for key in shared}
    for member, checkpoint in zip(config["members"], checkpoints):
        if any(checkpoint["args"][key] != settings[key] for key in shared):
            raise SystemExit(f"{member} differs from the first member in {shared}; members must share a backbone")
    backbone = Path(settings["backbone"])
    backbone = backbone if backbone.is_absolute() else PROJECT_ROOT / backbone
    device = torch.device(args.device)
    model = DinoPhenology(str(backbone), settings["mode"], lora_rank=settings["lora_rank"],
                          unfreeze=settings["unfreeze"]).to(device).eval()

    paths = find_images(args.inputs)
    if not paths:
        raise SystemExit("no images found")
    loader = DataLoader(NewSheets(paths, settings["image_size"], model.patch), batch_size=1,
                        num_workers=args.workers, collate_fn=collate_readable)
    print(f"{config['name']}: {len(config['members'])} members, {len(paths)} images", flush=True)

    totals, failures = {}, {}
    for number, checkpoint in enumerate(checkpoints, 1):
        model.load_state_dict(checkpoint["state"], strict=False)
        with torch.no_grad():
            for batch, batch_failures in loader:
                failures.update(batch_failures)
                if batch is None:
                    continue
                images, valid, _, batch_paths = batch
                with autocast(device, args.amp):
                    logits, _ = model(images.to(device, non_blocking=True), patch_mask_from_pixels(valid, model.patch).to(device))
                for path, probabilities in zip(batch_paths, logits.float().sigmoid().cpu()):
                    totals[path] = totals.get(path, 0) + probabilities
        print(f"member {number}/{len(checkpoints)} done", flush=True)

    thresholds = [config["thresholds"][organ] for organ in ORGANS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "path", *[f"{organ}_prob" for organ in ORGANS], *ORGANS])
        for path in map(str, paths):
            if path not in totals:
                continue
            mean = (totals[path] / len(checkpoints)).tolist()
            writer.writerow([Path(path).stem, path, *[f"{value:.6f}" for value in mean],
                             *[int(value >= threshold) for value, threshold in zip(mean, thresholds)]])
    print(f"wrote {len(totals)} sheets to {args.out}")
    if failures:
        failure_log = args.out.with_suffix(".failures.csv")
        with failure_log.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "error"])
            writer.writerows(sorted(failures.items()))
        print(f"{len(failures)} unreadable images listed in {failure_log}")


if __name__ == "__main__":
    main()
