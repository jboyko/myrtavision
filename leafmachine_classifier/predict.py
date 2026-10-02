"""Predict bud, flower, and fruit probabilities for complete herbarium sheets."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from .data import ORGANS, letterbox
from .model import load_classifier_checkpoint
from .train import resolve_device


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
Image.MAX_IMAGE_PIXELS = None


class PredictionDataset(Dataset):
    def __init__(self, paths, image_size):
        self.paths = paths
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as source:
            source.draft("RGB", (self.image_size, self.image_size))
            image = ImageOps.exif_transpose(source).convert("RGB")
        image, valid_mask = letterbox(image, self.image_size)
        return {"image": image, "valid_mask": valid_mask, "path": str(path)}


def collect_files(sources):
    files = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            files.extend(item for item in sorted(path.rglob("*")) if item.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            files.append(path)
        else:
            raise FileNotFoundError(f"No supported image source: {path}")
    if not files:
        raise ValueError("No supported images found")
    return [path.absolute() for path in files]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--detector-weights", type=Path, help="override relocated LeafMachine base weights")
    parser.add_argument("source", nargs="+")
    parser.add_argument("--out", type=Path, default=Path("leafmachine_predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, payload = load_classifier_checkpoint(args.checkpoint, args.detector_weights)
    image_size = int(payload["config"]["image_size"])
    thresholds = payload.get("thresholds", [0.5, 0.5, 0.5])
    files = collect_files(args.source)
    loader = DataLoader(PredictionDataset(files, image_size), batch_size=args.batch_size, num_workers=args.workers)
    model.to(device).eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", *[f"{organ}_prob" for organ in ORGANS], *[f"{organ}_present" for organ in ORGANS]])
        with torch.inference_mode():
            for batch in loader:
                logits = model(batch["image"].to(device), batch["valid_mask"].to(device))
                probabilities = logits.sigmoid().cpu()
                predictions = probabilities >= torch.tensor(thresholds).view(1, -1)
                for path, probability, prediction in zip(batch["path"], probabilities, predictions):
                    writer.writerow([
                        path,
                        *[f"{value:.6f}" for value in probability.tolist()],
                        *map(int, prediction.tolist()),
                    ])
    print(f"wrote {len(files)} predictions to {args.out}")


if __name__ == "__main__":
    main()
