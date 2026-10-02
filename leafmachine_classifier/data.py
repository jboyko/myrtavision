"""Manifest-backed herbarium dataset with full-sheet letterboxing."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import pil_to_tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORGANS = ("bud", "flower", "fruit")
Image.MAX_IMAGE_PIXELS = None


def letterbox(image: Image.Image, size: int, fill: int = 114):
    """Fit a complete image inside a square without cropping."""
    width, height = image.size
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    canvas.paste(image, (left, top))

    valid_mask = torch.zeros((1, size, size), dtype=torch.bool)
    valid_mask[:, top : top + resized_height, left : left + resized_width] = True
    return pil_to_tensor(canvas).float().div_(255.0), valid_mask


class PhenologyDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        split: str,
        image_size: int = 1280,
        augment: bool = False,
        limit: int | None = None,
    ):
        self.manifest = Path(manifest)
        self.image_size = image_size
        self.augment = augment
        with self.manifest.open() as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if limit is not None:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"No {split!r} rows in {self.manifest}")
        self.color_jitter = ColorJitter(brightness=0.08, contrast=0.08, saturation=0.04, hue=0.01)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        with Image.open(path) as source:
            source.draft("RGB", (self.image_size, self.image_size))
            image = ImageOps.exif_transpose(source).convert("RGB")
        image, valid_mask = letterbox(image, self.image_size)
        if self.augment:
            if random.random() < 0.5:
                image = image.flip(-1)
                valid_mask = valid_mask.flip(-1)
            image = self.color_jitter(image)
        target = torch.tensor([float(row[organ]) for organ in ORGANS], dtype=torch.float32)
        return {
            "image": image,
            "valid_mask": valid_mask,
            "target": target,
            "path": str(path),
            "gbif_id": row["gbif_id"],
        }

    def positive_counts(self) -> torch.Tensor:
        return torch.tensor(
            [sum(int(row[organ]) for row in self.rows) for organ in ORGANS],
            dtype=torch.float32,
        )
