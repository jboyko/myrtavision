"""Train a DINOv2 or DINOv3 phenology classifier and export per-organ probabilities.

Three ways to adapt the backbone, chosen with --mode:

    frozen  backbone fixed; only the attention-pooling head trains
    lora    low-rank adapters on every attention projection, plus the head
    last    the last --unfreeze blocks train, with layer-wise learning-rate decay

Sheets keep their aspect ratio: the long side is resized to --image-size and
both sides are rounded to whole patches (14 px for DINOv2, 16 for DINOv3).
Images come from the derived cache built by tools/build_datasets.py when it
exists.

The checkpoint with the best val macro average precision is kept. Its val and
test probabilities go to results/predictions/<name>/{val,test}.csv in the
format every method shares, and the checkpoint (trainable weights only) and
history to results/train/<name>/.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from dino_classifier.model import MEAN, ORGANS, STD, DinoPhenology, patch_mask_from_pixels  # noqa: E402
from leafmachine_classifier.metrics import average_precision  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def resize_to_patches(image: Image.Image, long_side: float, patch: int) -> torch.Tensor:
    """Resize so the long side is about long_side, both sides whole patches; float in [0, 1]."""
    scale = long_side / max(image.size)
    width, height = (max(patch, round(side * scale / patch) * patch) for side in image.size)
    return TF.to_dtype(TF.pil_to_tensor(image.resize((width, height), Image.Resampling.BILINEAR)), torch.float32, scale=True)


class Sheets(Dataset):
    def __init__(self, rows, image_size, patch, cache, augment):
        self.rows, self.image_size, self.patch, self.cache, self.augment = rows, image_size, patch, cache, augment
        self.jitter = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        cached = self.cache / f"{row['image_id']}.jpg"
        path = cached if cached.is_file() else PROJECT_ROOT / row["path"]
        with Image.open(path) as source:
            source.draft("RGB", (self.image_size, self.image_size))
            image = ImageOps.exif_transpose(source).convert("RGB")
        long_side = self.image_size * (random.uniform(0.85, 1.0) if self.augment else 1.0)
        image = resize_to_patches(image, long_side, self.patch)
        valid = torch.ones(1, *image.shape[1:])
        if self.augment:
            if random.random() < 0.5:
                image, valid = image.flip(-1), valid.flip(-1)
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, fill=0.5)
            valid = TF.rotate(valid, angle)
            image = self.jitter(image)
        image = TF.normalize(image, MEAN, STD)
        target = torch.tensor([float(row[organ]) for organ in ORGANS])
        return image, valid[0] > 0.5, target, row["image_id"]


def collate(batch):
    """Pad to the largest sheet in the batch; padding is masked out of pooling."""
    height = max(image.shape[1] for image, *_ in batch)
    width = max(image.shape[2] for image, *_ in batch)
    images = torch.zeros(len(batch), 3, height, width)
    valid = torch.zeros(len(batch), height, width, dtype=torch.bool)
    for i, (image, mask, _, _) in enumerate(batch):
        images[i, :, : image.shape[1], : image.shape[2]] = image
        valid[i, : mask.shape[0], : mask.shape[1]] = mask
    targets = torch.stack([target for _, _, target, _ in batch])
    return images, valid, targets, [image_id for *_, image_id in batch]


def autocast(device, amp):
    if amp == "none" or device.type != "cuda":
        return torch.autocast(device.type, enabled=False)
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    ids, targets, probabilities = [], [], []
    for images, valid, target, batch_ids in loader:
        with autocast(device, amp):
            logits, _ = model(images.to(device, non_blocking=True), patch_mask_from_pixels(valid, model.patch).to(device))
        probabilities.append(logits.float().sigmoid().cpu())
        targets.append(target)
        ids += batch_ids
    return ids, torch.cat(targets), torch.cat(probabilities)


def macro_ap(targets, probabilities):
    return float(np.nanmean([average_precision(targets[:, i], probabilities[:, i]) for i in range(len(ORGANS))]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--backbone", required=True, help="local folder of a DINOv2/v3 checkpoint, e.g. weights/dinov2-vitb14-reg")
    parser.add_argument("--mode", choices=("frozen", "lora", "last"), required=True)
    parser.add_argument("--unfreeze", type=int, default=4, help="blocks trained in --mode last")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=1536, help="long side, pixels")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=8, help="batches per optimizer step")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6, help="epochs without val macro AP gain before stopping")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=2e-4, help="LoRA rate; use ~2e-5 for --mode last")
    parser.add_argument("--layer-decay", type=float, default=0.75)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--amp", choices=("fp16", "bf16", "none"), default="fp16", help="V100s have no bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, help="first N images per split (smoke tests)")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "splits/phenology_v3.csv")
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "data/derived/1536")
    parser.add_argument("--runs", type=Path, default=PROJECT_ROOT / "runs/dino")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rows = list(csv.DictReader(args.manifest.open()))
    splits = {split: [row for row in rows if row["split"] == split][: args.limit] for split in ("train", "val", "test")}
    model = DinoPhenology(args.backbone, args.mode, lora_rank=args.lora_rank, unfreeze=args.unfreeze).to(device)
    loader = {
        split: DataLoader(
            Sheets(split_rows, args.image_size, model.patch, args.cache, augment=split == "train"),
            batch_size=args.batch_size, shuffle=split == "train", num_workers=args.workers,
            collate_fn=collate, pin_memory=device.type == "cuda", drop_last=split == "train",
            persistent_workers=args.workers > 0,
        )
        for split, split_rows in splits.items()
    }

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{args.name}: {args.mode}, {trainable / 1e6:.2f}M of {total / 1e6:.1f}M parameters train; "
          f"{len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test", flush=True)
    optimizer = torch.optim.AdamW(model.parameter_groups(args.head_lr, args.backbone_lr, args.layer_decay),
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=args.amp == "fp16" and device.type == "cuda")
    steps_per_epoch = max(1, len(loader["train"]) // args.accumulate)
    total_steps, warmup = steps_per_epoch * args.epochs, steps_per_epoch

    run_dir = args.runs / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    history, best, stale, step = [], -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, losses = time.time(), []
        optimizer.zero_grad(set_to_none=True)
        for index, (images, valid, targets, _) in enumerate(loader["train"], 1):
            with autocast(device, args.amp):
                logits, _ = model(images.to(device, non_blocking=True), patch_mask_from_pixels(valid, model.patch).to(device))
            loss = F.binary_cross_entropy_with_logits(logits.float(), targets.to(device))
            scaler.scale(loss / args.accumulate).backward()
            losses.append(loss.item())
            if index % args.accumulate == 0:
                step += 1
                factor = step / warmup if step <= warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * factor
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        _, val_targets, val_probabilities = predict(model, loader["val"], device, args.amp)
        val_ap = macro_ap(val_targets, val_probabilities)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_ap": val_ap,
                        "minutes": (time.time() - started) / 60})
        print(f"epoch {epoch}: loss {history[-1]['train_loss']:.4f}  val macro AP {val_ap:.4f}  "
              f"{history[-1]['minutes']:.1f} min", flush=True)
        if val_ap > best:
            best, stale = val_ap, 0
            torch.save({"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "epoch": epoch,
                        "state": model.trainable_state()}, run_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"no val gain for {args.patience} epochs; stopping")
                break
    with (run_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["state"], strict=False)
    print(f"best epoch {checkpoint['epoch']}: val macro AP {best:.4f}")
    out_dir = args.results / "predictions" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        ids, _, probabilities = predict(model, loader[split], device, args.amp)
        with (out_dir / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", *[f"{organ}_prob" for organ in ORGANS]])
            for image_id, row in zip(ids, probabilities.tolist()):
                writer.writerow([image_id, *[f"{value:.6f}" for value in row]])
        print(f"wrote {len(ids)} {split} predictions to {out_dir / f'{split}.csv'}")
    keep = args.results / "train" / args.name
    keep.mkdir(parents=True, exist_ok=True)
    for path in (run_dir / "best.pt", run_dir / "history.csv"):
        shutil.copy2(path, keep / path.name)


if __name__ == "__main__":
    main()
