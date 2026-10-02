"""Evaluate a trained LeafMachine classifier on validation or held-out test."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import PhenologyDataset
from .metrics import ORGANS, compute_metrics, format_metrics, optimize_thresholds
from .model import load_classifier_checkpoint, resolve_project_path
from .train import collect_predictions, resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--detector-weights", type=Path, help="override relocated LeafMachine base weights")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--thresholds", type=float, nargs=3)
    parser.add_argument("--optimize", action="store_true", help="optimize thresholds on this split (validation only)")
    parser.add_argument("--out", type=Path, help="optional per-image prediction CSV")
    args = parser.parse_args()
    if args.optimize and args.split != "val":
        raise ValueError("Refusing to optimize thresholds on the held-out test split")

    device = resolve_device(args.device)
    model, payload = load_classifier_checkpoint(args.checkpoint, args.detector_weights)
    config = payload["config"]
    manifest = args.manifest or resolve_project_path(config["manifest"])
    if not manifest.is_file():
        raise FileNotFoundError(f"Checkpoint manifest {manifest} not found; pass --manifest")
    dataset = PhenologyDataset(manifest, args.split, int(config["image_size"]), augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model.to(device)
    targets, probabilities, paths = collect_predictions(model, loader, device)
    thresholds = tuple(args.thresholds or payload.get("thresholds", [0.5, 0.5, 0.5]))
    if args.optimize:
        thresholds = optimize_thresholds(targets, probabilities)
        print("Thresholds optimized on these validation rows; metrics are optimistic.")
    metrics = compute_metrics(targets, probabilities, thresholds)
    print(format_metrics(metrics, thresholds))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fields = ["image", *[f"{organ}_truth" for organ in ORGANS], *[f"{organ}_prob" for organ in ORGANS]]
        with args.out.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            for path, truth, probability in zip(paths, targets.tolist(), probabilities.tolist()):
                writer.writerow([path, *map(int, truth), *[f"{value:.6f}" for value in probability]])
        print(f"wrote {len(paths)} predictions to {args.out}")


if __name__ == "__main__":
    main()
