"""Train the weakly supervised LeafMachine multiscale classifier."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import PhenologyDataset
from .metrics import compute_metrics, format_metrics, optimize_thresholds
from .model import DEFAULT_DETECTOR_WEIGHTS, LeafMachineClassifier, load_classifier_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_predictions(model, loader, device):
    model.eval()
    all_targets, all_probabilities, all_paths = [], [], []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            logits = model(images, valid_mask)
            all_targets.append(batch["target"].cpu())
            all_probabilities.append(logits.sigmoid().cpu())
            all_paths.extend(batch["path"])
    return torch.cat(all_targets), torch.cat(all_probabilities), all_paths


def checkpoint_payload(model, args, epoch, best_macro_f1):
    config = {
        **model.configuration(),
        "hidden_channels": args.hidden_channels,
        "dropout": args.dropout,
        "image_size": args.image_size,
        "manifest": str(args.manifest.resolve()),
    }
    backbone_from = model._trainable_from  # saved scope mirrors the explicit fine-tuning boundary
    state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("backbone.")
        or (backbone_from is not None and int(key.split(".")[1]) >= backbone_from)
    }
    return {
        "model": state,
        "backbone_from": backbone_from,
        "config": config,
        "epoch": epoch,
        "best_macro_f1": best_macro_f1,
        "thresholds": [0.5, 0.5, 0.5],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "splits/leafmachine_v1.csv")
    parser.add_argument("--detector-weights", type=Path, default=DEFAULT_DETECTOR_WEIGHTS)
    parser.add_argument("--init-checkpoint", type=Path, help="initialize from a previously trained classifier head")
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--unfreeze-from", type=int, help="first LeafMachine module to fine-tune; omitted freezes all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="defaults to 1e-3 for a frozen backbone and 1e-5 when fine-tuning",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--limit", type=int, help="limit each split for smoke tests")
    parser.add_argument("--name", default="frozen_1280")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/leafmachine_classifier")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.learning_rate is None:
        args.learning_rate = 1e-3 if args.unfreeze_from is None else 1e-5

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)
    run_dir = args.output / args.name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite run without --force: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_data = PhenologyDataset(args.manifest, "train", args.image_size, augment=True, limit=args.limit)
    val_data = PhenologyDataset(args.manifest, "val", args.image_size, augment=False, limit=args.limit)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=generator,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    if args.init_checkpoint:
        model, initial_payload = load_classifier_checkpoint(
            args.init_checkpoint, detector_weights=args.detector_weights
        )
        initial_config = initial_payload["config"]
        args.hidden_channels = int(initial_config["hidden_channels"])
        args.dropout = float(initial_config["dropout"])
        args.temperature = float(initial_config["temperature"])
    else:
        model = LeafMachineClassifier(
            detector_weights=args.detector_weights,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
            temperature=args.temperature,
        )
    model.set_backbone_trainable(args.unfreeze_from)
    if args.image_size % max(model.feature_strides):
        raise ValueError(
            f"image size must be divisible by the largest feature stride ({max(model.feature_strides)})"
        )
    model.to(device)
    counts = model.parameter_counts()
    print(
        f"device={device} train={len(train_data)} val={len(val_data)} "
        f"features={model.feature_indices} strides={model.feature_strides} "
        f"parameters={counts['total']:,} trainable={counts['trainable']:,}"
    )

    positives = train_data.positive_counts()
    negatives = len(train_data) - positives
    pos_weight = (negatives / positives.clamp_min(1)).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history_path = run_dir / "history.csv"
    best_macro_f1 = -1.0
    stale_epochs = 0
    with history_path.open("w", newline="") as history_file:
        history = csv.writer(history_file)
        history.writerow([
            "epoch", "train_loss", "bud_f1", "flower_f1", "fruit_f1",
            "val_macro_f1", "val_exact", "learning_rate",
        ])
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for batch_index, batch in enumerate(train_loader, start=1):
                images = batch["image"].to(device)
                valid_mask = batch["valid_mask"].to(device)
                targets = batch["target"].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images, valid_mask), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 5.0
                )
                optimizer.step()
                running_loss += float(loss.detach())
                if batch_index % 50 == 0:
                    print(f"epoch {epoch:02} batch {batch_index:04}/{len(train_loader)} loss {loss.item():.4f}")

            scheduler.step()
            targets, probabilities, _ = collect_predictions(model, val_loader, device)
            metrics = compute_metrics(targets, probabilities)
            train_loss = running_loss / len(train_loader)
            learning_rate = optimizer.param_groups[0]["lr"]
            history.writerow([
                epoch,
                train_loss,
                metrics["bud"]["f1"],
                metrics["flower"]["f1"],
                metrics["fruit"]["f1"],
                metrics["macro_f1"],
                metrics["exact_match"],
                learning_rate,
            ])
            history_file.flush()
            print(
                f"epoch {epoch:02} loss={train_loss:.4f} val_macro_f1={metrics['macro_f1']:.3f} "
                f"val_exact={metrics['exact_match']:.3f}"
            )

            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                stale_epochs = 0
                torch.save(checkpoint_payload(model, args, epoch, best_macro_f1), run_dir / "best.pt")
            else:
                stale_epochs += 1
            torch.save(checkpoint_payload(model, args, epoch, best_macro_f1), run_dir / "last.pt")
            if stale_epochs >= args.patience:
                print(f"early stopping after {stale_epochs} epochs without improvement")
                break

    best_path = run_dir / "best.pt"
    payload = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=False)
    model.to(device)
    targets, probabilities, _ = collect_predictions(model, val_loader, device)
    thresholds = optimize_thresholds(targets, probabilities)
    tuned_metrics = compute_metrics(targets, probabilities, thresholds)
    payload["thresholds"] = list(thresholds)
    payload["validation_metrics"] = tuned_metrics
    torch.save(payload, best_path)
    print("validation thresholds selected on the validation split:")
    print(format_metrics(tuned_metrics, thresholds))
    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
