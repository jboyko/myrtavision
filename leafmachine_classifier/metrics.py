"""Multi-label metrics and validation-only threshold selection."""
from __future__ import annotations

import torch


ORGANS = ("bud", "flower", "fruit")


def compute_metrics(targets: torch.Tensor, probabilities: torch.Tensor, thresholds=(0.5, 0.5, 0.5)):
    targets = targets.bool()
    thresholds = torch.as_tensor(thresholds, dtype=probabilities.dtype).view(1, -1)
    predictions = probabilities >= thresholds
    result = {}
    f1_values = []
    for index, organ in enumerate(ORGANS):
        truth = targets[:, index]
        prediction = predictions[:, index]
        tp = int((truth & prediction).sum())
        fp = int((~truth & prediction).sum())
        fn = int((truth & ~prediction).sum())
        correct = int((truth == prediction).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        result[organ] = {
            "accuracy": correct / len(truth),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    result["macro_f1"] = sum(f1_values) / len(f1_values)
    result["exact_match"] = float((targets == predictions).all(dim=1).float().mean())
    result["n"] = len(targets)
    return result


def optimize_thresholds(targets: torch.Tensor, probabilities: torch.Tensor):
    thresholds = []
    for index in range(len(ORGANS)):
        candidates = torch.unique(probabilities[:, index]).tolist() + [0.5]
        best_threshold, best_f1 = 0.5, -1.0
        for threshold in candidates:
            truth = targets[:, index].bool()
            prediction = probabilities[:, index] >= threshold
            tp = int((truth & prediction).sum())
            fp = int((~truth & prediction).sum())
            fn = int((truth & ~prediction).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if f1 > best_f1:
                best_threshold, best_f1 = float(threshold), f1
        thresholds.append(best_threshold)
    return tuple(thresholds)


def format_metrics(metrics: dict, thresholds=(0.5, 0.5, 0.5)) -> str:
    lines = [f"{'organ':8} {'thresh':>7} {'acc':>6} {'prec':>6} {'rec':>6} {'F1':>6}"]
    for organ, threshold in zip(ORGANS, thresholds):
        values = metrics[organ]
        lines.append(
            f"{organ:8} {threshold:7.3f} {values['accuracy']:6.3f} "
            f"{values['precision']:6.3f} {values['recall']:6.3f} {values['f1']:6.3f}"
        )
    lines.append(
        f"macro F1: {metrics['macro_f1']:.3f}  exact: {metrics['exact_match']:.3f}  n={metrics['n']}"
    )
    return "\n".join(lines)
