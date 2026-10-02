"""Multiscale classifier built from LeafMachine2's detector feature pyramid."""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LM2_DETECTOR_ROOT = PROJECT_ROOT / "third_party/LeafMachine2/leafmachine2/component_detector"
DEFAULT_DETECTOR_WEIGHTS = PROJECT_ROOT / "third_party/LeafMachine2/checkpoints/best.pt"
ORGANS = ("bud", "flower", "fruit")


def project_relative(path: str | Path) -> str:
    """Store paths inside the repo relative to it, so checkpoints survive moving machines."""
    path = Path(path).absolute()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_project_path(stored: str | Path) -> Path:
    """Inverse of project_relative: relative paths are anchored at the repo root."""
    path = Path(stored)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_leafmachine_detector(weights: str | Path):
    """Load the GPL-3.0 LeafMachine2 YOLOv5 checkpoint without fusing layers."""
    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Missing LeafMachine2 detector weights: {weights}")
    detector_root = str(LM2_DETECTOR_ROOT.resolve())
    if detector_root not in sys.path:
        sys.path.insert(0, detector_root)
    os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")
    mpl_cache = Path(tempfile.gettempdir()) / "myrtavision-matplotlib"
    mpl_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    from models.experimental import attempt_load  # pylint: disable=import-outside-toplevel

    return attempt_load(str(weights), map_location="cpu", inplace=True, fuse=False)


class EvidenceHead(nn.Module):
    """Turn one pyramid feature map into three spatial evidence maps."""

    def __init__(self, in_channels: int, hidden_channels: int, dropout: float):
        super().__init__()
        groups = min(8, hidden_channels)
        while hidden_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, len(ORGANS), kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class LeafMachineClassifier(nn.Module):
    """Three-label classifier using every feature scale consumed by YOLO Detect.

    LeafMachine's bounding-box head is discarded. The detector source layers
    are discovered from the checkpoint, given small spatial evidence heads, and
    pooled with normalized log-sum-exp so sparse structures are not averaged
    away. Scale weights are learned independently for each organ.
    """

    def __init__(
        self,
        detector_weights: str | Path = DEFAULT_DETECTOR_WEIGHTS,
        hidden_channels: int = 128,
        dropout: float = 0.1,
        temperature: float = 0.25,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        detector = _load_leafmachine_detector(detector_weights)
        detect_layer = detector.model[-1]
        if detect_layer.__class__.__name__ != "Detect":
            raise TypeError("LeafMachine checkpoint does not end in a YOLO Detect layer")

        self.detector_weights = str(Path(detector_weights).absolute())
        with Path(self.detector_weights).open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.detector_sha256 = digest.hexdigest()
        self.feature_indices = tuple(int(index) for index in detect_layer.f)
        self.feature_strides = tuple(int(value) for value in detector.stride.tolist())
        last_feature = max(self.feature_indices)
        self.backbone = nn.ModuleList(list(detector.model)[: last_feature + 1])
        self.saved_indices = set(int(index) for index in detector.save)
        self.temperature = float(temperature)
        self._trainable_from: int | None = None

        channels = self._infer_feature_channels()
        self.evidence_heads = nn.ModuleList(
            EvidenceHead(channel, hidden_channels, dropout) for channel in channels
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(channels), len(ORGANS)))
        self.organ_bias = nn.Parameter(torch.zeros(len(ORGANS)))
        self.set_backbone_trainable(None)

    def _forward_pyramid(self, images: torch.Tensor) -> list[torch.Tensor]:
        saved: list[torch.Tensor | None] = []
        features: dict[int, torch.Tensor] = {}
        x = images
        for layer in self.backbone:
            if layer.f != -1:
                if isinstance(layer.f, int):
                    x = saved[layer.f]
                else:
                    x = [x if index == -1 else saved[index] for index in layer.f]
            x = layer(x)
            saved.append(x if layer.i in self.saved_indices else None)
            if layer.i in self.feature_indices:
                features[layer.i] = x
        return [features[index] for index in self.feature_indices]

    def _infer_feature_channels(self) -> tuple[int, ...]:
        states = [layer.training for layer in self.backbone]
        self.backbone.eval()
        with torch.inference_mode():
            features = self._forward_pyramid(torch.zeros(1, 3, 128, 128))
        for layer, state in zip(self.backbone, states):
            layer.train(state)
        return tuple(int(feature.shape[1]) for feature in features)

    def set_backbone_trainable(self, from_layer: int | None) -> None:
        """Freeze the pyramid, or unfreeze modules at and after one layer index."""
        if from_layer is not None and not 0 <= from_layer <= max(self.feature_indices):
            raise ValueError(f"from_layer must be between 0 and {max(self.feature_indices)}")
        self._trainable_from = from_layer
        for index, layer in enumerate(self.backbone):
            trainable = from_layer is not None and index >= from_layer
            for parameter in layer.parameters():
                parameter.requires_grad = trainable
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        for index, layer in enumerate(self.backbone):
            trainable = self._trainable_from is not None and index >= self._trainable_from
            layer.train(mode and trainable)
        return self

    def _pool(self, evidence: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        batch, channels, height, width = evidence.shape
        if valid_mask is None:
            mask = torch.ones((batch, 1, height, width), dtype=torch.bool, device=evidence.device)
        else:
            mask = F.interpolate(valid_mask.float(), size=(height, width), mode="nearest") >= 0.5
        mask = mask.expand(-1, channels, -1, -1)
        scaled = evidence / self.temperature
        scaled = scaled.masked_fill(~mask, torch.finfo(scaled.dtype).min)
        counts = mask.flatten(2).sum(dim=-1).clamp_min(1)
        return self.temperature * (
            torch.logsumexp(scaled.flatten(2), dim=-1) - torch.log(counts)
        )

    def forward(
        self,
        images: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_maps: bool = False,
    ):
        features = self._forward_pyramid(images)
        maps = [head(feature) for head, feature in zip(self.evidence_heads, features)]
        pooled = torch.stack(
            [self._pool(evidence, valid_mask) for evidence in maps], dim=1
        )  # batch, scale, organ
        scale_weights = self.scale_logits.softmax(dim=0)
        logits = (pooled * scale_weights.unsqueeze(0)).sum(dim=1) + self.organ_bias
        if return_maps:
            return logits, maps, scale_weights
        return logits

    def configuration(self) -> dict:
        return {
            "detector_weights": project_relative(self.detector_weights),
            "detector_sha256": self.detector_sha256,
            "feature_indices": list(self.feature_indices),
            "feature_strides": list(self.feature_strides),
            "temperature": self.temperature,
            "trainable_from": self._trainable_from,
            "organs": list(ORGANS),
        }

    def parameter_counts(self) -> dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
        }


def load_classifier_checkpoint(
    checkpoint_path: str | Path,
    detector_weights: str | Path | None = None,
    map_location: str | torch.device = "cpu",
):
    """Rebuild the classifier from LeafMachine weights and load learned state."""
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    config = payload["config"]
    weights = detector_weights
    if weights is None:
        stored = config.get("detector_weights")
        weights = resolve_project_path(stored) if stored else DEFAULT_DETECTOR_WEIGHTS
        if not Path(weights).is_file():
            # e.g. an absolute path from another machine; the checksum below still guards identity
            weights = DEFAULT_DETECTOR_WEIGHTS
    model = LeafMachineClassifier(
        detector_weights=weights,
        hidden_channels=int(config.get("hidden_channels", 128)),
        dropout=float(config.get("dropout", 0.1)),
        temperature=float(config.get("temperature", 0.25)),
    )
    expected_digest = config.get("detector_sha256")
    if expected_digest and model.detector_sha256 != expected_digest:
        raise RuntimeError(
            "LeafMachine detector checksum differs from the checkpoint's pretrained base"
        )
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    backbone_from = payload.get("backbone_from")
    expected_missing = {
        key
        for key in model.state_dict()
        if key.startswith("backbone.")
        and (backbone_from is None or int(key.split(".")[1]) < int(backbone_from))
    }
    unexpected_missing = set(missing) - expected_missing
    if unexpected_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state mismatch: unexpected missing={unexpected_missing}, "
            f"unexpected keys={unexpected}"
        )
    model.set_backbone_trainable(config.get("trainable_from"))
    return model, payload
