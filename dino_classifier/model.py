"""DINOv2/DINOv3 backbone, LoRA adapters, and a per-organ attention-pooling head.

Buds, flowers, and fruits all occupy a small part of a sheet, so the head reads
every patch token rather than the global CLS token. Each organ has its own
gated attention over patches (Ilse et al. 2018, attention-based multiple
instance learning): the sheet label supervises the pooled feature, and the
attention weights show which patches drove the call.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

ORGANS = ("bud", "flower", "fruit")
# DINOv2 and DINOv3 LVD-1689M weights both use ImageNet normalization.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update (Hu et al. 2021)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)  # starts as the pretrained layer exactly
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * self.scale


class AttentionPool(nn.Module):
    """Gated attention over patch tokens, one attention map per organ."""

    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.value = nn.Linear(dim, hidden)
        self.gate = nn.Linear(dim, hidden)
        self.score = nn.Linear(hidden, len(ORGANS))
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.zeros(len(ORGANS), dim))
        self.bias = nn.Parameter(torch.zeros(len(ORGANS)))

    def forward(self, tokens, patch_mask):
        h = self.norm(tokens.float())
        scores = self.score(torch.tanh(self.value(h)) * torch.sigmoid(self.gate(h)))
        scores = scores.masked_fill(~patch_mask.unsqueeze(-1), float("-inf"))
        attention = scores.softmax(dim=1)  # (batch, patches, organs)
        pooled = torch.einsum("bpo,bpd->bod", attention, self.dropout(h))
        return (pooled * self.weight).sum(-1) + self.bias, attention


class DinoPhenology(nn.Module):
    def __init__(self, backbone: str, mode: str, lora_rank: int = 16, unfreeze: int = 4,
                 hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone, attn_implementation="sdpa")
        config = self.backbone.config
        self.prefix = 1 + getattr(config, "num_register_tokens", 0)
        self.patch = config.patch_size  # 14 for DINOv2, 16 for DINOv3
        self.mode = mode
        self.backbone.requires_grad_(False)
        blocks = self.blocks()
        if mode == "lora":
            for block in blocks:
                for name in ATTENTION_PROJECTIONS:
                    setattr(block.attention, name, LoRALinear(getattr(block.attention, name), lora_rank, lora_rank))
        elif mode == "last":
            for block in blocks[-unfreeze:]:
                block.requires_grad_(True)
            self.final_norm().requires_grad_(True)
        elif mode != "frozen":
            raise ValueError(f"unknown mode {mode}")
        if mode != "frozen":
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.head = AttentionPool(config.hidden_size, hidden, dropout)

    def blocks(self):
        encoder = self.backbone.encoder if hasattr(self.backbone, "encoder") else self.backbone.model  # v2, v3
        return encoder.layer

    def final_norm(self):
        return self.backbone.layernorm if hasattr(self.backbone, "layernorm") else self.backbone.norm  # v2, v3

    def train(self, mode: bool = True):
        super().train(mode)
        if self.mode == "frozen":
            self.backbone.eval()
        return self

    def forward(self, pixels, patch_mask):
        if self.mode == "frozen":
            with torch.no_grad():
                tokens = self.backbone(pixel_values=pixels).last_hidden_state
        else:
            tokens = self.backbone(pixel_values=pixels).last_hidden_state
        return self.head(tokens[:, self.prefix :], patch_mask)

    def trainable_state(self):
        """Only what training changed; the backbone is reloaded from its pretrained weights."""
        names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {name: tensor for name, tensor in self.state_dict().items() if name in names}

    def parameter_groups(self, head_lr: float, backbone_lr: float, layer_decay: float):
        groups = [{"params": list(self.head.parameters()), "lr": head_lr}]
        if self.mode == "lora":
            params = [p for n, p in self.backbone.named_parameters() if p.requires_grad]
            groups.append({"params": params, "lr": backbone_lr})
        elif self.mode == "last":
            groups.append({"params": list(self.final_norm().parameters()), "lr": backbone_lr})
            trainable = [block for block in self.blocks() if any(p.requires_grad for p in block.parameters())]
            for depth, block in enumerate(reversed(trainable)):
                groups.append({"params": list(block.parameters()), "lr": backbone_lr * layer_decay**depth})
        for group in groups:
            group["initial_lr"] = group["lr"]
        return groups


def patch_mask_from_pixels(valid: torch.Tensor, patch: int) -> torch.Tensor:
    """(batch, H, W) pixel validity -> (batch, patches) in the backbone's row-major patch order."""
    return F.max_pool2d(valid.float().unsqueeze(1), patch).flatten(1) > 0
