"""LoRA (Low-Rank Adaptation) and QLoRA support for efficient fine-tuning."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with Low-Rank Adaptation.

    Implements LoRA: adds trainable low-rank decomposition matrices (A, B) to
    a frozen pre-trained weight matrix. The modified forward is:
        output = Wx + (BAx) * scaling

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: Rank of the LoRA decomposition.
        alpha: LoRA scaling factor.
        dropout: Dropout probability on LoRA path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.linear.weight.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation.

        Args:
            x: Input tensor.

        Returns:
            Output with LoRA-adapted weights.
        """
        base_out = self.linear(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling

    def merge(self) -> None:
        """Merge LoRA weights into the base linear layer."""
        self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()


def apply_lora(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """Apply LoRA to specified linear layers in a model.

    Args:
        model: The base model.
        target_modules: List of module name patterns to apply LoRA to.
            If None, applies to all nn.Linear layers.
        rank: LoRA rank.
        alpha: LoRA alpha scaling factor.
        dropout: Dropout on LoRA path.

    Returns:
        Model with LoRA-adapted layers.
    """
    if target_modules is None:
        target_modules = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if not target_modules or any(t in name for t in target_modules):
                lora_layer = LoRALinear(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                lora_layer.linear = module
                module.weight.requires_grad_(False)
                _set_module(model, name, lora_layer)

    return model


def apply_qlora(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    bits: int = 4,
) -> nn.Module:
    """Apply QLoRA (quantized LoRA) to a model.

    Quantizes base weights to specified bit-width (simulated via rounding
    and clamping) and adds LoRA adapters on top. For true NF4 quantization,
    use bitsandbytes integration externally.

    Args:
        model: The base model.
        target_modules: Module name patterns to target.
        rank: LoRA rank.
        alpha: LoRA alpha scaling.
        bits: Quantization bit-width for base weights (4 or 8).

    Returns:
        Model with QLoRA adapters (base weights quantized).
    """
    if target_modules is None:
        target_modules = []

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if not target_modules or any(t in name for t in target_modules):
                lora_layer = LoRALinear(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=0.0,
                )

                lora_layer.linear = module
                module.weight.requires_grad_(False)

                with torch.no_grad():
                    weight = module.weight.data
                    if bits == 8:
                        scale = weight.abs().max() / 127.0
                        if scale > 0:
                            quantized = torch.clamp(torch.round(weight / scale), -128, 127)
                            module.weight.data = quantized * scale
                    elif bits == 4:
                        scale = weight.abs().max() / 7.0
                        if scale > 0:
                            quantized = torch.clamp(torch.round(weight / scale), -8, 7)
                            module.weight.data = quantized * scale

                _set_module(model, name, lora_layer)

    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge all LoRA weights into base model weights.

    After merging, LoRA parameters are zeroed and the model behaves
    as a standard model with no inference overhead.

    Args:
        model: Model with LoRA layers.

    Returns:
        Model with merged weights.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
    return model


def _set_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Set a submodule by dot-separated name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
