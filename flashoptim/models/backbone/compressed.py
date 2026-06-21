"""Compressed backbone variants for efficient inference."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class CompressedBackbone(nn.Module):
    """Compressed backbone network with reduced channels and depth.

    Provides a lighter backbone variant suitable for edge deployment.
    Can be created from a full backbone via channel pruning or NAS.

    Args:
        in_channels: Number of input channels (typically 3).
        base_channels: Base channel width (reduced from full model).
        depth_multiplier: Depth scaling factor (0.0 to 1.0).
        width_multiplier: Width scaling factor (0.0 to 1.0).
        activation: Activation function name ('relu', 'silu', 'hardswish').
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        depth_multiplier: float = 1.0,
        width_multiplier: float = 1.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = int(base_channels * width_multiplier)
        self.depth_multiplier = depth_multiplier
        self.width_multiplier = width_multiplier

        act_fn = self._get_activation(activation)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.base_channels),
            act_fn,
        )

        channels = [
            self.base_channels,
            self.base_channels * 2,
            self.base_channels * 4,
            self.base_channels * 8,
        ]
        self.stages = nn.ModuleList()
        in_ch = self.base_channels
        for out_ch in channels[1:]:
            num_blocks = max(1, int(3 * depth_multiplier))
            stage = self._make_stage(in_ch, out_ch, num_blocks, act_fn)
            self.stages.append(stage)
            in_ch = out_ch

        self.out_channels = channels[-1]

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(inplace=True),
            "silu": nn.SiLU(inplace=True),
            "hardswish": nn.Hardswish(inplace=True),
        }
        return activations.get(name, nn.SiLU(inplace=True))

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, num_blocks: int, act_fn: nn.Module) -> nn.Sequential:
        """Create a backbone stage with multiple blocks."""
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            act_fn,
        ]
        for _ in range(num_blocks - 1):
            layers.extend([
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                act_fn,
            ])
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning multi-scale features.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            List of feature maps at different scales.
        """
        features = []
        x = self.stem(x)
        features.append(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

    @classmethod
    def from_full_backbone(
        cls,
        full_backbone: nn.Module,
        width_multiplier: float = 0.5,
        depth_multiplier: float = 0.5,
    ) -> "CompressedBackbone":
        """Create a compressed backbone from a full-size backbone.

        Args:
            full_backbone: The original full-size backbone.
            width_multiplier: Channel reduction factor.
            depth_multiplier: Depth reduction factor.

        Returns:
            A new CompressedBackbone instance.
        """
        return cls(
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
        )
