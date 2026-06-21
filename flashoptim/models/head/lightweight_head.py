"""Lightweight detection/classification head for optimized models."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class LightweightHead(nn.Module):
    """Lightweight prediction head with reduced parameters.

    Designed for efficient inference on edge devices. Uses depthwise-separable
    convolutions and shared parameters to minimize computation.

    Args:
        in_channels: List of input channel counts from neck features.
        num_classes: Number of prediction classes.
        num_anchors: Number of anchors per location.
        use_depthwise: Use depthwise-separable convolutions.
        shared_head: Share head weights across scales.
    """

    def __init__(
        self,
        in_channels: List[int] = [64, 128, 256],
        num_classes: int = 80,
        num_anchors: int = 1,
        use_depthwise: bool = True,
        shared_head: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.use_depthwise = use_depthwise

        hidden_dim = in_channels[0]
        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        for ch in in_channels:
            cls_head = self._make_head(ch, hidden_dim, num_anchors * num_classes)
            reg_head = self._make_head(ch, hidden_dim, num_anchors * 4)
            self.cls_heads.append(cls_head)
            self.reg_heads.append(reg_head)

    def _make_head(self, in_ch: int, hidden_ch: int, out_ch: int) -> nn.Sequential:
        """Build a lightweight prediction branch."""
        if self.use_depthwise:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_ch, out_ch, kernel_size=1),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_ch, out_ch, kernel_size=1),
            )

    def forward(self, features: List[torch.Tensor]) -> dict:
        """Forward pass through the lightweight head.

        Args:
            features: List of feature maps from the neck.

        Returns:
            Dictionary with 'cls_preds' and 'reg_preds' lists.
        """
        cls_preds = []
        reg_preds = []

        for i, feat in enumerate(features):
            cls_preds.append(self.cls_heads[i](feat))
            reg_preds.append(self.reg_heads[i](feat))

        return {"cls_preds": cls_preds, "reg_preds": reg_preds}
