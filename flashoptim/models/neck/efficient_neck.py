"""Efficient Feature Pyramid Network (FPN) neck for optimized models."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientNeck(nn.Module):
    """Efficient FPN neck with reduced computation.

    Uses lightweight operations (depthwise convolutions, channel reduction)
    to create a multi-scale feature pyramid suitable for edge deployment.

    Args:
        in_channels: List of input channel counts from backbone stages.
        out_channels: Unified output channel count for all levels.
        num_extra_levels: Number of additional downsampled levels.
        use_depthwise: Use depthwise-separable convolutions.
    """

    def __init__(
        self,
        in_channels: List[int] = [64, 128, 256, 512],
        out_channels: int = 128,
        num_extra_levels: int = 0,
        use_depthwise: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for ch in in_channels:
            lateral = nn.Conv2d(ch, out_channels, kernel_size=1, bias=False)
            if use_depthwise:
                fpn = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
            else:
                fpn = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
            self.lateral_convs.append(lateral)
            self.fpn_convs.append(fpn)

        self.extra_levels = nn.ModuleList()
        for _ in range(num_extra_levels):
            self.extra_levels.append(
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False)
            )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass building the feature pyramid.

        Args:
            features: Multi-scale features from the backbone.

        Returns:
            List of FPN feature maps at each level.
        """
        assert len(features) == len(self.in_channels)

        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode="nearest")
            laterals[i - 1] = laterals[i - 1] + upsampled

        outputs = [fpn(lat) for fpn, lat in zip(self.fpn_convs, laterals)]

        for extra in self.extra_levels:
            outputs.append(extra(outputs[-1]))

        return outputs
