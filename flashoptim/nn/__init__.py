"""Neural network building blocks for FlashOptim."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ConvModule(nn.Module):
    """Convolution + BatchNorm + Activation building block.

    A fused module that combines a convolution layer with optional batch
    normalization and activation, commonly used as a repeating unit
    in detection and classification backbones.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding. Defaults to kernel_size // 2.
        groups: Convolution groups.
        bias: Whether to include a bias term (auto-disabled when using BN).
        norm: Whether to include BatchNorm.
        act: Activation function ('relu', 'silu', 'hardswish', or None).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        bias: bool = False,
        norm: bool = True,
        act: Optional[str] = "silu",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias if not norm else False,
        )
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = self._build_activation(act)

    @staticmethod
    def _build_activation(act: Optional[str]) -> nn.Module:
        """Build activation function by name."""
        if act is None:
            return nn.Identity()
        activations = {
            "relu": nn.ReLU(inplace=True),
            "silu": nn.SiLU(inplace=True),
            "hardswish": nn.Hardswish(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
        }
        if act not in activations:
            raise ValueError(f"Unknown activation: {act}. Options: {list(activations.keys())}")
        return activations[act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: conv -> bn -> act.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H', W').
        """
        return self.act(self.bn(self.conv(x)))


class DepthwiseConvModule(nn.Module):
    """Depthwise separable convolution block.

    Factorizes a standard convolution into a depthwise convolution
    (spatial filtering) followed by a pointwise convolution (channel mixing),
    significantly reducing computation and parameters.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Depthwise convolution kernel size.
        stride: Depthwise convolution stride.
        padding: Depthwise convolution padding. Defaults to kernel_size // 2.
        norm: Whether to include BatchNorm after each sub-convolution.
        act: Activation function name.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        norm: bool = True,
        act: Optional[str] = "silu",
    ) -> None:
        super().__init__()

        self.depthwise = ConvModule(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            norm=norm,
            act=act,
        )
        self.pointwise = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm=norm,
            act=act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: depthwise conv -> pointwise conv.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H', W').
        """
        return self.pointwise(self.depthwise(x))


__all__ = ["ConvModule", "DepthwiseConvModule"]
