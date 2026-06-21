"""Observers for tracking quantization statistics."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class MinMaxObserver(nn.Module):
    """Tracks min/max values of tensors for quantization range estimation.

    Simple and fast observer that tracks global or per-channel min/max values.

    Args:
        per_channel: Track statistics per output channel.
        symmetric: Use symmetric quantization range.
        dtype: Target quantization dtype ('int8', 'uint8').
    """

    def __init__(
        self,
        per_channel: bool = False,
        symmetric: bool = True,
        dtype: str = "int8",
    ) -> None:
        super().__init__()
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.dtype = dtype

        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("num_batches", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Observe tensor and update statistics.

        Args:
            x: Input tensor to observe.

        Returns:
            Input tensor unchanged (passthrough).
        """
        with torch.no_grad():
            if self.per_channel:
                min_val = x.detach().reshape(x.shape[0], -1).min(dim=1).values
                max_val = x.detach().reshape(x.shape[0], -1).max(dim=1).values
            else:
                min_val = x.detach().min()
                max_val = x.detach().max()

            self.min_val = torch.min(self.min_val, min_val)
            self.max_val = torch.max(self.max_val, max_val)
            self.num_batches += 1

        return x

    def compute_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute quantization scale and zero-point from observed statistics.

        Returns:
            Tuple of (scale, zero_point) tensors.
        """
        if self.dtype == "int8":
            qmin, qmax = -128, 127
        else:
            qmin, qmax = 0, 255

        min_val = self.min_val
        max_val = self.max_val

        if self.symmetric:
            abs_max = torch.max(min_val.abs(), max_val.abs())
            scale = abs_max / ((qmax - qmin) / 2)
            zero_point = torch.zeros_like(scale)
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - torch.round(min_val / scale)

        scale = torch.clamp(scale, min=1e-8)
        return scale, zero_point

    def reset(self) -> None:
        """Reset observer statistics."""
        self.min_val.fill_(float("inf"))
        self.max_val.fill_(float("-inf"))
        self.num_batches.zero_()


class HistogramObserver(nn.Module):
    """Histogram-based observer for more accurate quantization ranges.

    Collects a histogram of tensor values and uses entropy minimization
    or MSE minimization to find optimal clipping thresholds.

    Args:
        bins: Number of histogram bins.
        method: Threshold selection method ('entropy', 'mse', 'percentile').
        percentile: Percentile for clipping (used with 'percentile' method).
        per_channel: Track per-channel histograms.
    """

    def __init__(
        self,
        bins: int = 2048,
        method: str = "entropy",
        percentile: float = 99.99,
        per_channel: bool = False,
    ) -> None:
        super().__init__()
        self.bins = bins
        self.method = method
        self.percentile = percentile
        self.per_channel = per_channel

        self.register_buffer("histogram", torch.zeros(bins))
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("num_batches", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Observe tensor and update histogram.

        Args:
            x: Input tensor to observe.

        Returns:
            Input tensor unchanged (passthrough).
        """
        with torch.no_grad():
            x_flat = x.detach().flatten().float()
            self.min_val = torch.min(self.min_val, x_flat.min())
            self.max_val = torch.max(self.max_val, x_flat.max())

            hist = torch.histc(x_flat, bins=self.bins, min=self.min_val.item(), max=self.max_val.item())
            self.histogram += hist
            self.num_batches += 1

        return x

    def compute_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute optimal quantization parameters from histogram.

        Returns:
            Tuple of (scale, zero_point) tensors.

        Raises:
            NotImplementedError: Entropy/MSE threshold selection pending.
        """
        if self.method == "percentile":
            return self._percentile_threshold()
        elif self.method == "entropy":
            return self._entropy_threshold()
        elif self.method == "mse":
            return self._mse_threshold()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _percentile_threshold(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scale/zp using percentile clipping."""
        cumsum = self.histogram.cumsum(0)
        total = cumsum[-1]
        low_idx = (cumsum >= total * (1 - self.percentile / 100)).nonzero(as_tuple=True)[0][0]
        high_idx = (cumsum >= total * self.percentile / 100).nonzero(as_tuple=True)[0][0]

        bin_width = (self.max_val - self.min_val) / self.bins
        clip_min = self.min_val + low_idx * bin_width
        clip_max = self.min_val + high_idx * bin_width

        scale = (clip_max - clip_min) / 255
        zero_point = torch.round(-clip_min / scale)
        return torch.clamp(scale, min=1e-8), zero_point

    def _entropy_threshold(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find optimal threshold using KL-divergence (entropy) minimization.

        Iteratively tries different numbers of quantization bins and picks the
        threshold that minimizes information loss when quantizing the histogram.
        """
        hist = self.histogram.clone()
        hist = hist / (hist.sum() + 1e-12)
        bin_width = (self.max_val - self.min_val) / self.bins
        num_quantized_bins = 128

        best_divergence = float("inf")
        best_end = self.bins

        for num_bins in range(num_quantized_bins, self.bins + 1, 32):
            reference = hist[:num_bins].clone()
            reference[reference == 0] = 1e-12

            bin_ratio = num_bins / num_quantized_bins
            quantized = torch.zeros(num_quantized_bins)
            for i in range(num_quantized_bins):
                start_bin = int(i * bin_ratio)
                end_bin = int((i + 1) * bin_ratio)
                end_bin = min(end_bin, num_bins)
                quantized[i] = reference[start_bin:end_bin].sum()

            expanded = torch.zeros(num_bins)
            for i in range(num_quantized_bins):
                start_bin = int(i * bin_ratio)
                end_bin = int((i + 1) * bin_ratio)
                end_bin = min(end_bin, num_bins)
                count = end_bin - start_bin
                if count > 0:
                    expanded[start_bin:end_bin] = quantized[i] / count

            expanded = expanded / (expanded.sum() + 1e-12)
            reference = reference / (reference.sum() + 1e-12)

            divergence = (reference * (torch.log(reference + 1e-12) - torch.log(expanded + 1e-12))).sum()

            if divergence < best_divergence:
                best_divergence = divergence
                best_end = num_bins

        clip_min = self.min_val
        clip_max = self.min_val + best_end * bin_width

        scale = (clip_max - clip_min) / 255
        zero_point = torch.round(-clip_min / scale)
        return torch.clamp(scale, min=1e-8), zero_point

    def _mse_threshold(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find optimal threshold by minimizing quantization MSE.

        Searches over possible clipping thresholds and selects the one
        that minimizes the mean squared error of the quantized distribution.
        """
        hist = self.histogram.clone()
        bin_width = (self.max_val - self.min_val) / self.bins
        bin_centers = self.min_val + (torch.arange(self.bins).float() + 0.5) * bin_width

        best_mse = float("inf")
        best_clip_max = self.max_val

        num_candidates = 80
        for i in range(1, num_candidates + 1):
            candidate_bins = int(self.bins * i / num_candidates)
            if candidate_bins < 8:
                continue
            clip_max_candidate = self.min_val + candidate_bins * bin_width

            scale_candidate = (clip_max_candidate - self.min_val) / 255.0
            if scale_candidate <= 0:
                continue

            centers_clipped = bin_centers[:candidate_bins]
            quantized_centers = torch.clamp(
                torch.round((centers_clipped - self.min_val) / scale_candidate),
                0, 255,
            )
            dequantized = quantized_centers * scale_candidate + self.min_val

            hist_slice = hist[:candidate_bins]
            mse = ((centers_clipped - dequantized).pow(2) * hist_slice).sum()
            overflow_mse = (bin_centers[candidate_bins:].pow(2) * hist[candidate_bins:]).sum()
            total_mse = mse + overflow_mse

            if total_mse < best_mse:
                best_mse = total_mse
                best_clip_max = clip_max_candidate

        clip_min = self.min_val
        scale = (best_clip_max - clip_min) / 255
        zero_point = torch.round(-clip_min / scale)
        return torch.clamp(scale, min=1e-8), zero_point

    def reset(self) -> None:
        """Reset observer statistics."""
        self.histogram.zero_()
        self.min_val.fill_(float("inf"))
        self.max_val.fill_(float("-inf"))
        self.num_batches.zero_()
