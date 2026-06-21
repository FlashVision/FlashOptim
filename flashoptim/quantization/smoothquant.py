"""SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs.

Migrates quantization difficulty from activations to weights by applying
mathematically equivalent per-channel smoothing transformations.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashoptim.registry import QUANTIZERS


@QUANTIZERS.register("smoothquant")
class SmoothQuantizer:
    """SmoothQuant: activation-weight smoothing for easier quantization.

    Applies per-channel scaling: Y = (X diag(s)^{-1}) (diag(s) W)
    to balance quantization difficulty between activations and weights.

    Args:
        alpha: Migration strength (0 = all on activations, 1 = all on weights).
            Typical good values: 0.5 for LLMs, 0.75 for vision models.
        bits_weight: Weight quantization bit-width.
        bits_activation: Activation quantization bit-width.
        calibration_samples: Number of calibration samples.
        per_channel: Use per-channel quantization for weights.

    Example:
        >>> sq = SmoothQuantizer(alpha=0.5)
        >>> smoothed_model = sq.quantize(model, calibration_loader)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        bits_weight: int = 8,
        bits_activation: int = 8,
        calibration_samples: int = 128,
        per_channel: bool = True,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.bits_weight = bits_weight
        self.bits_activation = bits_activation
        self.calibration_samples = calibration_samples
        self.per_channel = per_channel
        self._smooth_scales: Dict[str, torch.Tensor] = {}

    def quantize(
        self,
        model: nn.Module,
        calibration_loader: Any,
    ) -> nn.Module:
        """Apply SmoothQuant and quantize the model.

        Args:
            model: Model to quantize.
            calibration_loader: DataLoader for collecting activation statistics.

        Returns:
            Smoothed and quantized model.
        """
        model = copy.deepcopy(model)
        model.eval()

        act_scales = self._collect_activation_scales(model, calibration_loader)

        self._apply_smoothing(model, act_scales)

        self._quantize_weights(model)

        return model

    def smooth_only(
        self,
        model: nn.Module,
        calibration_loader: Any,
    ) -> nn.Module:
        """Apply only the smoothing transformation without quantization.

        Useful for preparing a model before applying a separate quantization method.
        """
        model = copy.deepcopy(model)
        model.eval()
        act_scales = self._collect_activation_scales(model, calibration_loader)
        self._apply_smoothing(model, act_scales)
        return model

    def _collect_activation_scales(
        self,
        model: nn.Module,
        dataloader: Any,
    ) -> Dict[str, torch.Tensor]:
        """Collect per-channel activation magnitude statistics.

        Returns the maximum absolute activation value per input channel
        for each quantizable layer.
        """
        act_maxes: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                inp = input[0].detach()
                if inp.dim() > 2:
                    inp = inp.reshape(-1, inp.shape[-1])
                channel_max = inp.abs().max(dim=0).values
                if name in act_maxes:
                    act_maxes[name] = torch.max(act_maxes[name], channel_max)
                else:
                    act_maxes[name] = channel_max
            return hook_fn

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(make_hook(name)))

        count = 0
        with torch.no_grad():
            for batch in dataloader:
                if count >= self.calibration_samples:
                    break
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                model(inputs)
                count += inputs.shape[0]

        for h in hooks:
            h.remove()

        return act_maxes

    def _apply_smoothing(
        self,
        model: nn.Module,
        act_scales: Dict[str, torch.Tensor],
    ) -> None:
        """Apply per-channel smoothing to balance activation/weight ranges.

        Computes s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
        then applies: W_new = W * diag(s), which is equivalent to
        dividing activations by s.
        """
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue

            act_scale = act_scales.get(name)
            if act_scale is None:
                continue

            if isinstance(module, nn.Linear):
                weight = module.weight.data.float()
                n_in = weight.shape[1]
            elif isinstance(module, nn.Conv2d):
                weight = module.weight.data.float().flatten(1)
                n_in = weight.shape[1]
            else:
                continue

            act_s = act_scale[:n_in] if act_scale.shape[0] >= n_in else (
                F.pad(act_scale, (0, n_in - act_scale.shape[0]), value=1.0)
            )
            act_s = act_s.clamp(min=1e-8)

            weight_scale = weight.abs().max(dim=0).values.clamp(min=1e-8)

            smooth_scale = (act_s.pow(self.alpha) / weight_scale.pow(1 - self.alpha)).clamp(min=1e-8)

            if isinstance(module, nn.Linear):
                module.weight.data = (weight * smooth_scale.unsqueeze(0)).to(module.weight.dtype)
            elif isinstance(module, nn.Conv2d):
                scaled = weight * smooth_scale.unsqueeze(0)
                module.weight.data = scaled.reshape(module.weight.shape).to(module.weight.dtype)

            self._smooth_scales[name] = smooth_scale

    def _quantize_weights(self, model: nn.Module) -> None:
        """Apply weight quantization after smoothing."""
        qmin = 0
        qmax = (1 << self.bits_weight) - 1

        for module in model.modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue

            weight = module.weight.data.float()

            if self.per_channel:
                if weight.dim() >= 2:
                    flat_w = weight.flatten(1)
                    w_min = flat_w.min(dim=1, keepdim=True).values
                    w_max = flat_w.max(dim=1, keepdim=True).values
                    scale = (w_max - w_min).clamp(min=1e-8) / (qmax - qmin)
                    zp = (-w_min / scale).round().clamp(qmin, qmax)
                    quantized = (flat_w / scale + zp).round().clamp(qmin, qmax)
                    dequantized = (quantized - zp) * scale
                    module.weight.data = dequantized.reshape(weight.shape).to(module.weight.dtype)
                else:
                    module.weight.data = self._quantize_tensor(weight, qmin, qmax).to(module.weight.dtype)
            else:
                module.weight.data = self._quantize_tensor(weight, qmin, qmax).to(module.weight.dtype)

    @staticmethod
    def _quantize_tensor(tensor: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
        """Quantize and dequantize a tensor (per-tensor)."""
        t_min = tensor.min()
        t_max = tensor.max()
        scale = (t_max - t_min).clamp(min=1e-8) / (qmax - qmin)
        zp = (-t_min / scale).round().clamp(qmin, qmax)
        quantized = (tensor / scale + zp).round().clamp(qmin, qmax)
        return (quantized - zp) * scale

    @property
    def smooth_scales(self) -> Dict[str, torch.Tensor]:
        """Return computed smoothing scales per layer."""
        return self._smooth_scales

    def __repr__(self) -> str:
        return (
            f"SmoothQuantizer(alpha={self.alpha}, bits_w={self.bits_weight}, "
            f"bits_a={self.bits_activation}, per_channel={self.per_channel})"
        )
