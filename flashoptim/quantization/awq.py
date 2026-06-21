"""AWQ: Activation-Aware Weight Quantization.

Identifies salient weight channels via activation magnitudes and applies
per-channel scaling to protect important weights before quantization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashoptim.registry import QUANTIZERS


@QUANTIZERS.register("awq")
class AWQQuantizer:
    """Activation-Aware Weight Quantization.

    Protects salient weights by applying learned per-channel scaling factors
    derived from activation magnitudes, enabling low-bit quantization with
    minimal accuracy loss.

    Args:
        bits: Target bit-width (3, 4, or 8).
        group_size: Group size for quantization (-1 for per-channel).
        scale_search_steps: Number of grid search steps for optimal scale.
        scale_range: Range of scaling factors to search (min, max).
        calibration_samples: Number of calibration samples.
        auto_scale: Automatically compute per-channel scaling.
        zero_point: Whether to use asymmetric (zero-point) quantization.

    Example:
        >>> awq = AWQQuantizer(bits=4, group_size=128)
        >>> quantized_model = awq.quantize(model, calibration_loader)
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        scale_search_steps: int = 20,
        scale_range: Tuple[float, float] = (0.0, 1.0),
        calibration_samples: int = 128,
        auto_scale: bool = True,
        zero_point: bool = True,
    ) -> None:
        if bits not in (3, 4, 8):
            raise ValueError(f"Unsupported bit-width: {bits}. Options: 3, 4, 8")
        self.bits = bits
        self.group_size = group_size
        self.scale_search_steps = scale_search_steps
        self.scale_range = scale_range
        self.calibration_samples = calibration_samples
        self.auto_scale = auto_scale
        self.zero_point = zero_point
        self._scales: Dict[str, torch.Tensor] = {}

    def quantize(
        self,
        model: nn.Module,
        calibration_loader: Any,
    ) -> nn.Module:
        """Quantize model using AWQ algorithm.

        Args:
            model: Model to quantize.
            calibration_loader: DataLoader for activation statistics.

        Returns:
            Quantized model.
        """
        model.eval()

        activation_stats = self._collect_activation_stats(model, calibration_loader)

        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue

            act_scale = activation_stats.get(name)
            if act_scale is None:
                self._quantize_naive(module)
                continue

            if self.auto_scale:
                channel_scale = self._search_optimal_scale(module, act_scale)
                self._scales[name] = channel_scale
                self._apply_scale_and_quantize(module, channel_scale)
            else:
                self._quantize_naive(module)

        return model

    def _collect_activation_stats(
        self,
        model: nn.Module,
        dataloader: Any,
    ) -> Dict[str, torch.Tensor]:
        """Collect per-channel activation magnitude statistics."""
        stats: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                inp = input[0].detach()
                if inp.dim() > 2:
                    inp = inp.reshape(-1, inp.shape[-1])
                channel_magnitude = inp.abs().mean(dim=0)
                if name not in stats:
                    stats[name] = []
                stats[name].append(channel_magnitude)

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

        activation_scales = {}
        for name, magnitudes in stats.items():
            activation_scales[name] = torch.stack(magnitudes).mean(dim=0)

        return activation_scales

    def _search_optimal_scale(
        self,
        layer: nn.Module,
        activation_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Search for optimal per-channel scaling factor.

        Grid search over scaling factors to minimize quantization error
        weighted by activation importance.
        """
        if isinstance(layer, nn.Linear):
            weight = layer.weight.data.clone().float()
        elif isinstance(layer, nn.Conv2d):
            weight = layer.weight.data.clone().float().flatten(1)
        else:
            return torch.ones(1)

        n_out, n_in = weight.shape

        act_scale = (
            activation_scale[:n_in]
            if activation_scale.shape[0] >= n_in
            else (F.pad(activation_scale, (0, n_in - activation_scale.shape[0])))
        )
        act_scale = act_scale.clamp(min=1e-8)

        best_scale = torch.ones(n_in, device=weight.device)
        best_error = float("inf")

        for step in range(self.scale_search_steps):
            ratio = step / max(1, self.scale_search_steps - 1)
            alpha = self.scale_range[0] + ratio * (self.scale_range[1] - self.scale_range[0])

            scale = act_scale.pow(alpha)
            scale = scale / scale.mean()
            scale = scale.clamp(min=1e-4)

            scaled_weight = weight * scale.unsqueeze(0)
            q_weight = self._pseudo_quantize(scaled_weight)
            deq_weight = q_weight / scale.unsqueeze(0)

            error = ((weight - deq_weight).pow(2) * act_scale.unsqueeze(0)).sum().item()

            if error < best_error:
                best_error = error
                best_scale = scale.clone()

        return best_scale

    def _apply_scale_and_quantize(
        self,
        layer: nn.Module,
        scale: torch.Tensor,
    ) -> None:
        """Apply channel scaling and then quantize."""
        if isinstance(layer, nn.Linear):
            weight = layer.weight.data.float()
            n_in = weight.shape[1]
            s = scale[:n_in] if scale.shape[0] >= n_in else F.pad(scale, (0, n_in - scale.shape[0]), value=1.0)
            scaled_weight = weight * s.unsqueeze(0)
            q_weight = self._pseudo_quantize(scaled_weight)
            layer.weight.data = (q_weight / s.unsqueeze(0)).to(layer.weight.dtype)
        elif isinstance(layer, nn.Conv2d):
            weight = layer.weight.data.float()
            orig_shape = weight.shape
            weight_flat = weight.flatten(1)
            n_in = weight_flat.shape[1]
            s = scale[:n_in] if scale.shape[0] >= n_in else F.pad(scale, (0, n_in - scale.shape[0]), value=1.0)
            scaled = weight_flat * s.unsqueeze(0)
            q = self._pseudo_quantize(scaled)
            layer.weight.data = (q / s.unsqueeze(0)).reshape(orig_shape).to(layer.weight.dtype)

    def _quantize_naive(self, layer: nn.Module) -> None:
        """Simple round-to-nearest quantization without scaling."""
        if isinstance(layer, nn.Linear):
            layer.weight.data = self._pseudo_quantize(layer.weight.data.float()).to(layer.weight.dtype)
        elif isinstance(layer, nn.Conv2d):
            orig_shape = layer.weight.shape
            flat = layer.weight.data.float().flatten(1)
            layer.weight.data = self._pseudo_quantize(flat).reshape(orig_shape).to(layer.weight.dtype)

    def _pseudo_quantize(self, weight: torch.Tensor) -> torch.Tensor:
        """Pseudo-quantize (quantize + dequantize) weight tensor."""
        qmin = 0
        qmax = (1 << self.bits) - 1

        if self.group_size > 0 and weight.shape[-1] > self.group_size:
            n_cols = weight.shape[-1]
            result = torch.zeros_like(weight)
            for g_start in range(0, n_cols, self.group_size):
                g_end = min(g_start + self.group_size, n_cols)
                group = weight[:, g_start:g_end]
                result[:, g_start:g_end] = self._quantize_group(group, qmin, qmax)
            return result
        else:
            return self._quantize_group(weight, qmin, qmax)

    def _quantize_group(self, weight: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
        """Quantize and dequantize a weight group."""
        if self.zero_point:
            w_min = weight.min(dim=-1, keepdim=True).values
            w_max = weight.max(dim=-1, keepdim=True).values
            scale = (w_max - w_min).clamp(min=1e-8) / (qmax - qmin)
            zp = (-w_min / scale).round().clamp(qmin, qmax)
            quantized = (weight / scale + zp).round().clamp(qmin, qmax)
            return (quantized - zp) * scale
        else:
            abs_max = weight.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
            scale = abs_max / ((qmax - qmin) / 2)
            mid = (qmax + qmin) / 2
            quantized = (weight / scale + mid).round().clamp(qmin, qmax)
            return (quantized - mid) * scale

    @property
    def scales(self) -> Dict[str, torch.Tensor]:
        """Return learned per-channel scales."""
        return self._scales

    def __repr__(self) -> str:
        return f"AWQQuantizer(bits={self.bits}, group_size={self.group_size}, auto_scale={self.auto_scale})"
