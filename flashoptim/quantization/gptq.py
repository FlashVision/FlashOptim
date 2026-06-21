"""GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.

Layer-wise quantization using the Optimal Brain Surgeon (OBS) framework with
Hessian-based weight updates for minimal output perturbation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from flashoptim.registry import QUANTIZERS


@QUANTIZERS.register("gptq")
class GPTQQuantizer:
    """GPTQ quantization using Hessian-based optimal weight rounding.

    Quantizes weights layer-by-layer, using second-order (Hessian) information
    to optimally adjust remaining weights after each column quantization.

    Args:
        bits: Target bit-width (2, 3, 4, or 8).
        group_size: Group size for group quantization (-1 for per-channel).
        block_size: Number of columns to process simultaneously.
        damp_percent: Dampening percentage for Hessian diagonal.
        symmetric: Use symmetric quantization.
        act_order: Process columns in order of decreasing activation magnitude.
        calibration_samples: Number of calibration samples to use.

    Example:
        >>> gptq = GPTQQuantizer(bits=4, group_size=128)
        >>> quantized_model = gptq.quantize(model, calibration_loader)
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        block_size: int = 128,
        damp_percent: float = 0.01,
        symmetric: bool = False,
        act_order: bool = True,
        calibration_samples: int = 128,
    ) -> None:
        if bits not in (2, 3, 4, 8):
            raise ValueError(f"Unsupported bit-width: {bits}. Options: 2, 3, 4, 8")
        self.bits = bits
        self.group_size = group_size
        self.block_size = block_size
        self.damp_percent = damp_percent
        self.symmetric = symmetric
        self.act_order = act_order
        self.calibration_samples = calibration_samples
        self._quantization_stats: Dict[str, Dict[str, float]] = {}

    def quantize(
        self,
        model: nn.Module,
        calibration_loader: Any,
    ) -> nn.Module:
        """Quantize model using GPTQ algorithm.

        Args:
            model: Model to quantize.
            calibration_loader: DataLoader for calibration data.

        Returns:
            Quantized model with reduced precision weights.
        """
        model.eval()
        layers_to_quantize = self._find_linear_layers(model)

        hessians = self._collect_hessians(model, calibration_loader, layers_to_quantize)

        for name, module in layers_to_quantize.items():
            H = hessians.get(name)
            if H is None:
                continue
            self._quantize_layer(module, H, name)

        return model

    def _find_linear_layers(self, model: nn.Module) -> Dict[str, nn.Linear]:
        """Find all linear layers eligible for quantization."""
        layers = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layers[name] = module
            elif isinstance(module, nn.Conv2d):
                layers[name] = module
        return layers

    def _collect_hessians(
        self,
        model: nn.Module,
        dataloader: Any,
        layers: Dict[str, nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Collect Hessian approximations (X^T X) for each layer.

        Uses calibration data to accumulate input activation outer products.
        """
        hessians: Dict[str, torch.Tensor] = {}
        hooks = []
        activations: Dict[str, List[torch.Tensor]] = {name: [] for name in layers}

        def make_hook(name):
            def hook_fn(module, input, output):
                inp = input[0].detach()
                if inp.dim() > 2:
                    inp = inp.reshape(-1, inp.shape[-1])
                activations[name].append(inp)
            return hook_fn

        for name, module in layers.items():
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

        for name, acts in activations.items():
            if not acts:
                continue
            X = torch.cat(acts, dim=0)
            n_samples = X.shape[0]
            H = (X.T @ X) / n_samples
            hessians[name] = H

        return hessians

    def _quantize_layer(
        self,
        layer: nn.Module,
        hessian: torch.Tensor,
        layer_name: str,
    ) -> None:
        """Quantize a single layer using the GPTQ algorithm.

        Processes columns in blocks, using Hessian inverse to optimally
        update remaining weights after quantizing each column.
        """
        if isinstance(layer, nn.Linear):
            W = layer.weight.data.clone().float()
        elif isinstance(layer, nn.Conv2d):
            W = layer.weight.data.clone().float().flatten(1)
        else:
            return

        n_rows, n_cols = W.shape
        H = hessian.clone()

        damp = self.damp_percent * torch.diag(H).mean()
        H.diagonal().add_(damp)

        try:
            H_inv = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(H_inv)
        except RuntimeError:
            H_inv = torch.linalg.inv(H + damp * torch.eye(H.shape[0], device=H.device))

        if self.act_order:
            perm = torch.argsort(torch.diag(H_inv))
            W = W[:, perm]
            H_inv = H_inv[perm][:, perm]

        Q = torch.zeros_like(W)
        total_error = 0.0

        for col_start in range(0, n_cols, self.block_size):
            col_end = min(col_start + self.block_size, n_cols)
            block_cols = col_end - col_start

            W_block = W[:, col_start:col_end].clone()
            H_block_inv = H_inv[col_start:col_end, col_start:col_end]

            for j in range(block_cols):
                col_idx = col_start + j
                w_col = W_block[:, j]

                group_idx = col_idx // self.group_size if self.group_size > 0 else 0
                scale, zero_point = self._compute_quantization_params(
                    W[:, max(0, group_idx * self.group_size):min(n_cols, (group_idx + 1) * self.group_size)]
                    if self.group_size > 0 else W
                )

                q_col = self._quantize_weight(w_col, scale, zero_point)
                Q[:, col_idx] = q_col

                error = (w_col - q_col) / H_block_inv[j, j].clamp(min=1e-6)
                total_error += (w_col - q_col).pow(2).sum().item()

                if j + 1 < block_cols:
                    W_block[:, j + 1:] -= error.unsqueeze(1) * H_block_inv[j, j + 1:block_cols].unsqueeze(0)

        if self.act_order:
            inv_perm = torch.argsort(perm)
            Q = Q[:, inv_perm]

        if isinstance(layer, nn.Linear):
            layer.weight.data = Q.to(layer.weight.dtype)
        elif isinstance(layer, nn.Conv2d):
            layer.weight.data = Q.reshape(layer.weight.shape).to(layer.weight.dtype)

        self._quantization_stats[layer_name] = {
            "total_error": total_error,
            "mean_error": total_error / (n_rows * n_cols),
            "bits": self.bits,
        }

    def _compute_quantization_params(
        self, weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scale and zero-point for quantization."""
        qmin = 0
        qmax = (1 << self.bits) - 1

        if self.symmetric:
            abs_max = weight.abs().max(dim=-1).values.clamp(min=1e-8)
            scale = abs_max / ((qmax - qmin) / 2)
            zero_point = torch.zeros_like(scale) + (qmax + qmin) / 2
        else:
            w_min = weight.min(dim=-1).values
            w_max = weight.max(dim=-1).values
            scale = (w_max - w_min).clamp(min=1e-8) / (qmax - qmin)
            zero_point = (-w_min / scale).round().clamp(qmin, qmax)

        return scale, zero_point

    def _quantize_weight(
        self, weight: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        """Quantize and dequantize a weight tensor."""
        qmin = 0
        qmax = (1 << self.bits) - 1

        if scale.dim() == 0:
            s, z = scale, zero_point
        else:
            s = scale.mean()
            z = zero_point.mean()

        quantized = (weight / s + z).round().clamp(qmin, qmax)
        dequantized = (quantized - z) * s
        return dequantized

    @property
    def stats(self) -> Dict[str, Dict[str, float]]:
        """Return per-layer quantization statistics."""
        return self._quantization_stats

    def __repr__(self) -> str:
        return (
            f"GPTQQuantizer(bits={self.bits}, group_size={self.group_size}, "
            f"block_size={self.block_size}, act_order={self.act_order})"
        )
