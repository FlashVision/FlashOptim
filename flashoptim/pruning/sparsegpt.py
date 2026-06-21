"""SparseGPT: One-shot unstructured pruning using approximate sparse regression.

Prunes weights to target sparsity in a single pass by solving a sparse
reconstruction problem using Hessian information, similar to OBS/OBD.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from flashoptim.registry import PRUNERS


@PRUNERS.register("sparsegpt")
class SparseGPTPruner:
    """SparseGPT: one-shot pruning via approximate sparse regression.

    Uses Hessian information to optimally prune and update remaining weights
    to minimize output reconstruction error, achieving high sparsity
    without retraining.

    Args:
        sparsity: Target sparsity ratio (0.0 to 1.0).
        block_size: Column block size for processing.
        damp_percent: Dampening for Hessian diagonal.
        prunen: N in N:M structured pattern (0 for unstructured).
        prunem: M in N:M structured pattern (0 for unstructured).
        calibration_samples: Number of calibration samples.

    Example:
        >>> pruner = SparseGPTPruner(sparsity=0.5)
        >>> pruned_model = pruner.prune(model, calibration_loader)
    """

    def __init__(
        self,
        sparsity: float = 0.5,
        block_size: int = 128,
        damp_percent: float = 0.01,
        prunen: int = 0,
        prunem: int = 0,
        calibration_samples: int = 128,
    ) -> None:
        if not 0.0 < sparsity < 1.0:
            raise ValueError(f"Sparsity must be in (0, 1), got {sparsity}")
        self.sparsity = sparsity
        self.block_size = block_size
        self.damp_percent = damp_percent
        self.prunen = prunen
        self.prunem = prunem
        self.calibration_samples = calibration_samples
        self._pruning_stats: Dict[str, Dict[str, float]] = {}

    def prune(
        self,
        model: nn.Module,
        calibration_loader: Any,
    ) -> nn.Module:
        """Prune model using SparseGPT algorithm.

        Args:
            model: Model to prune.
            calibration_loader: DataLoader for Hessian estimation.

        Returns:
            Pruned model.
        """
        model.eval()
        layers = self._find_prunable_layers(model)
        hessians = self._collect_hessians(model, calibration_loader, layers)

        for name, module in layers.items():
            H = hessians.get(name)
            if H is None:
                continue
            self._prune_layer(module, H, name)

        return model

    def _find_prunable_layers(self, model: nn.Module) -> Dict[str, nn.Module]:
        """Find layers eligible for pruning."""
        layers = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers[name] = module
        return layers

    def _collect_hessians(
        self,
        model: nn.Module,
        dataloader: Any,
        layers: Dict[str, nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Collect Hessian approximations (X^T X) for each layer."""
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
            X = torch.cat(acts, dim=0).float()
            n_samples = X.shape[0]
            H = (X.T @ X) / n_samples
            hessians[name] = H

        return hessians

    def _prune_layer(
        self,
        layer: nn.Module,
        hessian: torch.Tensor,
        layer_name: str,
    ) -> None:
        """Prune a single layer using SparseGPT algorithm.

        For each column block:
        1. Find the smallest-magnitude weights to prune
        2. Compute optimal weight updates for unpruned weights
        3. Apply pruning mask and weight corrections
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

        mask = torch.ones_like(W, dtype=torch.bool)

        for col_start in range(0, n_cols, self.block_size):
            col_end = min(col_start + self.block_size, n_cols)
            block_cols = col_end - col_start

            W_block = W[:, col_start:col_end].clone()
            H_block = H_inv[col_start:col_end, col_start:col_end]
            H_diag = torch.diag(H_block).clamp(min=1e-8)

            if self.prunen > 0 and self.prunem > 0:
                block_mask = self._nm_prune_block(W_block, self.prunen, self.prunem)
            else:
                scores = W_block.pow(2) / H_diag.unsqueeze(0)
                n_prune = int(block_cols * self.sparsity)
                _, prune_idx = torch.topk(scores.flatten(), n_prune, largest=False)
                block_mask = torch.ones_like(W_block, dtype=torch.bool)
                flat_mask = block_mask.flatten()
                flat_mask[prune_idx] = False
                block_mask = flat_mask.reshape(W_block.shape)

            pruned_weights = W_block * (~block_mask).float()
            errors = pruned_weights / H_diag.unsqueeze(0).clamp(min=1e-8)

            W_block[~block_mask] = 0.0

            for j in range(block_cols):
                col_mask = block_mask[:, j]
                if not col_mask.all():
                    error_col = errors[:, j]
                    if j + 1 < block_cols:
                        update = error_col.unsqueeze(1) * H_block[j, j + 1:block_cols].unsqueeze(0)
                        W_block[:, j + 1:] -= update * block_mask[:, j + 1:].float()

            W[:, col_start:col_end] = W_block
            mask[:, col_start:col_end] = block_mask

        W[~mask] = 0.0

        if isinstance(layer, nn.Linear):
            layer.weight.data = W.to(layer.weight.dtype)
        elif isinstance(layer, nn.Conv2d):
            layer.weight.data = W.reshape(layer.weight.shape).to(layer.weight.dtype)

        actual_sparsity = (~mask).float().mean().item()
        self._pruning_stats[layer_name] = {
            "target_sparsity": self.sparsity,
            "actual_sparsity": actual_sparsity,
            "n_pruned": (~mask).sum().item(),
            "n_total": mask.numel(),
        }

    @staticmethod
    def _nm_prune_block(
        weight: torch.Tensor, n: int, m: int
    ) -> torch.Tensor:
        """Apply N:M structured sparsity pattern to a block."""
        n_rows, n_cols = weight.shape
        mask = torch.ones_like(weight, dtype=torch.bool)

        for col in range(0, n_cols - m + 1, m):
            group = weight[:, col:col + m]
            _, indices = torch.topk(group.abs(), k=m - n, dim=1, largest=True)
            group_mask = torch.zeros_like(group, dtype=torch.bool)
            group_mask.scatter_(1, indices, True)
            mask[:, col:col + m] = group_mask

        return mask

    @staticmethod
    def get_sparsity(model: nn.Module) -> Dict[str, float]:
        """Compute per-layer and global sparsity."""
        total_zeros = 0
        total_elements = 0
        layer_sparsity = {}

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                zeros = (weight == 0).sum().item()
                elements = weight.numel()
                total_zeros += zeros
                total_elements += elements
                layer_sparsity[name] = zeros / elements

        layer_sparsity["global"] = total_zeros / total_elements if total_elements > 0 else 0.0
        return layer_sparsity

    @property
    def stats(self) -> Dict[str, Dict[str, float]]:
        """Return per-layer pruning statistics."""
        return self._pruning_stats

    def __repr__(self) -> str:
        nm_str = f", N:M={self.prunen}:{self.prunem}" if self.prunen > 0 else ""
        return (
            f"SparseGPTPruner(sparsity={self.sparsity}, "
            f"block_size={self.block_size}{nm_str})"
        )
