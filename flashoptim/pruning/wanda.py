"""Wanda: Pruning by Weights AND Activations.

A simple yet effective pruning method that scores weights by the product
of weight magnitude and input activation norm, requiring only a single
forward pass for calibration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from flashoptim.registry import PRUNERS


@PRUNERS.register("wanda")
class WandaPruner:
    """Wanda: pruning by Weights AND Activations.

    Scores each weight by |w_ij| * ||X_j||_2, where X_j is the
    j-th column of the input activation matrix. Prunes weights
    with smallest scores.

    Args:
        sparsity: Target sparsity ratio (0.0 to 1.0).
        pruning_scope: 'global' for global threshold, 'layer' for per-layer,
            'row' for per-output-channel pruning.
        calibration_samples: Number of calibration samples.
        use_nm: If True, apply N:M structured sparsity pattern.
        n: N in N:M sparsity (number of zeros per group of M).
        m: M in N:M sparsity (group size).

    Example:
        >>> pruner = WandaPruner(sparsity=0.5, pruning_scope="row")
        >>> pruned_model = pruner.prune(model, calibration_loader)
    """

    SCOPES = ("global", "layer", "row")

    def __init__(
        self,
        sparsity: float = 0.5,
        pruning_scope: str = "row",
        calibration_samples: int = 128,
        use_nm: bool = False,
        n: int = 2,
        m: int = 4,
    ) -> None:
        if not 0.0 < sparsity < 1.0:
            raise ValueError(f"Sparsity must be in (0, 1), got {sparsity}")
        if pruning_scope not in self.SCOPES:
            raise ValueError(f"Unknown scope: {pruning_scope}. Options: {self.SCOPES}")
        self.sparsity = sparsity
        self.pruning_scope = pruning_scope
        self.calibration_samples = calibration_samples
        self.use_nm = use_nm
        self.n = n
        self.m = m
        self._pruning_stats: Dict[str, Dict[str, float]] = {}

    def prune(
        self,
        model: nn.Module,
        calibration_loader: Any,
    ) -> nn.Module:
        """Prune model using Wanda scoring.

        Args:
            model: Model to prune.
            calibration_loader: DataLoader for activation collection.

        Returns:
            Pruned model with zeroed-out weights.
        """
        model.eval()
        layers = self._find_prunable_layers(model)
        activation_norms = self._collect_activation_norms(model, calibration_loader, layers)

        if self.pruning_scope == "global":
            self._prune_global(layers, activation_norms)
        else:
            for name, module in layers.items():
                act_norm = activation_norms.get(name)
                if act_norm is None:
                    continue
                self._prune_layer(module, act_norm, name)

        return model

    def _find_prunable_layers(self, model: nn.Module) -> Dict[str, nn.Module]:
        """Find layers eligible for pruning."""
        layers = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers[name] = module
        return layers

    def _collect_activation_norms(
        self,
        model: nn.Module,
        dataloader: Any,
        layers: Dict[str, nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Collect L2 norms of input activations per channel."""
        norms: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                inp = input[0].detach()
                if inp.dim() > 2:
                    inp = inp.reshape(-1, inp.shape[-1])
                col_norm = inp.norm(p=2, dim=0)
                if name in norms:
                    norms[name] += col_norm
                    counts[name] += 1
                else:
                    norms[name] = col_norm
                    counts[name] = 1
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

        for name in norms:
            norms[name] = norms[name] / max(counts[name], 1)

        return norms

    def _prune_layer(
        self,
        layer: nn.Module,
        activation_norm: torch.Tensor,
        layer_name: str,
    ) -> None:
        """Prune a single layer using Wanda scores."""
        if isinstance(layer, nn.Linear):
            weight = layer.weight.data.float()
        elif isinstance(layer, nn.Conv2d):
            weight = layer.weight.data.float().flatten(1)
        else:
            return

        n_rows, n_cols = weight.shape
        act_norm = activation_norm[:n_cols] if activation_norm.shape[0] >= n_cols else (
            torch.nn.functional.pad(activation_norm, (0, n_cols - activation_norm.shape[0]), value=1.0)
        )

        scores = weight.abs() * act_norm.unsqueeze(0)

        if self.use_nm:
            mask = self._nm_prune(scores, self.n, self.m)
        elif self.pruning_scope == "row":
            mask = self._prune_by_row(scores)
        else:
            n_prune = int(n_rows * n_cols * self.sparsity)
            _, prune_idx = torch.topk(scores.flatten(), n_prune, largest=False)
            mask = torch.ones_like(weight, dtype=torch.bool)
            flat_mask = mask.flatten()
            flat_mask[prune_idx] = False
            mask = flat_mask.reshape(weight.shape)

        weight[~mask] = 0.0

        if isinstance(layer, nn.Linear):
            layer.weight.data = weight.to(layer.weight.dtype)
        elif isinstance(layer, nn.Conv2d):
            layer.weight.data = weight.reshape(layer.weight.shape).to(layer.weight.dtype)

        actual_sparsity = (~mask).float().mean().item()
        self._pruning_stats[layer_name] = {
            "target_sparsity": self.sparsity,
            "actual_sparsity": actual_sparsity,
            "n_pruned": (~mask).sum().item(),
            "n_total": mask.numel(),
        }

    def _prune_by_row(self, scores: torch.Tensor) -> torch.Tensor:
        """Per-row (per-output-channel) pruning."""
        n_rows, n_cols = scores.shape
        n_prune_per_row = int(n_cols * self.sparsity)
        mask = torch.ones_like(scores, dtype=torch.bool)

        for i in range(n_rows):
            _, prune_idx = torch.topk(scores[i], n_prune_per_row, largest=False)
            mask[i, prune_idx] = False

        return mask

    def _prune_global(
        self,
        layers: Dict[str, nn.Module],
        activation_norms: Dict[str, torch.Tensor],
    ) -> None:
        """Global pruning across all layers."""
        all_scores = []
        layer_info = []

        for name, module in layers.items():
            act_norm = activation_norms.get(name)
            if act_norm is None:
                continue

            if isinstance(module, nn.Linear):
                weight = module.weight.data.float()
            elif isinstance(module, nn.Conv2d):
                weight = module.weight.data.float().flatten(1)
            else:
                continue

            n_cols = weight.shape[1]
            an = act_norm[:n_cols] if act_norm.shape[0] >= n_cols else (
                torch.nn.functional.pad(act_norm, (0, n_cols - act_norm.shape[0]), value=1.0)
            )
            scores = (weight.abs() * an.unsqueeze(0)).flatten()
            all_scores.append(scores)
            layer_info.append((name, module, weight.shape))

        if not all_scores:
            return

        all_scores_cat = torch.cat(all_scores)
        n_total = all_scores_cat.numel()
        n_prune = int(n_total * self.sparsity)
        threshold = torch.topk(all_scores_cat, n_prune, largest=False).values[-1]

        offset = 0
        for name, module, shape in layer_info:
            n_elements = shape[0] * shape[1]
            layer_scores = all_scores_cat[offset:offset + n_elements].reshape(shape)
            mask = layer_scores > threshold

            if isinstance(module, nn.Linear):
                module.weight.data[~mask] = 0.0
            elif isinstance(module, nn.Conv2d):
                flat_w = module.weight.data.float().flatten(1)
                flat_w[~mask] = 0.0
                module.weight.data = flat_w.reshape(module.weight.shape).to(module.weight.dtype)

            self._pruning_stats[name] = {
                "actual_sparsity": (~mask).float().mean().item(),
            }
            offset += n_elements

    @staticmethod
    def _nm_prune(scores: torch.Tensor, n: int, m: int) -> torch.Tensor:
        """Apply N:M structured sparsity based on Wanda scores."""
        n_rows, n_cols = scores.shape
        mask = torch.ones_like(scores, dtype=torch.bool)

        for col in range(0, n_cols - m + 1, m):
            group = scores[:, col:col + m]
            _, keep_idx = torch.topk(group, k=m - n, dim=1, largest=True)
            group_mask = torch.zeros_like(group, dtype=torch.bool)
            group_mask.scatter_(1, keep_idx, True)
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
        nm_str = f", N:M={self.n}:{self.m}" if self.use_nm else ""
        return (
            f"WandaPruner(sparsity={self.sparsity}, scope='{self.pruning_scope}'{nm_str})"
        )
