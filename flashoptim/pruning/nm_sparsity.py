"""N:M Structured Sparsity for hardware-accelerated sparse inference.

Implements N:M sparsity patterns (e.g., 2:4) compatible with NVIDIA
Ampere+ Sparse Tensor Cores for 2x throughput on supported hardware.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from flashoptim.registry import PRUNERS


@PRUNERS.register("nm_sparsity")
class NMSparsityPruner:
    """N:M Structured Sparsity pruner.

    Enforces that exactly N out of every M consecutive weights are zero,
    enabling hardware acceleration on NVIDIA Ampere+ GPUs with Sparse
    Tensor Cores (2:4 sparsity → 2x inference speedup).

    Args:
        n: Number of zeros per group (e.g., 2 in 2:4).
        m: Group size (e.g., 4 in 2:4).
        criterion: Weight selection criterion ('magnitude', 'wanda', 'gradient').
        permute_columns: Try column permutations for better accuracy.
        calibration_samples: Samples for activation-based criteria.

    Example:
        >>> pruner = NMSparsityPruner(n=2, m=4)
        >>> sparse_model = pruner.prune(model)
    """

    CRITERIA = ("magnitude", "wanda", "gradient")
    VALID_PATTERNS = ((1, 2), (2, 4), (4, 8), (1, 4), (3, 4))

    def __init__(
        self,
        n: int = 2,
        m: int = 4,
        criterion: str = "magnitude",
        permute_columns: bool = False,
        calibration_samples: int = 128,
    ) -> None:
        if criterion not in self.CRITERIA:
            raise ValueError(f"Unknown criterion: {criterion}. Options: {self.CRITERIA}")
        if n >= m:
            raise ValueError(f"N must be < M, got N={n}, M={m}")
        self.n = n
        self.m = m
        self.criterion = criterion
        self.permute_columns = permute_columns
        self.calibration_samples = calibration_samples
        self._masks: Dict[str, torch.Tensor] = {}
        self._permutations: Dict[str, torch.Tensor] = {}

    @property
    def sparsity(self) -> float:
        """Effective sparsity ratio."""
        return self.n / self.m

    def prune(
        self,
        model: nn.Module,
        calibration_loader: Optional[Any] = None,
    ) -> nn.Module:
        """Apply N:M structured sparsity to the model.

        Args:
            model: Model to prune.
            calibration_loader: Required for 'wanda' criterion.

        Returns:
            Model with N:M sparsity pattern applied.
        """
        model.eval()

        if self.criterion == "wanda" and calibration_loader is not None:
            activation_norms = self._collect_activation_norms(model, calibration_loader)
        else:
            activation_norms = {}

        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue

            act_norm = activation_norms.get(name)
            self._prune_layer(module, name, act_norm)

        return model

    def _prune_layer(
        self,
        layer: nn.Module,
        layer_name: str,
        activation_norm: Optional[torch.Tensor] = None,
    ) -> None:
        """Apply N:M pattern to a single layer."""
        if isinstance(layer, nn.Linear):
            weight = layer.weight.data.float()
            is_conv = False
        elif isinstance(layer, nn.Conv2d):
            weight = layer.weight.data.float().flatten(1)
            is_conv = True
        else:
            return

        n_rows, n_cols = weight.shape

        scores = self._compute_scores(weight, activation_norm)

        if self.permute_columns:
            perm = self._find_best_permutation(scores)
            self._permutations[layer_name] = perm
            weight = weight[:, perm]
            scores = scores[:, perm]

        mask = self._apply_nm_pattern(scores)

        weight[~mask] = 0.0

        if self.permute_columns:
            inv_perm = torch.argsort(perm)
            weight = weight[:, inv_perm]
            mask.clone()
            mask = mask[:, inv_perm]

        if is_conv:
            layer.weight.data = weight.reshape(layer.weight.shape).to(layer.weight.dtype)
        else:
            layer.weight.data = weight.to(layer.weight.dtype)

        self._masks[layer_name] = mask

    def _compute_scores(
        self,
        weight: torch.Tensor,
        activation_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute importance scores for weight selection."""
        if self.criterion == "magnitude":
            return weight.abs()
        elif self.criterion == "wanda" and activation_norm is not None:
            n_cols = weight.shape[1]
            an = (
                activation_norm[:n_cols]
                if activation_norm.shape[0] >= n_cols
                else (torch.nn.functional.pad(activation_norm, (0, n_cols - activation_norm.shape[0]), value=1.0))
            )
            return weight.abs() * an.unsqueeze(0)
        else:
            return weight.abs()

    def _apply_nm_pattern(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply N:M sparsity pattern based on scores.

        For each group of M consecutive elements, keep the top (M-N)
        and prune the bottom N.
        """
        n_rows, n_cols = scores.shape
        mask = torch.ones_like(scores, dtype=torch.bool)

        n_groups = n_cols // self.m
        remainder = n_cols % self.m

        for g in range(n_groups):
            start = g * self.m
            end = start + self.m
            group_scores = scores[:, start:end]
            _, keep_indices = torch.topk(group_scores, k=self.m - self.n, dim=1)
            group_mask = torch.zeros(n_rows, self.m, dtype=torch.bool, device=scores.device)
            group_mask.scatter_(1, keep_indices, True)
            mask[:, start:end] = group_mask

        if remainder > 0:
            start = n_groups * self.m
            n_prune_remainder = max(0, int(remainder * self.n / self.m))
            if n_prune_remainder > 0:
                rem_scores = scores[:, start:]
                _, prune_indices = torch.topk(rem_scores, k=n_prune_remainder, dim=1, largest=False)
                rem_mask = torch.ones(n_rows, remainder, dtype=torch.bool, device=scores.device)
                rem_mask.scatter_(1, prune_indices, False)
                mask[:, start:] = rem_mask

        return mask

    def _find_best_permutation(self, scores: torch.Tensor) -> torch.Tensor:
        """Find a column permutation that improves the N:M pattern quality.

        Uses a greedy heuristic to group similar-importance columns together.
        """
        scores.shape[1]
        col_importance = scores.sum(dim=0)
        perm = torch.argsort(col_importance, descending=True)
        return perm

    def _collect_activation_norms(
        self,
        model: nn.Module,
        dataloader: Any,
    ) -> Dict[str, torch.Tensor]:
        """Collect activation L2 norms for Wanda criterion."""
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

        for name in norms:
            norms[name] = norms[name] / max(counts[name], 1)

        return norms

    def verify_pattern(self, model: nn.Module) -> Dict[str, bool]:
        """Verify that all layers satisfy the N:M sparsity constraint."""
        results = {}
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue

            if isinstance(module, nn.Linear):
                weight = module.weight.data
            else:
                weight = module.weight.data.flatten(1)

            is_valid = self._check_nm_constraint(weight)
            results[name] = is_valid

        return results

    def _check_nm_constraint(self, weight: torch.Tensor) -> bool:
        """Check if weight satisfies N:M constraint."""
        n_rows, n_cols = weight.shape
        n_groups = n_cols // self.m

        for g in range(n_groups):
            start = g * self.m
            end = start + self.m
            group = weight[:, start:end]
            zeros_per_row = (group == 0).sum(dim=1)
            if not (zeros_per_row >= self.n).all():
                return False

        return True

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
    def masks(self) -> Dict[str, torch.Tensor]:
        """Return computed pruning masks."""
        return self._masks

    def __repr__(self) -> str:
        return f"NMSparsityPruner(N={self.n}, M={self.m}, criterion='{self.criterion}', sparsity={self.sparsity:.2f})"
