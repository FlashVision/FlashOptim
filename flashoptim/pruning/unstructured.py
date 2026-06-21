"""Unstructured (weight-level) pruning."""

from __future__ import annotations

from typing import Any, Dict, List

import torch.nn as nn
import torch.nn.utils.prune as prune_utils


class UnstructuredPruner:
    """Unstructured pruning — removes individual weights based on importance.

    Supports magnitude-based, random, and gradient-based pruning criteria.
    Can be applied iteratively with fine-tuning between rounds.

    Args:
        sparsity: Target sparsity ratio (0.0 to 1.0).
        method: Pruning criterion ('magnitude', 'random', 'gradient').
        iterative: Apply pruning iteratively over multiple rounds.
        iterations: Number of pruning iterations (if iterative).
        global_pruning: Apply global threshold vs per-layer threshold.
    """

    METHODS = ("magnitude", "random", "gradient")

    def __init__(
        self,
        sparsity: float = 0.5,
        method: str = "magnitude",
        iterative: bool = True,
        iterations: int = 3,
        global_pruning: bool = True,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Options: {self.METHODS}")
        if not 0.0 < sparsity < 1.0:
            raise ValueError(f"Sparsity must be in (0, 1), got {sparsity}")

        self.sparsity = sparsity
        self.method = method
        self.iterative = iterative
        self.iterations = iterations
        self.global_pruning = global_pruning

    def prune(self, model: nn.Module, **kwargs: Any) -> nn.Module:
        """Apply unstructured pruning to the model.

        Args:
            model: PyTorch model to prune.
            **kwargs: Additional pruning parameters.

        Returns:
            Pruned model with weight masks applied.
        """
        parameters_to_prune = self._get_prunable_layers(model)

        if not parameters_to_prune:
            return model

        if self.iterative:
            sparsity_per_iter = 1.0 - (1.0 - self.sparsity) ** (1.0 / self.iterations)
            for i in range(self.iterations):
                self._apply_pruning(model, parameters_to_prune, sparsity_per_iter)
        else:
            self._apply_pruning(model, parameters_to_prune, self.sparsity)

        return model

    def _get_prunable_layers(self, model: nn.Module) -> List[tuple]:
        """Get layers eligible for pruning."""
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers.append((module, "weight"))
        return layers

    def _apply_pruning(
        self,
        model: nn.Module,
        parameters: List[tuple],
        amount: float,
    ) -> None:
        """Apply pruning to specified parameters."""
        if self.method == "magnitude":
            if self.global_pruning:
                prune_utils.global_unstructured(parameters, pruning_method=prune_utils.L1Unstructured, amount=amount)
            else:
                for module, param_name in parameters:
                    prune_utils.l1_unstructured(module, param_name, amount=amount)
        elif self.method == "random":
            for module, param_name in parameters:
                prune_utils.random_unstructured(module, param_name, amount=amount)
        elif self.method == "gradient":
            raise NotImplementedError("Gradient-based pruning coming in v1.1.")

    @staticmethod
    def remove_pruning(model: nn.Module) -> nn.Module:
        """Make pruning permanent by removing masks and reparameterization.

        Args:
            model: Pruned model with masks.

        Returns:
            Model with pruning made permanent.
        """
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    prune_utils.remove(module, "weight")
                except ValueError:
                    pass
        return model

    @staticmethod
    def get_sparsity(model: nn.Module) -> Dict[str, float]:
        """Get per-layer and global sparsity statistics.

        Args:
            model: Model to analyze.

        Returns:
            Dictionary with 'global' sparsity and per-layer sparsities.
        """
        total_zeros = 0
        total_elements = 0
        layer_sparsity = {}

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data
                zeros = (weight == 0).sum().item()
                elements = weight.numel()
                total_zeros += zeros
                total_elements += elements
                layer_sparsity[name] = zeros / elements

        layer_sparsity["global"] = total_zeros / total_elements if total_elements > 0 else 0.0
        return layer_sparsity

    def __repr__(self) -> str:
        return (
            f"UnstructuredPruner(sparsity={self.sparsity}, method={self.method}, "
            f"iterative={self.iterative}, iterations={self.iterations})"
        )
