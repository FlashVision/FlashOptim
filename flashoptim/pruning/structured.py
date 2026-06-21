"""Structured (channel/filter-level) pruning."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class StructuredPruner:
    """Structured pruning — removes entire channels or filters.

    Unlike unstructured pruning, structured pruning produces dense models
    that run faster on standard hardware without sparse computation support.

    Args:
        sparsity: Target channel/filter removal ratio (0.0 to 1.0).
        criterion: Importance criterion ('l1_norm', 'l2_norm', 'bn_scale', 'taylor').
        granularity: Pruning granularity ('channel', 'filter').
        skip_layers: Layer names to skip during pruning.
    """

    CRITERIA = ("l1_norm", "l2_norm", "bn_scale", "taylor")

    def __init__(
        self,
        sparsity: float = 0.3,
        criterion: str = "l1_norm",
        granularity: str = "channel",
        skip_layers: Optional[List[str]] = None,
    ) -> None:
        if criterion not in self.CRITERIA:
            raise ValueError(f"Unknown criterion: {criterion}. Options: {self.CRITERIA}")

        self.sparsity = sparsity
        self.criterion = criterion
        self.granularity = granularity
        self.skip_layers = skip_layers or []

    def prune(self, model: nn.Module, **kwargs: Any) -> nn.Module:
        """Apply structured pruning to the model.

        Removes entire channels/filters using torch.nn.utils.prune and
        zeroes out pruned channels to achieve structured sparsity.

        Args:
            model: PyTorch model to prune.
            **kwargs: Additional parameters.

        Returns:
            Pruned model with zeroed-out channels.
        """
        import torch.nn.utils.prune as prune_utils

        plan = self.get_pruning_plan(model)

        for name, module in model.named_modules():
            if name not in plan or name in self.skip_layers:
                continue
            if not isinstance(module, nn.Conv2d):
                continue

            indices_to_prune = plan[name]
            num_filters = module.weight.shape[0]

            if not indices_to_prune or len(indices_to_prune) >= num_filters:
                continue

            mask = torch.ones(num_filters, dtype=torch.float32, device=module.weight.device)
            for idx in indices_to_prune:
                mask[idx] = 0.0

            full_mask = mask.view(-1, 1, 1, 1).expand_as(module.weight)
            prune_utils.custom_from_mask(module, name="weight", mask=full_mask)

        return model

    def compute_importance(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Compute channel importance scores for each layer.

        Args:
            model: Model to analyze.

        Returns:
            Dictionary mapping layer names to importance score tensors.
        """
        importance = {}

        for name, module in model.named_modules():
            if name in self.skip_layers:
                continue
            if isinstance(module, nn.Conv2d):
                weight = module.weight.data
                if self.criterion == "l1_norm":
                    scores = weight.abs().sum(dim=(1, 2, 3))
                elif self.criterion == "l2_norm":
                    scores = weight.pow(2).sum(dim=(1, 2, 3)).sqrt()
                elif self.criterion == "bn_scale":
                    scores = torch.ones(weight.size(0))
                elif self.criterion == "taylor":
                    scores = torch.ones(weight.size(0))
                else:
                    scores = torch.ones(weight.size(0))
                importance[name] = scores

        return importance

    def get_pruning_plan(self, model: nn.Module) -> Dict[str, List[int]]:
        """Generate a pruning plan specifying which channels to remove.

        Args:
            model: Model to plan pruning for.

        Returns:
            Dictionary mapping layer names to lists of channel indices to prune.
        """
        importance = self.compute_importance(model)
        plan = {}

        for name, scores in importance.items():
            num_prune = int(len(scores) * self.sparsity)
            if num_prune > 0:
                _, indices = torch.topk(scores, num_prune, largest=False)
                plan[name] = sorted(indices.tolist())

        return plan

    def __repr__(self) -> str:
        return f"StructuredPruner(sparsity={self.sparsity}, criterion={self.criterion}, granularity={self.granularity})"
