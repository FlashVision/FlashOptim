"""Importance scoring methods for pruning decisions."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ImportanceScorer:
    """Computes importance scores for model parameters to guide pruning.

    Supports multiple scoring methods: magnitude, Taylor expansion,
    gradient-based, and activation-based importance.

    Args:
        method: Scoring method ('magnitude', 'taylor', 'gradient', 'activation').
        granularity: Score granularity ('weight', 'channel', 'filter', 'layer').
        normalize: Whether to normalize scores to [0, 1].
    """

    METHODS = ("magnitude", "taylor", "gradient", "activation")

    def __init__(
        self,
        method: str = "magnitude",
        granularity: str = "channel",
        normalize: bool = True,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Options: {self.METHODS}")

        self.method = method
        self.granularity = granularity
        self.normalize = normalize

    def score(
        self,
        model: nn.Module,
        dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute importance scores for all prunable layers.

        Args:
            model: Model to score.
            dataloader: Data for gradient/activation-based methods.

        Returns:
            Dictionary mapping layer names to importance score tensors.
        """
        if self.method == "magnitude":
            return self._magnitude_importance(model)
        elif self.method == "taylor":
            return self._taylor_importance(model, dataloader)
        elif self.method == "gradient":
            return self._gradient_importance(model, dataloader)
        elif self.method == "activation":
            return self._activation_importance(model, dataloader)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _magnitude_importance(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Compute magnitude-based importance scores.

        Score = L1-norm or L2-norm of weights at the specified granularity.
        """
        scores = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight.data
                if self.granularity == "filter":
                    s = weight.abs().sum(dim=(1, 2, 3))
                elif self.granularity == "channel":
                    s = weight.abs().sum(dim=(0, 2, 3))
                else:
                    s = weight.abs().flatten()
                scores[name] = self._maybe_normalize(s)
        return scores

    def _taylor_importance(self, model: nn.Module, dataloader: Optional[DataLoader]) -> Dict[str, torch.Tensor]:
        """Compute Taylor expansion-based importance.

        Score = |weight * gradient| — approximates loss change from removal.
        """
        if dataloader is None:
            raise ValueError("dataloader is required for Taylor importance scoring")

        model.train()
        scores: Dict[str, torch.Tensor] = {}
        grad_accum: Dict[str, torch.Tensor] = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                grad_accum[name] = torch.zeros_like(module.weight)

        criterion = nn.CrossEntropyLoss()
        num_batches = 0

        for batch in dataloader:
            if num_batches >= 10:
                break
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                continue

            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                    grad_accum[name] += module.weight.grad.data.abs()

            num_batches += 1

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and name in grad_accum:
                taylor_score = (module.weight.data * grad_accum[name]).abs()
                if self.granularity == "filter":
                    s = taylor_score.sum(dim=(1, 2, 3))
                elif self.granularity == "channel":
                    s = taylor_score.sum(dim=(0, 2, 3))
                else:
                    s = taylor_score.flatten()
                scores[name] = self._maybe_normalize(s)

        model.eval()
        return scores

    def _gradient_importance(self, model: nn.Module, dataloader: Optional[DataLoader]) -> Dict[str, torch.Tensor]:
        """Compute gradient-based importance.

        Score = mean |gradient| across calibration batches.
        """
        if dataloader is None:
            raise ValueError("dataloader is required for gradient importance scoring")

        model.train()
        scores: Dict[str, torch.Tensor] = {}
        grad_accum: Dict[str, torch.Tensor] = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                grad_accum[name] = torch.zeros_like(module.weight)

        criterion = nn.CrossEntropyLoss()
        num_batches = 0

        for batch in dataloader:
            if num_batches >= 10:
                break
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                continue

            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                    grad_accum[name] += module.weight.grad.data.abs()

            num_batches += 1

        for name in grad_accum:
            grad = grad_accum[name] / max(num_batches, 1)
            if self.granularity == "filter":
                s = grad.sum(dim=(1, 2, 3))
            elif self.granularity == "channel":
                s = grad.sum(dim=(0, 2, 3))
            else:
                s = grad.flatten()
            scores[name] = self._maybe_normalize(s)

        model.eval()
        return scores

    def _activation_importance(self, model: nn.Module, dataloader: Optional[DataLoader]) -> Dict[str, torch.Tensor]:
        """Compute activation-based importance.

        Score = mean activation magnitude at each channel/filter.
        """
        if dataloader is None:
            raise ValueError("dataloader is required for activation importance scoring")

        model.eval()
        activation_sums: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(name: str):
            def hook_fn(module, inp, output):
                act = output.detach().abs()
                channel_score = act.mean(dim=(0, 2, 3)) if act.ndim == 4 else act.mean(dim=0)
                if name not in activation_sums:
                    activation_sums[name] = channel_score
                else:
                    activation_sums[name] += channel_score

            return hook_fn

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(make_hook(name)))

        num_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                if num_batches >= 10:
                    break
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                model(inputs)
                num_batches += 1

        for h in hooks:
            h.remove()

        scores = {}
        for name, act_sum in activation_sums.items():
            s = act_sum / max(num_batches, 1)
            scores[name] = self._maybe_normalize(s)

        return scores

    def _maybe_normalize(self, scores: torch.Tensor) -> torch.Tensor:
        """Normalize scores to [0, 1] if configured."""
        if self.normalize and scores.numel() > 0:
            min_val = scores.min()
            max_val = scores.max()
            if max_val - min_val > 0:
                return (scores - min_val) / (max_val - min_val)
        return scores

    def rank_layers(self, scores: Dict[str, torch.Tensor]) -> List[str]:
        """Rank layers by their mean importance score (lowest first).

        Args:
            scores: Per-layer importance scores.

        Returns:
            Layer names sorted by ascending importance.
        """
        layer_means = {name: s.mean().item() for name, s in scores.items()}
        return sorted(layer_means, key=lambda x: layer_means[x])

    def __repr__(self) -> str:
        return f"ImportanceScorer(method={self.method}, granularity={self.granularity})"
