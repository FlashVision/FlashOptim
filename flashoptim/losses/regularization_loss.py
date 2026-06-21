"""Regularization losses for pruning and sparsity enforcement."""

from __future__ import annotations

import torch
import torch.nn as nn


class SparsityLoss(nn.Module):
    """Sparsity-inducing regularization loss.

    Encourages weight sparsity during training to aid pruning.
    Uses L1 penalty on model parameters.

    Args:
        target_sparsity: Target sparsity ratio (0.0 to 1.0).
        lambda_sparse: Regularization strength.
    """

    def __init__(self, target_sparsity: float = 0.5, lambda_sparse: float = 1e-4) -> None:
        super().__init__()
        self.target_sparsity = target_sparsity
        self.lambda_sparse = lambda_sparse

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute sparsity regularization loss over model parameters.

        Args:
            model: The model whose parameters to regularize.

        Returns:
            Sparsity regularization loss scalar.
        """
        l1_sum = torch.tensor(0.0, device=next(model.parameters()).device)
        num_params = 0

        for param in model.parameters():
            if param.requires_grad:
                l1_sum = l1_sum + param.abs().sum()
                num_params += param.numel()

        if num_params == 0:
            return torch.tensor(0.0)

        return self.lambda_sparse * l1_sum / num_params


class L1RegularizationLoss(nn.Module):
    """L1 regularization for structured pruning.

    Applies group-level L1 penalty to encourage entire channels/filters
    to become zero, facilitating structured pruning.

    Args:
        lambda_reg: Regularization coefficient.
        granularity: Regularization level ('channel', 'filter', 'layer').
    """

    def __init__(self, lambda_reg: float = 1e-4, granularity: str = "channel") -> None:
        super().__init__()
        self.lambda_reg = lambda_reg
        self.granularity = granularity

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute group L1 regularization.

        Args:
            model: Target model.

        Returns:
            Group L1 regularization loss.
        """
        reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        count = 0

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight
                if self.granularity == "filter":
                    group_norm = weight.view(weight.size(0), -1).norm(p=1, dim=1)
                elif self.granularity == "channel":
                    group_norm = weight.view(weight.size(0), weight.size(1), -1).norm(p=1, dim=2).mean(0)
                else:
                    group_norm = weight.abs().sum().unsqueeze(0)

                reg_loss = reg_loss + group_norm.sum()
                count += group_norm.numel()

        if count == 0:
            return torch.tensor(0.0)

        return self.lambda_reg * reg_loss / count
