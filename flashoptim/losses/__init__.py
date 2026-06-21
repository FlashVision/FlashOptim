"""Loss functions for distillation and optimization."""

from flashoptim.losses.distillation_loss import DistillationLoss, KLDivergenceLoss, FeatureMatchingLoss
from flashoptim.losses.regularization_loss import SparsityLoss, L1RegularizationLoss

__all__ = [
    "DistillationLoss",
    "KLDivergenceLoss",
    "FeatureMatchingLoss",
    "SparsityLoss",
    "L1RegularizationLoss",
]
