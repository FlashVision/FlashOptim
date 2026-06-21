"""Distillation loss functions for knowledge transfer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Combined distillation loss (task loss + KD loss).

    Computes: L = alpha * L_KD + (1 - alpha) * L_task

    Args:
        temperature: Softmax temperature for distillation.
        alpha: Weight of distillation loss vs task loss.
        task_loss_fn: Task-specific loss function.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        task_loss_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.task_loss_fn = task_loss_fn or nn.CrossEntropyLoss()
        self.kd_loss = KLDivergenceLoss(temperature=temperature)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined distillation loss.

        Args:
            student_logits: Student model output logits.
            teacher_logits: Teacher model output logits (detached).
            targets: Ground truth labels.

        Returns:
            Combined loss scalar.
        """
        kd_loss = self.kd_loss(student_logits, teacher_logits)
        task_loss = self.task_loss_fn(student_logits, targets)
        return self.alpha * kd_loss + (1 - self.alpha) * task_loss


class KLDivergenceLoss(nn.Module):
    """KL Divergence loss for soft label distillation.

    Args:
        temperature: Temperature for softening probabilities.
        reduction: Loss reduction mode ('mean', 'sum', 'batchmean').
    """

    def __init__(self, temperature: float = 4.0, reduction: str = "batchmean") -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between softened teacher and student outputs.

        Args:
            student_logits: Student predictions (raw logits).
            teacher_logits: Teacher predictions (raw logits, detached).

        Returns:
            Scaled KL divergence loss.
        """
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits.detach() / T, dim=-1)
        loss = F.kl_div(student_soft, teacher_soft, reduction=self.reduction)
        return loss * (T * T)


class FeatureMatchingLoss(nn.Module):
    """Feature-level matching loss for intermediate layer distillation.

    Supports MSE, L1, and cosine similarity matching between teacher
    and student feature maps.

    Args:
        loss_type: Type of matching loss ('mse', 'l1', 'cosine').
        normalize: Whether to L2-normalize features before matching.
    """

    def __init__(self, loss_type: str = "mse", normalize: bool = False) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute feature matching loss.

        Args:
            student_features: Student intermediate features.
            teacher_features: Teacher intermediate features (detached).

        Returns:
            Feature matching loss scalar.
        """
        teacher_features = teacher_features.detach()

        if self.normalize:
            student_features = F.normalize(student_features, dim=1)
            teacher_features = F.normalize(teacher_features, dim=1)

        if self.loss_type == "mse":
            return F.mse_loss(student_features, teacher_features)
        elif self.loss_type == "l1":
            return F.l1_loss(student_features, teacher_features)
        elif self.loss_type == "cosine":
            cos_sim = F.cosine_similarity(student_features, teacher_features, dim=1)
            return 1.0 - cos_sim.mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
