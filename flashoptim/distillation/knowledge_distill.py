"""Logit-level Knowledge Distillation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistiller:
    """Logit-level knowledge distillation (Hinton et al., 2015).

    Transfers "dark knowledge" from a large teacher model to a smaller
    student model by matching softened output distributions.

    Args:
        temperature: Softmax temperature for softening distributions.
        alpha: Weight of KD loss vs task loss (0.0 to 1.0).
        loss_type: Type of divergence ('kl_div', 'mse', 'cosine').
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        loss_type: str = "kl_div",
    ) -> None:
        self.temperature = temperature
        self.alpha = alpha
        self.loss_type = loss_type

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        task_loss_fn: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Compute the distillation loss.

        Args:
            student_logits: Raw logits from student model.
            teacher_logits: Raw logits from teacher model (no grad).
            targets: Ground truth labels (for task loss component).
            task_loss_fn: Task-specific loss function.

        Returns:
            Combined distillation + task loss.
        """
        T = self.temperature
        teacher_logits = teacher_logits.detach()

        if self.loss_type == "kl_div":
            student_soft = F.log_softmax(student_logits / T, dim=-1)
            teacher_soft = F.softmax(teacher_logits / T, dim=-1)
            kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)
        elif self.loss_type == "mse":
            kd_loss = F.mse_loss(student_logits / T, teacher_logits / T) * (T * T)
        elif self.loss_type == "cosine":
            cos_sim = F.cosine_similarity(student_logits, teacher_logits, dim=-1)
            kd_loss = (1.0 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if targets is not None and task_loss_fn is not None:
            task_loss = task_loss_fn(student_logits, targets)
            return self.alpha * kd_loss + (1 - self.alpha) * task_loss

        return kd_loss

    def distill(
        self,
        teacher: nn.Module,
        student: nn.Module,
        train_data: Optional[str] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Run full distillation training pipeline.

        Args:
            teacher: Pre-trained teacher model.
            student: Student model to train.
            train_data: Path to training data.
            **kwargs: Additional training parameters (train_loader, val_loader,
                epochs, lr, device).

        Returns:
            Trained student model.
        """
        train_loader = kwargs.get("train_loader")
        val_loader = kwargs.get("val_loader")
        epochs = kwargs.get("epochs", 100)
        lr = kwargs.get("lr", 0.001)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        if train_loader is None:
            raise ValueError("train_loader is required for distillation")

        teacher = teacher.to(device).eval()
        student = student.to(device).train()

        optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        task_loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            student.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                else:
                    continue

                with torch.no_grad():
                    teacher_logits = teacher(inputs)

                student_logits = student(inputs)
                loss = self.compute_loss(student_logits, teacher_logits, targets, task_loss_fn)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(num_batches, 1)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Distill Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

        return student

    def __repr__(self) -> str:
        return f"KnowledgeDistiller(T={self.temperature}, alpha={self.alpha}, loss={self.loss_type})"
