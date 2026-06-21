#!/usr/bin/env python3
"""Example: Knowledge Distillation with FlashOptim.

Demonstrates how to set up a teacher-student distillation pipeline
using KnowledgeDistiller and FeatureDistiller.

Usage:
    python examples/distill_model.py
"""

import torch
import torch.nn as nn

from flashoptim.distillation import KnowledgeDistiller, FeatureDistiller, SelfDistiller
from flashoptim.utils.model_utils import count_parameters, get_model_size_mb


def build_teacher() -> nn.Module:
    """Build a larger teacher model."""
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(256, 10),
    )


def build_student() -> nn.Module:
    """Build a smaller student model."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )


def main():
    print("=" * 60)
    print("FlashOptim — Knowledge Distillation Example")
    print("=" * 60)

    teacher = build_teacher()
    student = build_student()

    print(f"\nTeacher Model:")
    print(f"  Parameters: {count_parameters(teacher):,}")
    print(f"  Size:       {get_model_size_mb(teacher):.2f} MB")

    print(f"\nStudent Model:")
    print(f"  Parameters: {count_parameters(student):,}")
    print(f"  Size:       {get_model_size_mb(student):.2f} MB")

    compression = count_parameters(teacher) / count_parameters(student)
    print(f"\nCompression ratio: {compression:.1f}x")

    print("\n--- Logit-Level Distillation ---")
    kd = KnowledgeDistiller(temperature=4.0, alpha=0.7, loss_type="kl_div")
    print(f"  Distiller: {kd}")

    dummy_input = torch.randn(4, 3, 32, 32)
    teacher.eval()
    student.train()

    with torch.no_grad():
        teacher_logits = teacher(dummy_input)
    student_logits = student(dummy_input)

    targets = torch.randint(0, 10, (4,))
    loss = kd.compute_loss(
        student_logits,
        teacher_logits,
        targets,
        task_loss_fn=nn.CrossEntropyLoss(),
    )
    print(f"  KD Loss: {loss.item():.4f}")

    print("\n--- Feature-Level Distillation ---")
    fd = FeatureDistiller(loss_type="mse", projector=True)
    print(f"  Distiller: {fd}")

    print("\n--- Self-Distillation ---")
    sd = SelfDistiller(temperature=3.0, aux_weight=0.5)
    print(f"  Distiller: {sd}")

    print("\nDone!")


if __name__ == "__main__":
    main()
