#!/usr/bin/env python3
"""Example: Unstructured Pruning with FlashOptim.

Demonstrates how to prune a model using magnitude-based unstructured pruning
and measure the resulting sparsity.

Usage:
    python examples/prune_model.py
"""

import torch
import torch.nn as nn

from flashoptim.pruning import UnstructuredPruner, StructuredPruner, ImportanceScorer
from flashoptim.utils.model_utils import count_parameters, get_model_size_mb, get_sparsity


def build_demo_model() -> nn.Module:
    """Build a simple CNN for demonstration."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 10),
    )


def main():
    print("=" * 60)
    print("FlashOptim — Unstructured Pruning Example")
    print("=" * 60)

    model = build_demo_model()
    model.eval()

    print(f"\nOriginal Model:")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Size:       {get_model_size_mb(model):.2f} MB")
    original_sparsity = get_sparsity(model)
    print(f"  Sparsity:   {original_sparsity['global']:.2%}")

    print("\n--- Importance Scoring ---")
    scorer = ImportanceScorer(method="magnitude", granularity="filter")
    scores = scorer.score(model)
    for name, score_tensor in scores.items():
        print(f"  {name}: mean={score_tensor.mean():.4f}, min={score_tensor.min():.4f}")

    print("\n--- Unstructured Pruning (50% sparsity) ---")
    pruner = UnstructuredPruner(sparsity=0.5, method="magnitude", iterative=False)
    model = pruner.prune(model)

    pruned_sparsity = UnstructuredPruner.get_sparsity(model)
    print(f"  Global sparsity: {pruned_sparsity['global']:.2%}")

    for name, sp in pruned_sparsity.items():
        if name != "global":
            print(f"  {name}: {sp:.2%}")

    print("\n--- Making pruning permanent ---")
    UnstructuredPruner.remove_pruning(model)

    final_sparsity = get_sparsity(model)
    print(f"  Final sparsity (permanent): {final_sparsity['global']:.2%}")

    dummy_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\nModel output shape: {output.shape}")
    print("Done!")


if __name__ == "__main__":
    main()
