#!/usr/bin/env python3
"""Example: Benchmarking Optimized vs Original Models.

Demonstrates how to use Benchmark and DeploymentProfiler to compare
model performance before and after optimization.

Usage:
    python examples/benchmark_optimized.py
"""

import torch
import torch.nn as nn

from flashoptim.analytics import Benchmark
from flashoptim.solutions import AutoOptimizer, DeploymentProfiler
from flashoptim.pruning import UnstructuredPruner
from flashoptim.utils.model_utils import count_parameters, get_model_size_mb, get_sparsity


def build_demo_model() -> nn.Module:
    """Build a simple CNN for demonstration."""
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


def main():
    print("=" * 60)
    print("FlashOptim — Benchmark Optimized vs Original")
    print("=" * 60)

    original = build_demo_model()
    original.eval()

    print("\n--- Benchmarking Original Model ---")
    benchmark = Benchmark(device="cpu", input_size=(3, 32, 32), warmup=5, runs=20)
    original_result = benchmark.run(original)
    print(f"  FPS:        {original_result['fps']}")
    print(f"  Latency:    {original_result['latency_ms']:.3f} ms")
    print(f"  Parameters: {original_result['params']:,}")
    print(f"  Size:       {original_result['model_size_mb']:.2f} MB")

    print("\n--- Pruning Model (50% sparsity) ---")
    optimized = build_demo_model()
    pruner = UnstructuredPruner(sparsity=0.5, method="magnitude", iterative=False)
    pruner.prune(optimized)
    UnstructuredPruner.remove_pruning(optimized)
    optimized.eval()

    optimized_result = benchmark.run(optimized)
    print(f"  FPS:        {optimized_result['fps']}")
    print(f"  Latency:    {optimized_result['latency_ms']:.3f} ms")
    print(f"  Sparsity:   {get_sparsity(optimized)['global']:.2%}")

    print("\n--- Comparison ---")
    comparison = benchmark.compare(
        {
            "original": original,
            "pruned_50pct": optimized,
        }
    )
    for name, res in comparison.items():
        print(f"  {name}: {res['fps']:.1f} FPS, {res['latency_ms']:.3f} ms, {res['params']:,} params")

    print("\n--- Deployment Profiler ---")
    profiler = DeploymentProfiler(
        device="cpu",
        input_size=(3, 32, 32),
        batch_sizes=[1, 4],
        warmup_runs=5,
        benchmark_runs=20,
    )
    profile = profiler.profile(optimized)
    print(f"  Total params:    {profile['total_params']:,}")
    print(f"  Model size:      {profile['model_size_mb']:.2f} MB")
    print(f"  Sparsity:        {profile['sparsity']:.2%}")
    print(f"  Latency (bs=1):  {profile['latency_ms'].get('batch_1', 'N/A')} ms")

    suggestions = profiler.suggest_optimizations(profile)
    print("\n  Optimization Suggestions:")
    for s in suggestions:
        print(f"    - {s}")

    print("\nDone!")


if __name__ == "__main__":
    main()
