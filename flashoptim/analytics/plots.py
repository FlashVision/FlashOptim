"""Visualization utilities for training curves, optimization comparisons, and sparsity maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn


def plot_training_curves(
    logs: Dict[str, List[float]],
    title: str = "Training Curves",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot training and validation loss/accuracy curves.

    Args:
        logs: Dictionary mapping metric names to lists of values per epoch.
              E.g. {'train_loss': [...], 'val_loss': [...], 'val_accuracy': [...]}.
        title: Plot title.
        save_path: Path to save the figure. If None, calls plt.show().

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install flashoptim[analytics]")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    loss_keys = [k for k in logs if "loss" in k.lower()]
    acc_keys = [k for k in logs if "acc" in k.lower() or "map" in k.lower()]

    for key in loss_keys:
        axes[0].plot(logs[key], label=key)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for key in acc_keys:
        axes[1].plot(logs[key], label=key)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].set_title(f"{title} — Accuracy / mAP")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_optimization_comparison(
    results: Dict[str, Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Bar chart comparing models across multiple optimization metrics.

    Args:
        results: Dictionary mapping model names to metric dictionaries.
                 E.g. {'original': {'latency_ms': 10, 'params': 1e6}, ...}.
        metrics: List of metric keys to plot. Defaults to all numeric metrics.
        save_path: Path to save the figure. If None, calls plt.show().

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np  # noqa: F401  # noqa: F401
    except ImportError:
        raise ImportError(
            "matplotlib and numpy are required for plotting. Install with: pip install flashoptim[analytics]"
        )

    model_names = list(results.keys())
    if not metrics:
        sample = next(iter(results.values()))
        metrics = [k for k, v in sample.items() if isinstance(v, (int, float))]

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))

    if num_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = [results[m].get(metric, 0) for m in model_names]
        bars = ax.bar(model_names, values, color=plt.cm.Set2.colors[: len(values)])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}" if isinstance(val, float) else str(val),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_sparsity_map(
    model: nn.Module,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Visualize per-layer sparsity as a horizontal bar chart.

    Args:
        model: Pruned PyTorch model.
        save_path: Path to save the figure. If None, calls plt.show().

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install flashoptim[analytics]")

    layer_names = []
    sparsities = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            total = weight.numel()
            zeros = (weight == 0).sum().item()
            sparsity = zeros / total if total > 0 else 0.0
            layer_names.append(name)
            sparsities.append(sparsity)

    fig, ax = plt.subplots(figsize=(10, max(4, len(layer_names) * 0.4)))
    colors = ["#e74c3c" if s > 0.8 else "#f39c12" if s > 0.5 else "#2ecc71" for s in sparsities]

    ax.barh(layer_names, sparsities, color=colors)
    ax.set_xlabel("Sparsity")
    ax.set_title("Per-Layer Sparsity Map")
    ax.set_xlim(0, 1.0)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
