"""Validator for evaluating optimized model quality."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Validator:
    """Evaluates model performance before and after optimization.

    Computes accuracy, latency, model size, and other metrics to quantify
    the impact of optimization techniques.

    Args:
        device: Device to run validation on.
        metrics: List of metric names to compute.
        verbose: Whether to print results.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        metrics: Optional[list] = None,
        verbose: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = metrics or ["accuracy", "latency", "model_size", "flops"]
        self.verbose = verbose

    def validate(
        self,
        model: nn.Module,
        dataloader: Optional[DataLoader] = None,
        data_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run validation and compute metrics.

        Args:
            model: The model to evaluate.
            dataloader: Optional pre-built DataLoader.
            data_path: Optional path to validation data.

        Returns:
            Dictionary of metric names to values.
        """
        import time

        model = model.to(self.device).eval()
        results: Dict[str, Any] = {}

        if "model_size" in self.metrics:
            results["model_size_mb"] = self.model_size_mb(model)

        if "accuracy" in self.metrics and dataloader is not None:
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in dataloader:
                    if isinstance(batch, (list, tuple)):
                        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                    else:
                        continue
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            results["accuracy"] = correct / total if total > 0 else 0.0

        if "latency" in self.metrics:
            dummy = torch.randn(1, 3, 640, 640, device=self.device)
            with torch.no_grad():
                for _ in range(10):
                    model(dummy)

            if self.device == "cuda":
                torch.cuda.synchronize()
            timings = []
            with torch.no_grad():
                for _ in range(50):
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model(dummy)
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    timings.append((time.perf_counter() - t0) * 1000)
            results["latency_ms"] = sum(timings) / len(timings)
            results["fps"] = 1000.0 / results["latency_ms"]

        results["total_params"] = self.count_parameters(model)
        results["trainable_params"] = self.count_parameters(model, trainable_only=True)

        if self.verbose:
            print("Validation Results:")
            for k, v in results.items():
                print(f"  {k}: {v}")

        return results

    def compare(
        self,
        original: nn.Module,
        optimized: nn.Module,
        dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare metrics between original and optimized models.

        Args:
            original: The original (unoptimized) model.
            optimized: The optimized model.
            dataloader: DataLoader for evaluation data.

        Returns:
            Dictionary with 'original', 'optimized', and 'delta' results.
        """
        orig_results = self.validate(original, dataloader)
        opt_results = self.validate(optimized, dataloader)

        delta: Dict[str, Any] = {}
        for key in orig_results:
            if isinstance(orig_results[key], (int, float)) and key in opt_results:
                delta[key] = opt_results[key] - orig_results[key]

        comparison = {
            "original": orig_results,
            "optimized": opt_results,
            "delta": delta,
        }

        if self.verbose:
            print("\nModel Comparison:")
            print(f"  {'Metric':<20} {'Original':<15} {'Optimized':<15} {'Delta':<15}")
            print("  " + "-" * 65)
            for key in orig_results:
                if key in delta:
                    print(
                        f"  {key:<20} {orig_results[key]:<15.4f} "
                        f"{opt_results[key]:<15.4f} {delta[key]:<+15.4f}"
                    )

        return comparison

    @staticmethod
    def model_size_mb(model: nn.Module) -> float:
        """Calculate model size in megabytes.

        Args:
            model: PyTorch model.

        Returns:
            Model size in MB.
        """
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

    @staticmethod
    def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
        """Count model parameters.

        Args:
            model: PyTorch model.
            trainable_only: If True, count only trainable parameters.

        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())
