"""Deployment profiler — measure model performance characteristics."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class DeploymentProfiler:
    """Profile model performance for deployment readiness assessment.

    Measures latency, throughput, memory usage, and model size, then
    suggests optimizations based on the results.

    Args:
        device: Device for profiling ('cpu', 'cuda').
        input_size: Input tensor dimensions (C, H, W).
        batch_sizes: Batch sizes to profile.
        warmup_runs: Warmup iterations before timing.
        benchmark_runs: Timed benchmark iterations.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        input_size: Tuple[int, int, int] = (3, 640, 640),
        batch_sizes: Optional[List[int]] = None,
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.batch_sizes = batch_sizes or [1, 4, 8, 16]
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

    def profile(self, model: nn.Module) -> Dict[str, Any]:
        """Run full profiling suite on a model.

        Args:
            model: PyTorch model to profile.

        Returns:
            Dictionary with latency, throughput, memory, and size metrics.
        """
        model = model.to(self.device).eval()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)

        zero_params = sum((p == 0).sum().item() for p in model.parameters())
        sparsity = zero_params / total_params if total_params > 0 else 0.0

        latency_results = {}
        throughput_results = {}

        for bs in self.batch_sizes:
            dummy = torch.randn(bs, *self.input_size, device=self.device)
            latency_ms = self._measure_latency(model, dummy)
            latency_results[f"batch_{bs}"] = round(latency_ms, 3)
            throughput_results[f"batch_{bs}"] = round(bs / (latency_ms / 1000.0), 1)

        memory_info = self._measure_memory(model)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": round(model_size_mb, 2),
            "sparsity": round(sparsity, 4),
            "latency_ms": latency_results,
            "throughput_fps": throughput_results,
            "memory": memory_info,
            "device": self.device,
        }

    def _measure_latency(self, model: nn.Module, dummy_input: torch.Tensor) -> float:
        """Measure inference latency for a given input.

        Args:
            model: Model to benchmark.
            dummy_input: Input tensor.

        Returns:
            Average latency in milliseconds.
        """
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                model(dummy_input)

            if self.device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(self.benchmark_runs):
                model(dummy_input)

            if self.device == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

        return (elapsed / self.benchmark_runs) * 1000.0

    def _measure_memory(self, model: nn.Module) -> Dict[str, Any]:
        """Measure memory usage.

        Args:
            model: Model to profile.

        Returns:
            Memory usage statistics.
        """
        result: Dict[str, Any] = {}

        param_mem = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        buffer_mem = sum(b.nelement() * b.element_size() for b in model.buffers()) / (1024 * 1024)
        result["param_memory_mb"] = round(param_mem, 2)
        result["buffer_memory_mb"] = round(buffer_mem, 2)

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            dummy = torch.randn(1, *self.input_size, device=self.device)
            with torch.no_grad():
                model(dummy)
            result["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

        return result

    def suggest_optimizations(self, profile_result: Dict[str, Any]) -> List[str]:
        """Suggest optimizations based on profiling results.

        Args:
            profile_result: Output from :meth:`profile`.

        Returns:
            List of optimization suggestions.
        """
        suggestions = []

        if profile_result["model_size_mb"] > 100:
            suggestions.append("Model is large (>100 MB). Consider INT8 quantization to reduce size by ~4x.")

        if profile_result["sparsity"] < 0.1:
            suggestions.append(
                "Model has low sparsity. Magnitude pruning at 50% can reduce compute with minimal accuracy loss."
            )

        batch1_latency = profile_result["latency_ms"].get("batch_1", 0)
        if batch1_latency > 50:
            suggestions.append(
                f"Single-sample latency is {batch1_latency:.1f} ms. "
                "Consider structured pruning or knowledge distillation for a smaller architecture."
            )

        if profile_result["total_params"] > 10_000_000:
            suggestions.append(
                "Model has >10M parameters. Knowledge distillation to a smaller student can yield significant speedups."
            )

        if not suggestions:
            suggestions.append("Model appears well-optimized for deployment.")

        return suggestions

    def compare(
        self,
        original: Dict[str, Any],
        optimized: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare original vs optimized model profiles.

        Args:
            original: Profile of the original model.
            optimized: Profile of the optimized model.

        Returns:
            Comparison dictionary with improvements.
        """
        comparison: Dict[str, Any] = {
            "param_reduction": round(1.0 - optimized["total_params"] / max(original["total_params"], 1), 4),
            "size_reduction": round(1.0 - optimized["model_size_mb"] / max(original["model_size_mb"], 0.01), 4),
            "sparsity_gain": round(optimized["sparsity"] - original["sparsity"], 4),
        }

        for key in original["latency_ms"]:
            if key in optimized["latency_ms"]:
                orig_lat = original["latency_ms"][key]
                opt_lat = optimized["latency_ms"][key]
                speedup = orig_lat / opt_lat if opt_lat > 0 else float("inf")
                comparison[f"speedup_{key}"] = round(speedup, 2)

        return comparison

    def __repr__(self) -> str:
        return f"DeploymentProfiler(device={self.device}, input_size={self.input_size}, batch_sizes={self.batch_sizes})"
