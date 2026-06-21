"""Model benchmarking — FPS, latency, parameter count, and model size."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class Benchmark:
    """Benchmark a model for speed, size, and efficiency metrics.

    Args:
        device: Device for benchmarking ('cpu', 'cuda').
        input_size: Model input dimensions (C, H, W).
        warmup: Number of warmup iterations.
        runs: Number of timed iterations.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        input_size: tuple = (3, 640, 640),
        warmup: int = 10,
        runs: int = 100,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.warmup = warmup
        self.runs = runs

    def run(self, model: nn.Module) -> Dict[str, Any]:
        """Run benchmark on a single model.

        Args:
            model: PyTorch model to benchmark.

        Returns:
            Dictionary with keys: 'fps', 'latency_ms', 'params',
            'model_size_mb', 'device'.
        """
        model = model.to(self.device).eval()
        dummy = torch.randn(1, *self.input_size, device=self.device)

        with torch.no_grad():
            for _ in range(self.warmup):
                model(dummy)

            if self.device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(self.runs):
                model(dummy)

            if self.device == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

        latency_ms = (elapsed / self.runs) * 1000.0
        fps = self.runs / elapsed

        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(
            p.nelement() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)

        return {
            "fps": round(fps, 1),
            "latency_ms": round(latency_ms, 3),
            "params": total_params,
            "model_size_mb": round(model_size_mb, 2),
            "device": self.device,
        }

    def compare(self, models: Dict[str, nn.Module]) -> Dict[str, Dict[str, Any]]:
        """Benchmark multiple models and compare results.

        Args:
            models: Dictionary mapping model names to nn.Module instances.

        Returns:
            Dictionary mapping model names to their benchmark results.
        """
        results = {}
        for name, model in models.items():
            results[name] = self.run(model)
        return results

    def __repr__(self) -> str:
        return (
            f"Benchmark(device={self.device}, input_size={self.input_size}, "
            f"warmup={self.warmup}, runs={self.runs})"
        )
