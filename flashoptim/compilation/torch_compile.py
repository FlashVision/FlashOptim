"""torch.compile integration for graph-based model optimization.

Provides a high-level wrapper around torch.compile with mode selection,
backend configuration, and performance benchmarking utilities.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn


class TorchCompiler:
    """torch.compile wrapper with mode selection and benchmarking.

    Wraps PyTorch 2.x torch.compile with convenient mode presets,
    warmup handling, and performance comparison utilities.

    Args:
        mode: Compilation mode:
            - 'default': balanced compilation time and performance.
            - 'reduce-overhead': minimize framework overhead (CUDA graphs).
            - 'max-autotune': maximum performance (benchmarks multiple kernels).
        backend: Compiler backend ('inductor', 'cudagraphs', 'eager').
        fullgraph: Whether to require full-graph compilation (no graph breaks).
        dynamic: Enable dynamic shape support.
        disable: Set True to disable compilation (passthrough mode).

    Example:
        >>> compiler = TorchCompiler(mode="max-autotune")
        >>> compiled_model = compiler.compile(model)
        >>> benchmark = compiler.benchmark(compiled_model, sample_input)
    """

    MODES = ("default", "reduce-overhead", "max-autotune")
    BACKENDS = ("inductor", "cudagraphs", "eager")

    def __init__(
        self,
        mode: str = "default",
        backend: str = "inductor",
        fullgraph: bool = False,
        dynamic: bool = False,
        disable: bool = False,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}. Options: {self.MODES}")
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. Options: {self.BACKENDS}")
        self.mode = mode
        self.backend = backend
        self.fullgraph = fullgraph
        self.dynamic = dynamic
        self.disable = disable
        self._compile_time: Optional[float] = None
        self._compiled_model: Optional[nn.Module] = None

    def compile(
        self,
        model: nn.Module,
        **compile_kwargs: Any,
    ) -> nn.Module:
        """Compile a model using torch.compile.

        Args:
            model: PyTorch model to compile.
            **compile_kwargs: Additional kwargs passed to torch.compile.

        Returns:
            Compiled model (or original if disable=True or torch.compile unavailable).
        """
        if self.disable:
            self._compiled_model = model
            return model

        if not hasattr(torch, "compile"):
            import warnings

            warnings.warn("torch.compile not available (requires PyTorch 2.0+). Returning original model.")
            self._compiled_model = model
            return model

        start = time.perf_counter()

        compiled = torch.compile(
            model,
            mode=self.mode,
            backend=self.backend,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
            **compile_kwargs,
        )

        self._compile_time = time.perf_counter() - start
        self._compiled_model = compiled
        return compiled

    def compile_function(
        self,
        fn: Callable,
        **compile_kwargs: Any,
    ) -> Callable:
        """Compile a standalone function using torch.compile.

        Args:
            fn: Function to compile.
            **compile_kwargs: Additional kwargs passed to torch.compile.

        Returns:
            Compiled function.
        """
        if self.disable or not hasattr(torch, "compile"):
            return fn

        return torch.compile(
            fn,
            mode=self.mode,
            backend=self.backend,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
            **compile_kwargs,
        )

    def warmup(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        warmup_iterations: int = 3,
    ) -> None:
        """Run warmup iterations to trigger JIT compilation.

        Args:
            model: Compiled model.
            sample_input: Representative input tensor.
            warmup_iterations: Number of warmup forward passes.
        """
        model.eval()
        with torch.no_grad():
            for _ in range(warmup_iterations):
                model(sample_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def benchmark(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        n_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> Dict[str, float]:
        """Benchmark model inference performance.

        Args:
            model: Model to benchmark (compiled or not).
            sample_input: Input tensor for benchmarking.
            n_iterations: Number of timed iterations.
            warmup_iterations: Number of warmup iterations.

        Returns:
            Dictionary with timing statistics.
        """
        model.eval()

        self.warmup(model, sample_input, warmup_iterations)

        timings = []
        with torch.no_grad():
            for _ in range(n_iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()

                model(sample_input)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                timings.append(elapsed * 1000)

        timings_tensor = torch.tensor(timings)
        return {
            "mean_ms": timings_tensor.mean().item(),
            "median_ms": timings_tensor.median().item(),
            "std_ms": timings_tensor.std().item(),
            "min_ms": timings_tensor.min().item(),
            "max_ms": timings_tensor.max().item(),
            "p95_ms": timings_tensor.quantile(0.95).item(),
            "p99_ms": timings_tensor.quantile(0.99).item(),
            "throughput_fps": 1000.0 / timings_tensor.mean().item() * sample_input.shape[0],
            "n_iterations": n_iterations,
        }

    def compare(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        n_iterations: int = 100,
    ) -> Dict[str, Any]:
        """Compare compiled vs uncompiled model performance.

        Args:
            model: Original (uncompiled) model.
            sample_input: Input for benchmarking.
            n_iterations: Iterations per benchmark.

        Returns:
            Dictionary with 'eager', 'compiled', and 'speedup' stats.
        """
        eager_stats = self.benchmark(model, sample_input, n_iterations)

        compiled_model = self.compile(model)
        compiled_stats = self.benchmark(compiled_model, sample_input, n_iterations)

        speedup = eager_stats["mean_ms"] / max(compiled_stats["mean_ms"], 1e-6)

        return {
            "eager": eager_stats,
            "compiled": compiled_stats,
            "speedup": speedup,
            "compile_time_s": self._compile_time,
        }

    @staticmethod
    def get_backend_info() -> Dict[str, Any]:
        """Get information about available compilation backends."""
        info = {
            "torch_version": torch.__version__,
            "compile_available": hasattr(torch, "compile"),
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = torch.cuda.get_device_capability(0)

        if hasattr(torch, "_dynamo"):
            info["dynamo_available"] = True
            try:
                info["available_backends"] = list(torch._dynamo.list_backends())
            except Exception:
                info["available_backends"] = ["inductor", "eager"]
        else:
            info["dynamo_available"] = False

        return info

    @staticmethod
    def reset_dynamo() -> None:
        """Reset torch._dynamo state (useful for debugging compilation issues)."""
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()

    @property
    def compile_time(self) -> Optional[float]:
        """Return the time taken for compilation in seconds."""
        return self._compile_time

    def __repr__(self) -> str:
        return (
            f"TorchCompiler(mode='{self.mode}', backend='{self.backend}', "
            f"fullgraph={self.fullgraph}, dynamic={self.dynamic})"
        )
