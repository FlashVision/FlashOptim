"""Layer-by-layer model profiler — timing, memory, and parameter analysis."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class Profiler:
    """Layer-by-layer profiler for deep learning models.

    Hooks into each layer to measure per-layer execution time,
    parameter count, and memory footprint.

    Args:
        device: Device for profiling ('cpu', 'cuda').
        input_size: Input tensor shape (C, H, W).
    """

    def __init__(
        self,
        device: Optional[str] = None,
        input_size: tuple = (3, 640, 640),
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self._layer_times: Dict[str, float] = {}
        self._hooks: List[Any] = []

    def run(self, model: nn.Module) -> Dict[str, Any]:
        """Profile per-layer execution time.

        Registers forward hooks on all leaf modules, runs a forward pass,
        and records time spent in each layer.

        Args:
            model: Model to profile.

        Returns:
            Dictionary mapping layer names to timing and parameter info.
        """
        model = model.to(self.device).eval()
        self._layer_times.clear()

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                post_hook, pre_hook = self._make_timing_hook(name)
                h_pre = module.register_forward_pre_hook(pre_hook)
                h_post = module.register_forward_hook(post_hook)
                self._hooks.append(h_pre)
                self._hooks.append(h_post)

        dummy = torch.randn(1, *self.input_size, device=self.device)
        with torch.no_grad():
            model(dummy)

        self._remove_hooks()

        result = {}
        total_time = sum(self._layer_times.values())

        for name, module in model.named_modules():
            if name in self._layer_times:
                params = sum(p.numel() for p in module.parameters())
                layer_time = self._layer_times[name]
                result[name] = {
                    "type": module.__class__.__name__,
                    "params": params,
                    "time_ms": round(layer_time * 1000, 4),
                    "time_pct": round(layer_time / total_time * 100 if total_time > 0 else 0, 2),
                }

        return result

    def memory_profile(self, model: nn.Module) -> Dict[str, Any]:
        """Profile memory usage per layer.

        Args:
            model: Model to profile.

        Returns:
            Dictionary with per-layer memory consumption in MB.
        """
        model = model.to(self.device).eval()
        result: Dict[str, Any] = {}

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                param_mem = sum(p.nelement() * p.element_size() for p in module.parameters()) / (1024 * 1024)
                buffer_mem = sum(b.nelement() * b.element_size() for b in module.buffers()) / (1024 * 1024)

                if param_mem > 0 or buffer_mem > 0:
                    result[name] = {
                        "type": module.__class__.__name__,
                        "param_memory_mb": round(param_mem, 4),
                        "buffer_memory_mb": round(buffer_mem, 4),
                        "total_memory_mb": round(param_mem + buffer_mem, 4),
                    }

        return result

    def _make_timing_hook(self, name: str):
        """Create pre and post forward hooks that measure execution time.

        Uses a closure to store the start time from the pre-hook, then
        computes the elapsed time in the post-hook.
        """
        timing_state = {}

        def pre_hook(module, input):
            if self.device == "cuda":
                torch.cuda.synchronize()
            timing_state["start"] = time.perf_counter()

        def post_hook(module, input, output):
            if self.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - timing_state["start"]
            self._layer_times[name] = self._layer_times.get(name, 0) + elapsed

        self._pre_hooks_storage = getattr(self, "_pre_hooks_storage", [])
        return post_hook, pre_hook

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __repr__(self) -> str:
        return f"Profiler(device={self.device}, input_size={self.input_size})"
