"""Tests for torch.compile integration."""

import pytest
import torch
import torch.nn as nn

from flashoptim.compilation import TorchCompiler


def _make_simple_model():
    """Create a simple model for compilation testing."""
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


class TestTorchCompiler:
    """Tests for TorchCompiler wrapper."""

    def test_init_default(self):
        compiler = TorchCompiler()
        assert compiler.mode == "default"
        assert compiler.backend == "inductor"
        assert compiler.disable is False

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            TorchCompiler(mode="invalid")

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            TorchCompiler(backend="invalid")

    def test_compile_disabled(self):
        model = _make_simple_model()
        compiler = TorchCompiler(disable=True)
        compiled = compiler.compile(model)
        assert compiled is model

    def test_compile(self):
        model = _make_simple_model()
        compiler = TorchCompiler(mode="default")
        compiled = compiler.compile(model)
        assert compiled is not None
        x = torch.randn(2, 32)
        out = compiled(x)
        assert out.shape == (2, 10)

    def test_warmup(self):
        model = _make_simple_model()
        compiler = TorchCompiler(disable=True)
        compiled = compiler.compile(model)
        sample = torch.randn(2, 32)
        compiler.warmup(compiled, sample, warmup_iterations=2)

    def test_benchmark(self):
        model = _make_simple_model()
        compiler = TorchCompiler(disable=True)
        compiled = compiler.compile(model)
        sample = torch.randn(2, 32)
        stats = compiler.benchmark(compiled, sample, n_iterations=5, warmup_iterations=2)
        assert "mean_ms" in stats
        assert "median_ms" in stats
        assert "throughput_fps" in stats
        assert stats["mean_ms"] > 0

    def test_compare(self):
        model = _make_simple_model()
        compiler = TorchCompiler(disable=True)
        sample = torch.randn(2, 32)
        result = compiler.compare(model, sample, n_iterations=5)
        assert "eager" in result
        assert "compiled" in result
        assert "speedup" in result

    def test_get_backend_info(self):
        info = TorchCompiler.get_backend_info()
        assert "torch_version" in info
        assert "compile_available" in info

    def test_compile_function(self):
        def my_fn(x):
            return x * 2 + 1

        compiler = TorchCompiler(disable=True)
        compiled_fn = compiler.compile_function(my_fn)
        result = compiled_fn(torch.tensor(5.0))
        assert result.item() == 11.0

    def test_repr(self):
        compiler = TorchCompiler(mode="max-autotune")
        r = repr(compiler)
        assert "TorchCompiler" in r
        assert "max-autotune" in r
