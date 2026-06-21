"""Tests for GPTQ, AWQ, and SmoothQuant quantization methods."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from flashoptim.quantization import GPTQQuantizer, AWQQuantizer, SmoothQuantizer


def _make_simple_model():
    """Create a simple model for quantization testing."""
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )


def _make_calibration_loader(n_samples=20, input_dim=32):
    """Create calibration DataLoader."""
    data = torch.randn(n_samples, input_dim)
    labels = torch.randint(0, 10, (n_samples,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=4)


class TestGPTQQuantizer:
    """Tests for GPTQ quantization."""

    def test_init_default(self):
        gptq = GPTQQuantizer()
        assert gptq.bits == 4
        assert gptq.group_size == 128

    def test_init_custom(self):
        gptq = GPTQQuantizer(bits=8, group_size=64, act_order=False)
        assert gptq.bits == 8
        assert gptq.group_size == 64
        assert gptq.act_order is False

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="Unsupported bit-width"):
            GPTQQuantizer(bits=5)

    def test_quantize(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        gptq = GPTQQuantizer(bits=4, block_size=16, calibration_samples=10)
        quantized = gptq.quantize(model, loader)
        assert quantized is not None
        x = torch.randn(2, 32)
        out = quantized(x)
        assert out.shape == (2, 10)

    def test_stats(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        gptq = GPTQQuantizer(bits=4, block_size=16, calibration_samples=10)
        gptq.quantize(model, loader)
        stats = gptq.stats
        assert len(stats) > 0

    def test_repr(self):
        gptq = GPTQQuantizer(bits=4)
        r = repr(gptq)
        assert "GPTQQuantizer" in r
        assert "4" in r


class TestAWQQuantizer:
    """Tests for AWQ quantization."""

    def test_init_default(self):
        awq = AWQQuantizer()
        assert awq.bits == 4
        assert awq.auto_scale is True

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="Unsupported bit-width"):
            AWQQuantizer(bits=5)

    def test_quantize(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        awq = AWQQuantizer(bits=4, group_size=16, calibration_samples=10)
        quantized = awq.quantize(model, loader)
        assert quantized is not None
        x = torch.randn(2, 32)
        out = quantized(x)
        assert out.shape == (2, 10)

    def test_scales_computed(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        awq = AWQQuantizer(bits=4, calibration_samples=10)
        awq.quantize(model, loader)
        assert len(awq.scales) > 0

    def test_no_auto_scale(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        awq = AWQQuantizer(bits=4, auto_scale=False, calibration_samples=10)
        quantized = awq.quantize(model, loader)
        x = torch.randn(2, 32)
        out = quantized(x)
        assert out.shape == (2, 10)

    def test_repr(self):
        awq = AWQQuantizer(bits=4)
        r = repr(awq)
        assert "AWQQuantizer" in r


class TestSmoothQuantizer:
    """Tests for SmoothQuant quantization."""

    def test_init_default(self):
        sq = SmoothQuantizer()
        assert sq.alpha == 0.5
        assert sq.bits_weight == 8

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha must be"):
            SmoothQuantizer(alpha=1.5)

    def test_quantize(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        sq = SmoothQuantizer(alpha=0.5, calibration_samples=10)
        quantized = sq.quantize(model, loader)
        assert quantized is not None
        x = torch.randn(2, 32)
        out = quantized(x)
        assert out.shape == (2, 10)

    def test_smooth_only(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        sq = SmoothQuantizer(alpha=0.5, calibration_samples=10)
        smoothed = sq.smooth_only(model, loader)
        assert smoothed is not None
        x = torch.randn(2, 32)
        out = smoothed(x)
        assert out.shape == (2, 10)

    def test_smooth_scales(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        sq = SmoothQuantizer(alpha=0.5, calibration_samples=10)
        sq.quantize(model, loader)
        assert len(sq.smooth_scales) > 0

    def test_different_alpha(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        sq = SmoothQuantizer(alpha=0.75, calibration_samples=10)
        quantized = sq.quantize(model, loader)
        x = torch.randn(2, 32)
        out = quantized(x)
        assert out.shape == (2, 10)

    def test_repr(self):
        sq = SmoothQuantizer(alpha=0.5)
        r = repr(sq)
        assert "SmoothQuantizer" in r
