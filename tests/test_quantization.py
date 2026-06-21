"""Tests for quantization module — PTQuantizer and QATTrainer."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from flashoptim.quantization import PTQuantizer, QATTrainer, Calibrator, MinMaxObserver, HistogramObserver


def _make_simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )


def _make_calibration_loader(num_samples=20, img_size=32, num_classes=10):
    """Create a simple calibration DataLoader."""
    images = torch.randn(num_samples, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=4)


class TestPTQuantizer:
    """Tests for Post-Training Quantization."""

    def test_init_default(self):
        quantizer = PTQuantizer()
        assert quantizer.dtype == "int8"
        assert quantizer.per_channel is True
        assert quantizer.symmetric is True

    def test_init_custom(self):
        quantizer = PTQuantizer(dtype="fp16", per_channel=False, symmetric=False)
        assert quantizer.dtype == "fp16"
        assert quantizer.per_channel is False
        assert quantizer.symmetric is False

    def test_invalid_dtype(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            PTQuantizer(dtype="int4")

    def test_repr(self):
        quantizer = PTQuantizer()
        r = repr(quantizer)
        assert "PTQuantizer" in r
        assert "int8" in r

    def test_quantize_fp16(self):
        quantizer = PTQuantizer(dtype="fp16")
        model = _make_simple_model()
        result = quantizer.quantize(model)
        first_param = next(result.parameters())
        assert first_param.dtype == torch.float16

    def test_quantize_int8_with_loader(self):
        quantizer = PTQuantizer(dtype="int8")
        model = _make_simple_model()
        loader = _make_calibration_loader()
        result = quantizer.quantize(model, calibration_loader=loader)
        assert result is not None


class TestQATTrainer:
    """Tests for Quantization-Aware Training."""

    def test_init_default(self):
        trainer = QATTrainer()
        assert trainer is not None

    def test_repr(self):
        trainer = QATTrainer()
        r = repr(trainer)
        assert "QATTrainer" in r

    def test_convert(self):
        trainer = QATTrainer()
        model = _make_simple_model()
        model.train()
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
        prepared = torch.ao.quantization.prepare_qat(model)
        dummy = torch.randn(2, 3, 32, 32)
        prepared(dummy)
        converted = trainer.convert(prepared)
        assert converted is not None


class TestCalibrator:
    """Tests for the calibration engine."""

    def test_init(self):
        calibrator = Calibrator(num_samples=100, batch_size=16)
        assert calibrator.num_samples == 100
        assert calibrator.batch_size == 16

    def test_calibrate_with_loader(self):
        calibrator = Calibrator(num_samples=20)
        model = _make_simple_model()
        loader = _make_calibration_loader()
        stats = calibrator.calibrate(model, dataloader=loader)
        assert len(stats) > 0
        for name, layer_stats in stats.items():
            assert "min" in layer_stats
            assert "max" in layer_stats

    def test_get_ranges(self):
        calibrator = Calibrator(num_samples=20)
        model = _make_simple_model()
        loader = _make_calibration_loader()
        calibrator.calibrate(model, dataloader=loader)
        ranges = calibrator.get_ranges()
        assert len(ranges) > 0
        for name, (min_val, max_val) in ranges.items():
            assert min_val <= max_val


class TestObservers:
    """Tests for quantization observers."""

    def test_minmax_observer_forward(self):
        obs = MinMaxObserver()
        x = torch.randn(4, 16, 8, 8)
        out = obs(x)
        assert torch.equal(x, out)
        assert obs.min_val.item() <= x.min().item()
        assert obs.max_val.item() >= x.max().item()

    def test_minmax_observer_qparams(self):
        obs = MinMaxObserver()
        obs(torch.randn(4, 16, 8, 8))
        scale, zp = obs.compute_qparams()
        assert scale.item() > 0

    def test_histogram_observer_forward(self):
        obs = HistogramObserver(bins=256)
        x = torch.randn(4, 16, 8, 8)
        out = obs(x)
        assert torch.equal(x, out)
        assert obs.histogram.sum() > 0

    def test_histogram_observer_entropy(self):
        obs = HistogramObserver(bins=256, method="entropy")
        obs(torch.randn(10, 16, 8, 8))
        obs(torch.randn(10, 16, 8, 8))
        scale, zp = obs.compute_qparams()
        assert scale.item() > 0

    def test_histogram_observer_mse(self):
        obs = HistogramObserver(bins=256, method="mse")
        obs(torch.randn(10, 16, 8, 8))
        obs(torch.randn(10, 16, 8, 8))
        scale, zp = obs.compute_qparams()
        assert scale.item() > 0
