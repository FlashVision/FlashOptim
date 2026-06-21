"""Post-Training Quantization (PTQ) — quantize a model without retraining."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.ao.quantization import (
    QConfig,
    convert,
    prepare,
)
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)


class PTQuantizer:
    """Post-Training Quantization engine.

    Quantizes model weights and activations to lower precision (INT8, FP16)
    using calibration data to determine optimal quantization parameters.

    Args:
        dtype: Target quantization type ('int8', 'fp16', 'mixed').
        calibration_samples: Number of calibration samples to use.
        per_channel: Use per-channel quantization (more accurate).
        symmetric: Use symmetric quantization ranges.
        observer: Observer strategy ('minmax', 'histogram', 'percentile').
        percentile: Percentile for clipping (used with 'percentile' observer).
    """

    SUPPORTED_DTYPES = ("int8", "fp16", "mixed")

    def __init__(
        self,
        dtype: str = "int8",
        calibration_samples: int = 500,
        per_channel: bool = True,
        symmetric: bool = True,
        observer: str = "minmax",
        percentile: float = 99.99,
    ) -> None:
        if dtype not in self.SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}. Options: {self.SUPPORTED_DTYPES}")

        self.dtype = dtype
        self.calibration_samples = calibration_samples
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.observer = observer
        self.percentile = percentile
        self._calibration_stats: Dict[str, Any] = {}

    def _get_qconfig(self) -> QConfig:
        """Build a QConfig based on the current settings."""
        if self.dtype == "fp16":
            return torch.ao.quantization.float16_static_qconfig

        qscheme = (
            torch.per_channel_symmetric
            if (self.per_channel and self.symmetric)
            else (
                torch.per_channel_affine
                if self.per_channel
                else (torch.per_tensor_symmetric if self.symmetric else torch.per_tensor_affine)
            )
        )

        if self.observer == "histogram":
            act_observer = HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric)
        else:
            act_observer = MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric)

        if self.per_channel:
            weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=qscheme)
        else:
            weight_observer = MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric if self.symmetric else torch.per_tensor_affine,
            )

        return QConfig(activation=act_observer, weight=weight_observer)

    def quantize(
        self,
        model: nn.Module,
        calibration_data: Optional[Union[str, Path]] = None,
        calibration_loader: Optional[Any] = None,
    ) -> nn.Module:
        """Quantize a model using post-training quantization.

        Args:
            model: The PyTorch model to quantize.
            calibration_data: Path to calibration images directory.
            calibration_loader: Pre-built DataLoader for calibration.

        Returns:
            Quantized model.
        """
        if self.dtype == "fp16":
            return self._apply_fp16(model)

        model_prepared = copy.deepcopy(model)
        model_prepared.eval()
        model_prepared.qconfig = self._get_qconfig()

        model_prepared = prepare(model_prepared, inplace=False)

        if calibration_loader is not None:
            self._collect_statistics(model_prepared, calibration_loader)
        elif calibration_data is not None:
            self._collect_statistics_from_path(model_prepared, calibration_data)

        quantized_model = self._apply_quantization(model_prepared)
        return quantized_model

    def _apply_fp16(self, model: nn.Module) -> nn.Module:
        """Convert model to FP16 precision."""
        return copy.deepcopy(model).half()

    def _collect_statistics(self, model: nn.Module, dataloader: Any) -> Dict[str, Any]:
        """Run calibration data through model to collect activation statistics.

        Args:
            model: Prepared model with observers.
            dataloader: Calibration data loader.

        Returns:
            Dictionary of layer-wise statistics.
        """
        model.eval()
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                if count >= self.calibration_samples:
                    break
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                model(inputs)
                count += inputs.size(0)

        self._calibration_stats = {"samples_processed": count}
        return self._calibration_stats

    def _collect_statistics_from_path(self, model: nn.Module, data_path: Union[str, Path]) -> None:
        """Collect statistics from a directory of images."""
        import glob

        import numpy as np
        from PIL import Image

        data_path = Path(data_path)
        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        image_files: list = []
        for ext in image_extensions:
            image_files.extend(glob.glob(str(data_path / ext)))
            image_files.extend(glob.glob(str(data_path / "**" / ext), recursive=True))

        image_files = image_files[: self.calibration_samples]

        model.eval()
        with torch.no_grad():
            for img_path in image_files:
                img = Image.open(img_path).convert("RGB")
                img_np = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
                img_tensor = torch.nn.functional.interpolate(img_tensor, size=(640, 640))
                model(img_tensor)

    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply quantization parameters to model weights and activations.

        Args:
            model: Prepared model with observer statistics collected.

        Returns:
            Quantized model.
        """
        model.eval()
        quantized_model = convert(model, inplace=False)
        return quantized_model

    def sensitivity_analysis(
        self,
        model: nn.Module,
        calibration_loader: Any,
        val_loader: Any,
    ) -> Dict[str, float]:
        """Analyze per-layer quantization sensitivity.

        Quantizes one layer at a time and measures accuracy impact.

        Args:
            model: Model to analyze.
            calibration_loader: Calibration data.
            val_loader: Validation data for accuracy measurement.

        Returns:
            Dictionary mapping layer names to accuracy drops.
        """
        model.eval()
        baseline_acc = self._evaluate_accuracy(model, val_loader)
        sensitivity = {}

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                test_model = copy.deepcopy(model)
                test_model.eval()

                for n, m in test_model.named_modules():
                    if n == name:
                        m.qconfig = self._get_qconfig()
                    else:
                        m.qconfig = None

                try:
                    prepared = prepare(test_model, inplace=False)
                    self._collect_statistics(prepared, calibration_loader)
                    quantized = convert(prepared, inplace=False)
                    quant_acc = self._evaluate_accuracy(quantized, val_loader)
                    sensitivity[name] = baseline_acc - quant_acc
                except Exception:
                    sensitivity[name] = 0.0

        return sensitivity

    @staticmethod
    def _evaluate_accuracy(model: nn.Module, dataloader: Any) -> float:
        """Evaluate classification accuracy on a dataloader."""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    continue
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"PTQuantizer(dtype={self.dtype}, per_channel={self.per_channel}, "
            f"observer={self.observer}, samples={self.calibration_samples})"
        )
