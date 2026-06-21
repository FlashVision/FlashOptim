"""Calibration runner for quantization range estimation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Calibrator:
    """Runs calibration data through a model to estimate quantization ranges.

    Collects activation statistics (min, max, histograms) at each layer
    to determine optimal quantization parameters.

    Args:
        num_samples: Number of calibration samples.
        batch_size: Batch size for calibration inference.
        device: Device for calibration ('cpu' or 'cuda').
    """

    def __init__(
        self,
        num_samples: int = 500,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._hooks: List[Any] = []
        self._statistics: Dict[str, Dict[str, torch.Tensor]] = {}

    def calibrate(
        self,
        model: nn.Module,
        dataloader: Optional[DataLoader] = None,
        data_path: Optional[str | Path] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Run calibration and collect layer statistics.

        Args:
            model: Model to calibrate.
            dataloader: Pre-built calibration DataLoader.
            data_path: Path to calibration data directory.

        Returns:
            Dictionary mapping layer names to their collected statistics
            (min, max, histogram bins).
        """
        self._statistics.clear()
        model = model.to(self.device).eval()
        self._register_hooks(model)

        try:
            if dataloader is not None:
                self._run_calibration_loader(model, dataloader)
            elif data_path is not None:
                self._run_calibration_path(model, data_path)
            else:
                raise ValueError("Either dataloader or data_path must be provided")
        finally:
            self._remove_hooks()

        return self._statistics

    def _run_calibration_loader(self, model: nn.Module, dataloader: DataLoader) -> None:
        """Run calibration using a DataLoader."""
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                if count >= self.num_samples:
                    break
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                inputs = inputs.to(self.device)
                model(inputs)
                count += inputs.size(0)

    def _run_calibration_path(self, model: nn.Module, data_path: str | Path) -> None:
        """Run calibration from an image directory."""
        import glob

        import numpy as np
        from PIL import Image

        data_path = Path(data_path)
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        image_files: List[str] = []
        for pat in patterns:
            image_files.extend(glob.glob(str(data_path / "**" / pat), recursive=True))
        image_files = image_files[: self.num_samples]

        with torch.no_grad():
            for i in range(0, len(image_files), self.batch_size):
                batch_paths = image_files[i : i + self.batch_size]
                tensors = []
                for img_path in batch_paths:
                    img = Image.open(img_path).convert("RGB")
                    img_np = np.array(img).astype(np.float32) / 255.0
                    tensor = torch.from_numpy(img_np).permute(2, 0, 1)
                    tensor = torch.nn.functional.interpolate(tensor.unsqueeze(0), size=(640, 640)).squeeze(0)
                    tensors.append(tensor)
                batch_tensor = torch.stack(tensors).to(self.device)
                model(batch_tensor)

    def _register_hooks(self, model: nn.Module) -> None:
        """Register forward hooks on quantizable layers."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, layer_name: str):
        """Create a forward hook that collects activation statistics."""

        def hook_fn(module, input, output):
            if layer_name not in self._statistics:
                self._statistics[layer_name] = {
                    "min": output.detach().min(),
                    "max": output.detach().max(),
                }
            else:
                self._statistics[layer_name]["min"] = torch.min(
                    self._statistics[layer_name]["min"], output.detach().min()
                )
                self._statistics[layer_name]["max"] = torch.max(
                    self._statistics[layer_name]["max"], output.detach().max()
                )

        return hook_fn

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_ranges(self) -> Dict[str, tuple]:
        """Get computed quantization ranges for each layer.

        Returns:
            Dictionary mapping layer names to (min, max) tuples.
        """
        return {name: (stats["min"].item(), stats["max"].item()) for name, stats in self._statistics.items()}
