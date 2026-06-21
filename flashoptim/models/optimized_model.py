"""FlashOptim — Main model wrapper for optimization pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


class FlashOptim(nn.Module):
    """Main FlashOptim model wrapper.

    Wraps a PyTorch model and provides optimization-aware methods for
    quantization, pruning, distillation, and export.

    Args:
        model: Path to a model checkpoint or an nn.Module instance.
        device: Device to load the model on.
        task: Model task type ('detect', 'classify', 'segment').

    Example:
        >>> model = FlashOptim("pretrained/model.pth")
        >>> model.info()
        >>> model.export("output.onnx")
    """

    def __init__(
        self,
        model: Union[str, Path, nn.Module],
        device: Optional[str] = None,
        task: str = "detect",
    ) -> None:
        super().__init__()
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        self._compression_ratio: float = 1.0
        self._original_size: Optional[float] = None

        if isinstance(model, nn.Module):
            self.model = model
        else:
            self.model = self._load_checkpoint(model)

        self.to(self.device_name)

    def _load_checkpoint(self, path: Union[str, Path]) -> nn.Module:
        """Load model from checkpoint file.

        Handles standard PyTorch checkpoint formats:
        - Raw nn.Module (torch.save(model, path))
        - Dict with 'model' key containing nn.Module
        - Dict with 'state_dict' or 'model_state_dict' key (requires model class)

        Args:
            path: Path to the .pth checkpoint.

        Returns:
            Loaded nn.Module.

        Raises:
            FileNotFoundError: If checkpoint does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device_name, weights_only=False)

        if isinstance(checkpoint, nn.Module):
            return checkpoint

        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                model_data = checkpoint["model"]
                if isinstance(model_data, nn.Module):
                    return model_data
                elif isinstance(model_data, dict):
                    raise ValueError(
                        "Checkpoint has 'model' key with a state_dict. "
                        "Please provide the model architecture separately."
                    )
                return model_data

            if "state_dict" in checkpoint or "model_state_dict" in checkpoint:
                state_key = "state_dict" if "state_dict" in checkpoint else "model_state_dict"
                raise ValueError(
                    f"Checkpoint contains '{state_key}' but not a full model. "
                    f"Use FlashOptim(your_model_instance) and load state_dict separately: "
                    f"model.load_state_dict(torch.load(path)['{state_key}'])"
                )

            raise ValueError(
                f"Unsupported checkpoint dict format. "
                f"Keys found: {list(checkpoint.keys())}. "
                f"Expected 'model', 'state_dict', or 'model_state_dict'."
            )

        raise ValueError(
            f"Unsupported checkpoint type: {type(checkpoint)}. "
            f"Expected nn.Module or dict."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the wrapped model.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Model output tensor.
        """
        return self.model(x)

    @property
    def compression_ratio(self) -> float:
        """Return the compression ratio after optimization."""
        return self._compression_ratio

    def info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model stats (params, size, layers, etc.).
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size_mb = sum(p.nelement() * p.element_size() for p in self.parameters()) / (1024 * 1024)

        info = {
            "task": self.task,
            "device": self.device_name,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "size_mb": round(size_mb, 2),
            "compression_ratio": self._compression_ratio,
            "num_layers": len(list(self.modules())),
        }
        return info

    def export(
        self,
        output_path: str | Path,
        format: str = "onnx",
        input_size: Tuple[int, int] = (640, 640),
        **kwargs: Any,
    ) -> str:
        """Export the optimized model.

        Args:
            output_path: Path to save the exported model.
            format: Export format ('onnx', 'tensorrt', 'openvino').
            input_size: Model input dimensions (H, W).
            **kwargs: Additional export parameters.

        Returns:
            Path to the exported model.

        Raises:
            NotImplementedError: Export pending full implementation.
        """
        from flashoptim.engine.exporter import Exporter

        exporter = Exporter(model=self.model, input_size=input_size)
        return exporter.export(output_path, format=format, **kwargs)

    def sparsity(self) -> float:
        """Calculate the overall model sparsity (fraction of zero weights).

        Returns:
            Sparsity ratio between 0.0 and 1.0.
        """
        total = 0
        zeros = 0
        for p in self.parameters():
            total += p.numel()
            zeros += (p == 0).sum().item()
        return zeros / total if total > 0 else 0.0

    def __repr__(self) -> str:
        info = self.info()
        return (
            f"FlashOptim(task={info['task']}, params={info['total_params']:,}, "
            f"size={info['size_mb']}MB, compression={info['compression_ratio']}x)"
        )
