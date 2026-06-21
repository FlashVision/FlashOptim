"""Exporter for converting optimized models to deployment formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class Exporter:
    """Export optimized PyTorch models to deployment formats.

    Supported formats: ONNX, TensorRT (planned), OpenVINO (planned), CoreML (planned).

    Args:
        model: The PyTorch model to export.
        input_size: Model input size (H, W).
        batch_size: Export batch size (1 for dynamic).
        opset_version: ONNX opset version.
        simplify: Whether to simplify the ONNX graph.
    """

    SUPPORTED_FORMATS = ("onnx", "tensorrt", "openvino", "coreml")

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        input_size: Tuple[int, int] = (640, 640),
        batch_size: int = 1,
        opset_version: int = 17,
        simplify: bool = True,
    ) -> None:
        self.model = model
        self.input_size = input_size
        self.batch_size = batch_size
        self.opset_version = opset_version
        self.simplify = simplify

    def export(
        self,
        output_path: str | Path,
        format: str = "onnx",
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        **kwargs: Any,
    ) -> str:
        """Export model to the specified format.

        Args:
            output_path: Path to save the exported model.
            format: Export format ('onnx', 'tensorrt', 'openvino', 'coreml').
            dynamic_axes: Optional dynamic axis specification for ONNX.
            **kwargs: Additional export parameters.

        Returns:
            Path to the exported model file.

        Raises:
            ValueError: If format is not supported.
            NotImplementedError: If the format is not yet implemented.
        """
        format = format.lower()
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {format}. Supported: {self.SUPPORTED_FORMATS}"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "onnx":
            return self._export_onnx(output_path, dynamic_axes, **kwargs)
        else:
            raise NotImplementedError(
                f"Export to '{format}' is planned for a future release."
            )

    def _export_onnx(
        self,
        output_path: Path,
        dynamic_axes: Optional[Dict] = None,
        **kwargs: Any,
    ) -> str:
        """Export model to ONNX format.

        Args:
            output_path: Output file path.
            dynamic_axes: Dynamic axis specification.

        Returns:
            Path to exported ONNX file.
        """
        if self.model is None:
            raise ValueError("Model must be set before export")

        self.model.eval()
        dummy_input = torch.randn(
            self.batch_size, 3, *self.input_size, device=next(self.model.parameters()).device
        )

        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".onnx")

        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            opset_version=self.opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

        if self.simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify

                onnx_model = onnx.load(str(output_path))
                simplified_model, check = onnx_simplify(onnx_model)
                if check:
                    onnx.save(simplified_model, str(output_path))
            except ImportError:
                pass

        return str(output_path)

    @staticmethod
    def validate_onnx(model_path: str | Path) -> bool:
        """Validate an exported ONNX model.

        Args:
            model_path: Path to the ONNX model file.

        Returns:
            True if the model is valid.
        """
        try:
            import onnx

            onnx_model = onnx.load(str(model_path))
            onnx.checker.check_model(onnx_model)
        except ImportError:
            raise ImportError("onnx is required for validation: pip install onnx")

        try:
            import numpy as np
            import onnxruntime as ort

            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            input_info = session.get_inputs()[0]
            shape = [d if isinstance(d, int) else 1 for d in input_info.shape]
            dummy = np.random.randn(*shape).astype(np.float32)
            session.run(None, {input_info.name: dummy})
        except ImportError:
            pass

        return True
