"""Predictor for running inference with optimized models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


class Predictor:
    """Run inference using an optimized FlashOptim model.

    Supports PyTorch and ONNX model formats with automatic preprocessing.

    Args:
        model_path: Path to the optimized model file (.pth or .onnx).
        device: Inference device ('cpu', 'cuda').
        input_size: Expected input size (H, W).
        conf_threshold: Confidence threshold for predictions.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: Optional[str] = None,
        input_size: tuple = (640, 640),
        conf_threshold: float = 0.25,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.model = self._load_model()

    def _load_model(self) -> Any:
        """Load model from file based on extension.

        Returns:
            Loaded model (PyTorch nn.Module or ONNX session).
        """
        suffix = self.model_path.suffix.lower()

        if suffix in (".pth", ".pt"):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, nn.Module):
                model = checkpoint
            elif isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    raise ValueError(
                        "Checkpoint contains 'model_state_dict' but no model architecture. "
                        "Provide the model class and load state_dict separately."
                    )
                elif "model" in checkpoint:
                    model = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    raise ValueError(
                        "Checkpoint contains 'state_dict' but no model architecture. "
                        "Provide the model class and load state_dict separately."
                    )
                else:
                    raise ValueError(f"Unsupported checkpoint format. Keys: {list(checkpoint.keys())}")
            else:
                raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")

            model = model.to(self.device).eval()
            return model

        elif suffix == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError:
                raise ImportError("onnxruntime is required for ONNX inference: pip install onnxruntime")

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            session = ort.InferenceSession(str(self.model_path), providers=providers)
            return session

        else:
            raise ValueError(f"Unsupported model format: {suffix}")

    def predict(
        self,
        source: Union[str, Path, np.ndarray, torch.Tensor],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Run prediction on input source.

        Args:
            source: Image path, numpy array, or tensor.
            **kwargs: Additional prediction parameters.

        Returns:
            List of prediction dictionaries with keys like
            'boxes', 'scores', 'labels'.
        """
        input_tensor = self._preprocess(source)

        if isinstance(self.model, nn.Module):
            with torch.no_grad():
                output = self.model(input_tensor)
            return self._postprocess_torch(output)
        else:
            import onnxruntime as ort

            input_name = self.model.get_inputs()[0].name
            output = self.model.run(None, {input_name: input_tensor.cpu().numpy()})
            return self._postprocess_onnx(output)

    def _preprocess(self, source: Union[str, Path, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Preprocess input to a normalized tensor."""
        if isinstance(source, torch.Tensor):
            tensor = source
        elif isinstance(source, np.ndarray):
            if source.ndim == 3 and source.shape[2] == 3:
                tensor = torch.from_numpy(source.astype(np.float32) / 255.0).permute(2, 0, 1)
            else:
                tensor = torch.from_numpy(source.astype(np.float32))
        elif isinstance(source, (str, Path)):
            from PIL import Image
            img = Image.open(source).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        tensor = torch.nn.functional.interpolate(tensor, size=self.input_size, mode="bilinear", align_corners=False)
        return tensor.to(self.device)

    def _postprocess_torch(self, output: Any) -> List[Dict[str, Any]]:
        """Postprocess PyTorch model output."""
        if isinstance(output, torch.Tensor):
            if output.ndim == 2:
                scores, labels = output.softmax(dim=-1).max(dim=-1)
                return [{"scores": scores.cpu().tolist(), "labels": labels.cpu().tolist()}]
            return [{"raw_output": output.cpu()}]
        elif isinstance(output, (list, tuple)):
            return [{"raw_output": [o.cpu() if isinstance(o, torch.Tensor) else o for o in output]}]
        return [{"raw_output": output}]

    def _postprocess_onnx(self, output: list) -> List[Dict[str, Any]]:
        """Postprocess ONNX inference output."""
        return [{"raw_output": output}]

    def warmup(self, runs: int = 10) -> None:
        """Warm up the model with dummy inputs.

        Args:
            runs: Number of warmup iterations.
        """
        dummy = torch.randn(1, 3, *self.input_size).to(self.device)
        for _ in range(runs):
            with torch.no_grad():
                if isinstance(self.model, nn.Module):
                    self.model(dummy)

    def benchmark(self, runs: int = 100) -> Dict[str, float]:
        """Benchmark inference latency.

        Args:
            runs: Number of inference iterations.

        Returns:
            Dictionary with 'mean_ms', 'std_ms', 'fps' keys.
        """
        import time

        self.warmup(runs=10)

        dummy = torch.randn(1, 3, *self.input_size).to(self.device)
        timings = []

        with torch.no_grad():
            for _ in range(runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()

                if isinstance(self.model, nn.Module):
                    self.model(dummy)
                else:
                    input_name = self.model.get_inputs()[0].name
                    self.model.run(None, {input_name: dummy.cpu().numpy()})

                if self.device == "cuda":
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000
                timings.append(elapsed_ms)

        timings_tensor = torch.tensor(timings)
        mean_ms = timings_tensor.mean().item()
        std_ms = timings_tensor.std().item()
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        return {"mean_ms": round(mean_ms, 3), "std_ms": round(std_ms, 3), "fps": round(fps, 1)}
