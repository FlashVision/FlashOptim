"""AutoOptimizer — one-click model optimization for target deployments."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn


class AutoOptimizer:
    """Automated model optimization pipeline.

    Selects and applies the best combination of quantization, pruning,
    and distillation techniques based on the target deployment platform.

    Args:
        target: Deployment target ('mobile', 'edge', 'server').
        quantize: Enable quantization. Defaults to auto-detect from target.
        prune: Enable pruning. Defaults to auto-detect from target.
        sparsity: Target pruning sparsity (only used if prune=True).
        dtype: Quantization dtype ('int8', 'fp16').
    """

    TARGET_PROFILES = {
        "mobile": {"quantize": True, "prune": True, "sparsity": 0.6, "dtype": "int8"},
        "edge": {"quantize": True, "prune": True, "sparsity": 0.4, "dtype": "int8"},
        "server": {"quantize": True, "prune": False, "sparsity": 0.0, "dtype": "fp16"},
    }

    def __init__(
        self,
        target: str = "edge",
        quantize: Optional[bool] = None,
        prune: Optional[bool] = None,
        sparsity: Optional[float] = None,
        dtype: Optional[str] = None,
    ) -> None:
        if target not in self.TARGET_PROFILES:
            raise ValueError(f"Unknown target: {target}. Options: {list(self.TARGET_PROFILES.keys())}")

        profile = self.TARGET_PROFILES[target]
        self.target = target
        self.quantize = quantize if quantize is not None else profile["quantize"]
        self.prune = prune if prune is not None else profile["prune"]
        self.sparsity = sparsity if sparsity is not None else profile["sparsity"]
        self.dtype = dtype or profile["dtype"]
        self._report: Dict[str, Any] = {}

    def optimize(self, model: nn.Module, **kwargs: Any) -> nn.Module:
        """Run the full optimization pipeline.

        Applies pruning and quantization in sequence, recording metrics
        at each stage for the optimization report.

        Args:
            model: PyTorch model to optimize.
            **kwargs: Additional parameters passed to individual optimizers.

        Returns:
            Optimized model.
        """
        original_params = sum(p.numel() for p in model.parameters())
        original_size = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)

        self._report = {
            "target": self.target,
            "original_params": original_params,
            "original_size_mb": round(original_size, 2),
            "steps": [],
        }

        if self.prune:
            model = self._apply_pruning(model, **kwargs)

        if self.quantize:
            model = self._apply_quantization(model, **kwargs)

        final_params = sum(p.numel() for p in model.parameters())
        final_size = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)

        self._report["final_params"] = final_params
        self._report["final_size_mb"] = round(final_size, 2)
        self._report["compression_ratio"] = round(original_size / final_size if final_size > 0 else 1.0, 2)

        return model

    def _apply_pruning(self, model: nn.Module, **kwargs: Any) -> nn.Module:
        """Apply pruning optimization step.

        Args:
            model: Model to prune.
            **kwargs: Extra arguments.

        Returns:
            Pruned model.
        """
        from flashoptim.pruning import UnstructuredPruner

        pruner = UnstructuredPruner(sparsity=self.sparsity)
        model = pruner.prune(model)
        sparsity_stats = pruner.get_sparsity(model)

        self._report["steps"].append(
            {
                "step": "pruning",
                "method": "unstructured_magnitude",
                "target_sparsity": self.sparsity,
                "actual_sparsity": sparsity_stats.get("global", 0.0),
            }
        )

        return model

    def _apply_quantization(self, model: nn.Module, **kwargs: Any) -> nn.Module:
        """Apply quantization optimization step.

        Args:
            model: Model to quantize.
            **kwargs: Extra arguments (calibration_loader for PTQ calibration).

        Returns:
            Quantized model.
        """
        from flashoptim.quantization import PTQuantizer

        quantizer = PTQuantizer(dtype=self.dtype)
        calibration_loader = kwargs.get("calibration_loader")

        try:
            model = quantizer.quantize(
                model,
                calibration_loader=calibration_loader,
            )
            self._report["steps"].append(
                {
                    "step": "quantization",
                    "method": "ptq",
                    "dtype": self.dtype,
                    "status": "completed",
                }
            )
        except Exception as e:
            self._report["steps"].append(
                {
                    "step": "quantization",
                    "method": "ptq",
                    "dtype": self.dtype,
                    "status": f"failed — {str(e)}",
                }
            )

        return model

    def get_report(self) -> Dict[str, Any]:
        """Get the optimization report.

        Returns:
            Dictionary summarizing all optimization steps and results.
        """
        return self._report

    def __repr__(self) -> str:
        return (
            f"AutoOptimizer(target={self.target}, quantize={self.quantize}, "
            f"prune={self.prune}, sparsity={self.sparsity}, dtype={self.dtype})"
        )
