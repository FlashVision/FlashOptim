"""Architecture evaluator for NAS candidates."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Evaluator:
    """Evaluate candidate architectures for NAS search.

    Performs proxy training (few-epoch training) and measures accuracy,
    latency, FLOPs, and parameter count for each candidate.

    Args:
        train_loader: DataLoader for proxy training.
        val_loader: DataLoader for validation.
        proxy_epochs: Number of proxy training epochs.
        device: Device for evaluation ('cuda' or 'cpu').
        max_flops: Upper bound on FLOPs (reject architectures exceeding this).
        max_params: Upper bound on parameter count.
    """

    def __init__(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        proxy_epochs: int = 5,
        device: Optional[str] = None,
        max_flops: Optional[float] = None,
        max_params: Optional[float] = None,
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.proxy_epochs = proxy_epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_flops = max_flops
        self.max_params = max_params

    def evaluate(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a candidate architecture.

        Builds the architecture, checks constraints, runs proxy training,
        and returns a comprehensive evaluation result.

        Args:
            architecture: Architecture specification dict with keys
                          'channels', 'kernel_sizes', 'depths', 'operations'.

        Returns:
            Dictionary with keys: 'accuracy', 'latency_ms', 'params',
            'flops', 'score', 'feasible'.
        """
        model = self._build_model(architecture)
        params = sum(p.numel() for p in model.parameters())
        flops = self._estimate_flops(model)

        feasible = True
        if self.max_params and params > self.max_params:
            feasible = False
        if self.max_flops and flops > self.max_flops:
            feasible = False

        if not feasible:
            return {
                "accuracy": 0.0,
                "latency_ms": float("inf"),
                "params": params,
                "flops": flops,
                "score": 0.0,
                "feasible": False,
            }

        accuracy = self._proxy_train(model) if self.train_loader else 0.0
        latency_ms = self._measure_latency(model)

        score = self._compute_score(accuracy, latency_ms, params, flops)

        return {
            "accuracy": accuracy,
            "latency_ms": latency_ms,
            "params": params,
            "flops": flops,
            "score": score,
            "feasible": True,
        }

    def _build_model(self, architecture: Dict[str, Any]) -> nn.Module:
        """Build an nn.Module from an architecture specification.

        Creates a sequential model with the specified channels, kernel sizes,
        depths, and operations.

        Args:
            architecture: Architecture specification dict.

        Returns:
            Constructed nn.Module.
        """
        layers: List[nn.Module] = []
        in_channels = 3

        for i, (ch, ks, depth, op) in enumerate(
            zip(
                architecture["channels"],
                architecture["kernel_sizes"],
                architecture["depths"],
                architecture["operations"],
            )
        ):
            for d in range(depth):
                c_in = in_channels if d == 0 else ch
                layers.append(self._build_op(op, c_in, ch, ks))
                layers.append(nn.BatchNorm2d(ch))
                layers.append(nn.ReLU(inplace=True))
            in_channels = ch

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels, 10))

        model = nn.Sequential(*layers)
        return model.to(self.device)

    @staticmethod
    def _build_op(op: str, in_ch: int, out_ch: int, ks: int) -> nn.Module:
        """Build a single operation module.

        Args:
            op: Operation type ('conv', 'dwconv', 'mbconv', 'skip').
            in_ch: Input channels.
            out_ch: Output channels.
            ks: Kernel size.

        Returns:
            nn.Module implementing the operation.
        """
        padding = ks // 2
        if op == "conv":
            return nn.Conv2d(in_ch, out_ch, ks, padding=padding, bias=False)
        elif op == "dwconv":
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, ks, padding=padding, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
            )
        elif op == "mbconv":
            mid_ch = in_ch * 4
            return nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, mid_ch, ks, padding=padding, groups=mid_ch, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            )
        elif op == "skip":
            if in_ch == out_ch:
                return nn.Identity()
            return nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            return nn.Conv2d(in_ch, out_ch, ks, padding=padding, bias=False)

    def _proxy_train(self, model: nn.Module) -> float:
        """Run proxy training and return validation accuracy.

        Args:
            model: Model to train.

        Returns:
            Validation accuracy after proxy training (0.0 to 1.0).
        """
        if self.train_loader is None:
            return 0.0

        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.proxy_epochs):
            for batch_idx, (images, targets) in enumerate(self.train_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        return self._validate(model) if self.val_loader else 0.0

    def _validate(self, model: nn.Module) -> float:
        """Compute validation accuracy.

        Args:
            model: Trained model.

        Returns:
            Top-1 accuracy (0.0 to 1.0).
        """
        if self.val_loader is None:
            return 0.0

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return correct / total if total > 0 else 0.0

    def _measure_latency(self, model: nn.Module, input_size: tuple = (1, 3, 32, 32)) -> float:
        """Measure model inference latency.

        Args:
            model: Model to profile.
            input_size: Input tensor shape.

        Returns:
            Average latency in milliseconds.
        """
        model.eval()
        dummy = torch.randn(*input_size, device=self.device)
        warmup = 5
        runs = 20

        with torch.no_grad():
            for _ in range(warmup):
                model(dummy)

            if self.device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(runs):
                model(dummy)
            if self.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

        return (elapsed / runs) * 1000.0

    @staticmethod
    def _estimate_flops(model: nn.Module, input_size: tuple = (1, 3, 32, 32)) -> float:
        """Rough FLOPs estimate based on parameter count heuristic.

        Args:
            model: Model to estimate.
            input_size: Input tensor shape.

        Returns:
            Estimated FLOPs count.
        """
        total_params = sum(p.numel() for p in model.parameters())
        h, w = input_size[2], input_size[3]
        return total_params * h * w * 2.0

    @staticmethod
    def _compute_score(
        accuracy: float,
        latency_ms: float,
        params: int,
        flops: float,
    ) -> float:
        """Compute a composite score balancing accuracy and efficiency.

        Args:
            accuracy: Model accuracy (0-1).
            latency_ms: Inference latency in ms.
            params: Parameter count.
            flops: FLOPs estimate.

        Returns:
            Composite score (higher is better).
        """
        latency_penalty = max(0.0, 1.0 - latency_ms / 100.0)
        param_penalty = max(0.0, 1.0 - params / 1e7)
        return accuracy * 0.6 + latency_penalty * 0.2 + param_penalty * 0.2

    def __repr__(self) -> str:
        return (
            f"Evaluator(proxy_epochs={self.proxy_epochs}, device={self.device}, "
            f"max_flops={self.max_flops}, max_params={self.max_params})"
        )
