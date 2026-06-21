"""Quantization-Aware Training (QAT) — train with simulated quantization."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class QATTrainer:
    """Quantization-Aware Training engine.

    Inserts fake quantization nodes into the model graph and trains with
    simulated quantization to recover accuracy lost during PTQ.

    Args:
        dtype: Target quantization type ('int8', 'fp16').
        epochs: Number of QAT training epochs.
        lr: Learning rate for QAT fine-tuning.
        observer: Observer type for range estimation.
        freeze_bn_epochs: Epochs after which to freeze BatchNorm stats.
        device: Training device.
    """

    def __init__(
        self,
        dtype: str = "int8",
        epochs: int = 10,
        lr: float = 0.001,
        observer: str = "minmax",
        freeze_bn_epochs: int = 5,
        device: Optional[str] = None,
    ) -> None:
        self.dtype = dtype
        self.epochs = epochs
        self.lr = lr
        self.observer = observer
        self.freeze_bn_epochs = freeze_bn_epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for QAT by inserting fake quantization modules.

        Args:
            model: The float model to prepare.

        Returns:
            QAT-ready model with fake quant observers.
        """
        from torch.ao.quantization import get_default_qat_qconfig, prepare_qat

        model.train()
        model.qconfig = get_default_qat_qconfig("x86")
        model = (
            torch.ao.quantization.fuse_modules_qat(model, self._get_fusable_modules(model), inplace=True)
            if self._get_fusable_modules(model)
            else model
        )
        prepared = prepare_qat(model, inplace=False)
        return prepared

    def _get_fusable_modules(self, model: nn.Module) -> list:
        """Detect Conv-BN-ReLU patterns that can be fused."""
        modules = dict(model.named_modules())
        fuse_list = []
        module_names = list(modules.keys())

        for i, name in enumerate(module_names):
            m = modules[name]
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                group = [name]
                if i + 1 < len(module_names):
                    next_m = modules[module_names[i + 1]]
                    if isinstance(next_m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        group.append(module_names[i + 1])
                        if i + 2 < len(module_names):
                            next_next_m = modules[module_names[i + 2]]
                            if isinstance(next_next_m, nn.ReLU):
                                group.append(module_names[i + 2])
                if len(group) > 1:
                    fuse_list.append(group)
        return fuse_list

    def train(
        self,
        model: nn.Module,
        train_data: Optional[str] = None,
        val_data: Optional[str] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Run QAT training loop.

        Args:
            model: Model to train with QAT (should be prepared).
            train_data: Path to training data.
            val_data: Path to validation data.
            train_loader: Pre-built training DataLoader.
            val_loader: Pre-built validation DataLoader.

        Returns:
            QAT-trained model ready for conversion to quantized model.
        """
        if train_loader is None:
            raise ValueError("train_loader is required for QAT training")

        model = model.to(self.device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        for epoch in range(self.epochs):
            model.train()
            if epoch >= self.freeze_bn_epochs:
                model.apply(torch.ao.quantization.disable_observer)
                model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()

            if val_loader is not None:
                val_acc = self._validate(model, val_loader)
                print(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"Loss: {epoch_loss / max(num_batches, 1):.4f} - "
                    f"Val Acc: {val_acc:.4f}"
                )

        return model

    def _validate(self, model: nn.Module, val_loader: DataLoader) -> float:
        """Run validation and return accuracy."""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / total if total > 0 else 0.0

    def convert(self, model: nn.Module) -> nn.Module:
        """Convert QAT model to fully quantized model.

        Converts fake quantization nodes to actual integer operations.

        Args:
            model: QAT-trained model.

        Returns:
            Fully quantized model for deployment.
        """
        model.eval()
        quantized = torch.ao.quantization.convert(model, inplace=False)
        return quantized

    def __repr__(self) -> str:
        return f"QATTrainer(dtype={self.dtype}, epochs={self.epochs}, lr={self.lr})"
