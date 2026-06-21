"""Trainer for fine-tuning models after optimization (pruning, distillation, QAT)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader


class Trainer:
    """Fine-tuning trainer for optimized models.

    Supports post-pruning fine-tuning, QAT training loops, and distillation
    training with teacher-student pairs.

    Args:
        epochs: Number of training epochs.
        lr: Learning rate.
        optimizer: Optimizer name ('sgd', 'adam', 'adamw').
        scheduler: LR scheduler name ('cosine', 'step').
        device: Training device ('cuda' or 'cpu').
        distiller: Optional distillation module for KD training.
        callbacks: Optional callback manager.
        save_dir: Directory to save checkpoints.
    """

    def __init__(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        optimizer: str = "sgd",
        scheduler: str = "cosine",
        device: Optional[str] = None,
        distiller: Optional[Any] = None,
        callbacks: Optional[Any] = None,
        save_dir: str = "runs/train",
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.distiller = distiller
        self.callbacks = callbacks
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build optimizer from configuration."""
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizers = {
            "sgd": lambda: SGD(params, lr=self.lr, momentum=0.9, weight_decay=1e-4),
            "adam": lambda: Adam(params, lr=self.lr, weight_decay=1e-4),
            "adamw": lambda: AdamW(params, lr=self.lr, weight_decay=1e-2),
        }
        builder = optimizers.get(self.optimizer_name.lower())
        if builder is None:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        return builder()

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
        """Build learning rate scheduler."""
        schedulers = {
            "cosine": lambda: CosineAnnealingLR(optimizer, T_max=self.epochs),
            "step": lambda: StepLR(optimizer, step_size=30, gamma=0.1),
        }
        builder = schedulers.get(self.scheduler_name.lower())
        if builder is None:
            raise ValueError(f"Unknown scheduler: {self.scheduler_name}")
        return builder()

    def train(
        self,
        model: Optional[nn.Module] = None,
        data: Optional[str] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        teacher: Optional[nn.Module] = None,
        student: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Run the training loop.

        Args:
            model: Model to fine-tune (used for standard training).
            data: Path to training data directory.
            train_loader: Pre-built training DataLoader.
            val_loader: Pre-built validation DataLoader.
            teacher: Teacher model (for distillation).
            student: Student model (for distillation).
            **kwargs: Additional training parameters.

        Returns:
            The trained model.
        """
        from tqdm import tqdm

        active_model = student if student is not None else model
        if active_model is None:
            raise ValueError("Either 'model' or 'student' must be provided")
        if train_loader is None:
            raise ValueError("train_loader is required")

        active_model = active_model.to(self.device)
        if teacher is not None:
            teacher = teacher.to(self.device)
            teacher.eval()

        optimizer = self._build_optimizer(active_model)
        scheduler = self._build_scheduler(optimizer)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda") if self.device == "cuda" else None

        warmup_epochs = min(3, self.epochs // 10)
        accumulation_steps = kwargs.get("accumulation_steps", 1)

        if self.callbacks:
            self.callbacks.fire("on_train_start", {"epochs": self.epochs})

        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            active_model.train()
            epoch_loss = 0.0
            num_batches = 0

            if self.callbacks:
                self.callbacks.fire("on_epoch_start", epoch, {"lr": optimizer.param_groups[0]["lr"]})

            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(progress):
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue

                if self.callbacks:
                    self.callbacks.fire("on_batch_start", batch_idx, batch)

                use_amp = scaler is not None
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = active_model(inputs)

                    if self.distiller is not None and teacher is not None:
                        with torch.no_grad():
                            teacher_outputs = teacher(inputs)
                        loss = self.distiller.compute_loss(outputs, teacher_outputs, targets, criterion)
                    else:
                        loss = criterion(outputs, targets)

                    loss = loss / accumulation_steps

                if scaler is not None:
                    scaler.scale(loss).backward()
                    if (batch_idx + 1) % accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (batch_idx + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                epoch_loss += loss.item() * accumulation_steps
                num_batches += 1
                progress.set_postfix(loss=f"{epoch_loss / num_batches:.4f}")

                if self.callbacks:
                    self.callbacks.fire("on_batch_end", batch_idx, loss.item())

            if epoch < warmup_epochs:
                warmup_factor = (epoch + 1) / warmup_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = self.lr * warmup_factor
            else:
                scheduler.step()

            metrics: Dict[str, Any] = {"loss": epoch_loss / max(num_batches, 1)}

            if val_loader is not None:
                val_loss, val_acc = self._validate(active_model, val_loader, criterion)
                metrics["val_loss"] = val_loss
                metrics["val_accuracy"] = val_acc

            if self.callbacks:
                self.callbacks.fire("on_epoch_end", epoch, metrics)

            val_metric = metrics.get("val_loss", metrics["loss"])
            if val_metric < best_val_loss:
                best_val_loss = val_metric
                self.save_checkpoint(active_model, epoch, str(self.save_dir / "best.pt"))

        self.save_checkpoint(active_model, self.epochs - 1, str(self.save_dir / "last.pt"))

        if self.callbacks:
            self.callbacks.fire("on_train_end", {"best_val_loss": best_val_loss})

        return active_model

    def _validate(self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> tuple:
        """Run validation pass and return (loss, accuracy)."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        avg_loss = total_loss / max(total, 1) * (targets.size(0) if total > 0 else 1)
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def save_checkpoint(self, model: nn.Module, epoch: int, path: Optional[str] = None) -> str:
        """Save a training checkpoint.

        Args:
            model: Model to save.
            epoch: Current epoch number.
            path: Optional custom save path.

        Returns:
            Path to the saved checkpoint.
        """
        save_path = path or str(self.save_dir / f"checkpoint_epoch{epoch}.pt")
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict()},
            save_path,
        )
        return save_path
