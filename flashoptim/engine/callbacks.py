"""Callback system for FlashOptim training and optimization pipelines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class Callback:
    """Base callback class for hooking into training/optimization events.

    Override any of the on_* methods to implement custom behavior.
    """

    def on_train_start(self, state: Dict[str, Any]) -> None:
        """Called at the beginning of training."""

    def on_train_end(self, state: Dict[str, Any]) -> None:
        """Called at the end of training."""

    def on_epoch_start(self, epoch: int, state: Dict[str, Any]) -> None:
        """Called at the beginning of each epoch."""

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Called at the end of each epoch."""

    def on_batch_start(self, batch_idx: int, batch: Any) -> None:
        """Called before processing a batch."""

    def on_batch_end(self, batch_idx: int, loss: float) -> None:
        """Called after processing a batch."""

    def on_optimization_start(self, method: str, config: Dict[str, Any]) -> None:
        """Called when an optimization method starts."""

    def on_optimization_end(self, method: str, results: Dict[str, Any]) -> None:
        """Called when an optimization method completes."""

    def on_export_start(self, format: str, path: str) -> None:
        """Called before model export."""

    def on_export_end(self, format: str, path: str) -> None:
        """Called after model export."""


class CallbackManager:
    """Manages a collection of callbacks and dispatches events.

    Args:
        callbacks: Optional list of Callback instances.
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        self.callbacks: List[Callback] = callbacks or []

    def add(self, callback: Callback) -> None:
        """Add a callback to the manager."""
        self.callbacks.append(callback)

    def remove(self, callback: Callback) -> None:
        """Remove a callback from the manager."""
        self.callbacks.remove(callback)

    def fire(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Fire an event, calling the corresponding method on all callbacks.

        Args:
            event: Event name (e.g., 'on_epoch_end').
            *args: Positional arguments for the callback method.
            **kwargs: Keyword arguments for the callback method.
        """
        for cb in self.callbacks:
            handler = getattr(cb, event, None)
            if handler is not None:
                handler(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.callbacks)


class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' or 'max' — direction of improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Check if training should stop."""
        value = metrics.get("val_loss", metrics.get("loss"))
        if value is None:
            return

        if self.best is None:
            self.best = value
            return

        improved = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


class ModelCheckpoint(Callback):
    """Save model checkpoints during training.

    Args:
        save_dir: Directory to save checkpoints.
        save_best_only: Only save when metric improves.
        monitor: Metric to monitor.
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        save_best_only: bool = True,
        monitor: str = "val_loss",
    ) -> None:
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.best_value: Optional[float] = None

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Save checkpoint if conditions are met."""
        import os

        import torch

        value = metrics.get(self.monitor)
        if value is None:
            return

        should_save = True
        if self.save_best_only:
            if self.best_value is None or value < self.best_value:
                self.best_value = value
            else:
                should_save = False

        if should_save:
            os.makedirs(self.save_dir, exist_ok=True)
            model = metrics.get("_model")
            if model is not None:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": {k: v for k, v in metrics.items() if k != "_model"},
                }
                if self.save_best_only:
                    path = os.path.join(self.save_dir, "best.pt")
                else:
                    path = os.path.join(self.save_dir, f"epoch_{epoch}.pt")
                torch.save(checkpoint, path)
