"""Checkpoint saving and loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    best_metric: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a training checkpoint.

    Args:
        model: Model to save.
        path: File path for the checkpoint.
        optimizer: Optional optimizer state to include.
        epoch: Current epoch number.
        best_metric: Best validation metric achieved.
        extra: Additional metadata to store.

    Returns:
        Path to the saved checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_metric": best_metric,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, str(path))
    return str(path)


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: Model to load weights into (modifies in-place).
        optimizer: Optimizer to load state into (modifies in-place).
        device: Device to map tensors to.

    Returns:
        Checkpoint dictionary with all stored keys.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(path), map_location=map_location)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
