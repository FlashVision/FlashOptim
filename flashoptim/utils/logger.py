"""Logging utilities and metric tracking."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "flashoptim",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        name: Logger name.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_file: Optional file path to write logs to.
        fmt: Log message format string.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


class AverageMeter:
    """Running average tracker for training metrics.

    Tracks the sum, count, average, and current value of a metric
    across training iterations.

    Example:
        >>> meter = AverageMeter("loss")
        >>> meter.update(0.5)
        >>> meter.update(0.3)
        >>> meter.avg
        0.4
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """Record a new value.

        Args:
            val: Metric value for this batch.
            n: Number of samples in this batch.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter({self.name}: avg={self.avg:.4f}, count={self.count})"
