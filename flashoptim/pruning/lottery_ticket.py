"""Lottery Ticket Hypothesis implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class LotteryTicketPruner:
    """Lottery Ticket Hypothesis — find sparse trainable subnetworks.

    Implements the iterative magnitude pruning (IMP) procedure to discover
    sparse subnetworks ("winning tickets") that can train to full accuracy
    from their original initialization.

    Reference: Frankle & Carlin, "The Lottery Ticket Hypothesis" (ICLR 2019)

    Args:
        target_sparsity: Final target sparsity ratio.
        rounds: Number of pruning rounds.
        rewind_epoch: Epoch to rewind weights to (0 = initialization).
        pruning_rate: Per-round pruning rate (if None, computed from target).
    """

    def __init__(
        self,
        target_sparsity: float = 0.8,
        rounds: int = 5,
        rewind_epoch: int = 0,
        pruning_rate: Optional[float] = None,
    ) -> None:
        self.target_sparsity = target_sparsity
        self.rounds = rounds
        self.rewind_epoch = rewind_epoch
        self.pruning_rate = pruning_rate or (1.0 - (1.0 - target_sparsity) ** (1.0 / rounds))

        self._initial_weights: Optional[Dict[str, torch.Tensor]] = None
        self._masks: Dict[str, torch.Tensor] = {}

    def save_initial_weights(self, model: nn.Module) -> None:
        """Save initial model weights for rewinding.

        Args:
            model: Model at initialization (or rewind epoch).
        """
        self._initial_weights = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

    def find_ticket(
        self,
        model: nn.Module,
        train_data: Optional[str] = None,
        train_fn: Optional[Any] = None,
    ) -> nn.Module:
        """Find a winning lottery ticket through iterative magnitude pruning.

        Implements the IMP algorithm:
        1. Save initial weights
        2. Train the model
        3. Prune lowest-magnitude weights
        4. Rewind remaining weights to initialization
        5. Repeat for `rounds` iterations

        Args:
            model: The original model.
            train_data: Path to training data.
            train_fn: Training function that trains the model for one round.
                Should accept (model) and return the trained model.

        Returns:
            The sparse "winning ticket" model with masks applied.
        """
        if self._initial_weights is None:
            self.save_initial_weights(model)

        for round_idx in range(self.rounds):
            if train_fn is not None:
                model = train_fn(model)

            self._prune_round(model)
            self._rewind_weights(model)
            self._apply_masks(model)

            current_sp = self.current_sparsity
            print(
                f"  IMP Round {round_idx + 1}/{self.rounds} — "
                f"Sparsity: {current_sp:.2%}"
            )

            if current_sp >= self.target_sparsity:
                break

        return model

    def _prune_round(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Execute one round of magnitude pruning.

        Args:
            model: Current model state.

        Returns:
            Updated masks dictionary.
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data.abs()
                if name in self._masks:
                    weight = weight * self._masks[name]

                flat = weight.flatten()
                num_prune = int(flat.numel() * self.pruning_rate)
                if num_prune > 0:
                    threshold = flat.kthvalue(num_prune).values
                    mask = (weight > threshold).float()
                    self._masks[name] = mask

        return self._masks

    def _rewind_weights(self, model: nn.Module) -> None:
        """Rewind model weights to saved initialization.

        Args:
            model: Model to rewind.
        """
        if self._initial_weights is None:
            raise RuntimeError("Initial weights not saved. Call save_initial_weights() first.")

        for name, param in model.named_parameters():
            if name in self._initial_weights:
                param.data.copy_(self._initial_weights[name])

    def _apply_masks(self, model: nn.Module) -> None:
        """Apply computed masks to model weights.

        Args:
            model: Model to mask.
        """
        for name, module in model.named_modules():
            if name in self._masks and isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.data *= self._masks[name]

    @property
    def current_sparsity(self) -> float:
        """Calculate current sparsity from masks."""
        if not self._masks:
            return 0.0
        total = sum(m.numel() for m in self._masks.values())
        zeros = sum((m == 0).sum().item() for m in self._masks.values())
        return zeros / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"LotteryTicketPruner(target_sparsity={self.target_sparsity}, "
            f"rounds={self.rounds}, rewind_epoch={self.rewind_epoch})"
        )
