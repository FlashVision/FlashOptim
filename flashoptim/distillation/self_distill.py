"""Self-Distillation — a model distills knowledge from its own deeper layers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfDistiller:
    """Self-distillation: deeper layers teach shallower auxiliary classifiers.

    Attaches auxiliary classification heads at intermediate layers and
    trains them to mimic the final classifier's output, regularizing
    the network and improving feature quality at every stage.

    Args:
        aux_layers: Names of intermediate layers to attach auxiliary heads.
        temperature: Softmax temperature for distillation.
        aux_weight: Weight of auxiliary losses relative to the main loss.
        num_classes: Number of output classes for auxiliary heads.
    """

    def __init__(
        self,
        aux_layers: Optional[List[str]] = None,
        temperature: float = 3.0,
        aux_weight: float = 0.5,
        num_classes: int = 1000,
    ) -> None:
        self.aux_layers = aux_layers or []
        self.temperature = temperature
        self.aux_weight = aux_weight
        self.num_classes = num_classes
        self._aux_heads: nn.ModuleDict = nn.ModuleDict()
        self._features: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []

    def attach_aux_heads(self, model: nn.Module) -> nn.Module:
        """Attach auxiliary classification heads at specified layers.

        Registers forward hooks to capture intermediate features, runs a
        dummy forward pass to detect channel dimensions, then creates
        lightweight classifier heads for each layer.

        Args:
            model: Base model to augment with auxiliary heads.

        Returns:
            Model reference (modified in-place with hooks).
        """
        for layer_name in self.aux_layers:
            module = dict(model.named_modules()).get(layer_name)
            if module is None:
                raise ValueError(f"Layer '{layer_name}' not found in model")

            hook = module.register_forward_hook(self._make_hook(layer_name))
            self._hooks.append(hook)

        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        device = next(model.parameters()).device
        dummy = dummy.to(device)
        with torch.no_grad():
            model(dummy)

        for layer_name in self.aux_layers:
            feat = self._features.get(layer_name)
            if feat is not None and feat.ndim == 4:
                in_channels = feat.shape[1]
                head = self.build_aux_head(in_channels).to(device)
                self._aux_heads[layer_name] = head
            elif feat is not None and feat.ndim == 2:
                in_features = feat.shape[1]
                head = nn.Sequential(
                    nn.Linear(in_features, self.num_classes),
                ).to(device)
                self._aux_heads[layer_name] = head

        return model

    def build_aux_head(self, in_features: int) -> nn.Module:
        """Build a lightweight auxiliary classifier head.

        Args:
            in_features: Number of input features from the intermediate layer.

        Returns:
            Sequential classifier module.
        """
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, self.num_classes),
        )

    def _make_hook(self, name: str):
        """Create a forward hook that captures intermediate features."""

        def hook_fn(module, input, output):
            self._features[name] = output

        return hook_fn

    def compute_self_distill_loss(
        self,
        main_logits: torch.Tensor,
        targets: torch.Tensor,
        task_loss_fn: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Compute combined loss: main task loss + auxiliary self-distillation losses.

        Each auxiliary head is trained to match the softened main logits
        (self-distillation) and optionally the hard targets.

        Args:
            main_logits: Final classifier output logits.
            targets: Ground truth labels.
            task_loss_fn: Task-specific loss (defaults to CrossEntropyLoss).

        Returns:
            Combined loss scalar.
        """
        task_loss_fn = task_loss_fn or nn.CrossEntropyLoss()
        main_loss = task_loss_fn(main_logits, targets)

        if not self._aux_heads or not self._features:
            return main_loss

        T = self.temperature
        teacher_soft = F.softmax(main_logits.detach() / T, dim=-1)

        aux_loss = torch.tensor(0.0, device=main_logits.device)
        count = 0

        for layer_name, head in self._aux_heads.items():
            feat = self._features.get(layer_name)
            if feat is None:
                continue

            aux_logits = head(feat)
            student_soft = F.log_softmax(aux_logits / T, dim=-1)
            kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)
            hard_loss = task_loss_fn(aux_logits, targets)
            aux_loss = aux_loss + (0.5 * kd_loss + 0.5 * hard_loss)
            count += 1

        if count > 0:
            aux_loss = aux_loss / count

        return main_loss + self.aux_weight * aux_loss

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._features.clear()

    def train(
        self,
        model: nn.Module,
        train_loader: Any = None,
        val_loader: Any = None,
        epochs: int = 100,
        **kwargs: Any,
    ) -> nn.Module:
        """Run self-distillation training loop.

        Args:
            model: Model with auxiliary heads attached.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of training epochs.
            **kwargs: Additional training parameters (lr, device).

        Returns:
            Trained model.
        """
        if train_loader is None:
            raise ValueError("train_loader is required for self-distillation")

        lr = kwargs.get("lr", 0.001)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        model.train()

        all_params = list(model.parameters())
        for head in self._aux_heads.values():
            all_params.extend(head.parameters())

        optimizer = torch.optim.Adam(all_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            model.train()
            for head in self._aux_heads.values():
                head.train()

            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                else:
                    continue

                main_logits = model(inputs)
                loss = self.compute_self_distill_loss(main_logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = epoch_loss / max(num_batches, 1)
                print(f"  Self-Distill Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

        return model

    def __repr__(self) -> str:
        return f"SelfDistiller(aux_layers={self.aux_layers}, T={self.temperature}, aux_weight={self.aux_weight})"
