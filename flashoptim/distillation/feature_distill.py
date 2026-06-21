"""Feature-level Knowledge Distillation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistiller:
    """Feature-level distillation — matches intermediate representations.

    Aligns feature maps between teacher and student at specified layers,
    enabling structural knowledge transfer beyond output logits.

    Args:
        teacher_layers: Layer names to extract features from teacher.
        student_layers: Corresponding layer names in the student.
        loss_type: Feature matching loss ('mse', 'l1', 'cosine', 'attention').
        projector: Whether to use a projection layer for dimension matching.
    """

    def __init__(
        self,
        teacher_layers: List[str] = None,
        student_layers: List[str] = None,
        loss_type: str = "mse",
        projector: bool = True,
    ) -> None:
        self.teacher_layers = teacher_layers or []
        self.student_layers = student_layers or []
        self.loss_type = loss_type
        self.projector = projector
        self._teacher_features: Dict[str, torch.Tensor] = {}
        self._student_features: Dict[str, torch.Tensor] = {}
        self._projectors: nn.ModuleDict = nn.ModuleDict()

    def build_projectors(self, teacher: nn.Module, student: nn.Module) -> nn.ModuleDict:
        """Build projection layers to align feature dimensions.

        Runs a dummy input through both models to detect channel dimensions
        at the specified feature layers, then creates 1x1 conv projectors
        to align student channels to teacher channels.

        Args:
            teacher: Teacher model.
            student: Student model.

        Returns:
            ModuleDict of projection layers.
        """
        teacher_channels: Dict[str, int] = {}
        student_channels: Dict[str, int] = {}
        t_hooks = []
        s_hooks = []

        def make_channel_hook(storage: Dict[str, int], name: str):
            def hook_fn(module, inp, output):
                if isinstance(output, torch.Tensor) and output.ndim == 4:
                    storage[name] = output.shape[1]
            return hook_fn

        for layer_name in self.teacher_layers:
            module = dict(teacher.named_modules()).get(layer_name)
            if module is not None:
                t_hooks.append(module.register_forward_hook(
                    make_channel_hook(teacher_channels, layer_name)
                ))

        for layer_name in self.student_layers:
            module = dict(student.named_modules()).get(layer_name)
            if module is not None:
                s_hooks.append(module.register_forward_hook(
                    make_channel_hook(student_channels, layer_name)
                ))

        dummy = torch.randn(1, 3, 224, 224)
        teacher.eval()
        student.eval()
        with torch.no_grad():
            teacher(dummy)
            student(dummy)

        for h in t_hooks:
            h.remove()
        for h in s_hooks:
            h.remove()

        projectors = nn.ModuleDict()
        for t_name, s_name in zip(self.teacher_layers, self.student_layers):
            t_ch = teacher_channels.get(t_name)
            s_ch = student_channels.get(s_name)
            if t_ch is not None and s_ch is not None and t_ch != s_ch:
                projectors[s_name] = nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False)
            elif t_ch is not None and s_ch is not None:
                projectors[s_name] = nn.Identity()

        self._projectors = projectors
        return projectors

    def register_hooks(self, teacher: nn.Module, student: nn.Module) -> None:
        """Register forward hooks to capture intermediate features.

        Args:
            teacher: Teacher model.
            student: Student model.
        """
        for layer_name in self.teacher_layers:
            module = dict(teacher.named_modules()).get(layer_name)
            if module is not None:
                module.register_forward_hook(self._make_hook(self._teacher_features, layer_name))

        for layer_name in self.student_layers:
            module = dict(student.named_modules()).get(layer_name)
            if module is not None:
                module.register_forward_hook(self._make_hook(self._student_features, layer_name))

    def _make_hook(self, storage: Dict[str, torch.Tensor], name: str):
        """Create a forward hook that stores feature maps."""
        def hook_fn(module, input, output):
            storage[name] = output
        return hook_fn

    def compute_loss(self) -> torch.Tensor:
        """Compute feature matching loss from captured features.

        Call after running both teacher and student forward passes.

        Returns:
            Feature distillation loss scalar.
        """
        total_loss = torch.tensor(0.0)

        for t_name, s_name in zip(self.teacher_layers, self.student_layers):
            t_feat = self._teacher_features.get(t_name)
            s_feat = self._student_features.get(s_name)

            if t_feat is None or s_feat is None:
                continue

            t_feat = t_feat.detach()

            if t_feat.shape != s_feat.shape and self.projector:
                s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])
                if s_feat.shape[1] != t_feat.shape[1]:
                    continue

            if self.loss_type == "mse":
                loss = F.mse_loss(s_feat, t_feat)
            elif self.loss_type == "l1":
                loss = F.l1_loss(s_feat, t_feat)
            elif self.loss_type == "cosine":
                cos = F.cosine_similarity(
                    s_feat.flatten(2), t_feat.flatten(2), dim=2
                )
                loss = (1.0 - cos).mean()
            elif self.loss_type == "attention":
                s_att = self._attention_map(s_feat)
                t_att = self._attention_map(t_feat)
                loss = F.mse_loss(s_att, t_att)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            total_loss = total_loss + loss

        return total_loss / max(len(self.teacher_layers), 1)

    @staticmethod
    def _attention_map(features: torch.Tensor) -> torch.Tensor:
        """Compute spatial attention map from feature tensor."""
        return F.normalize(features.pow(2).mean(dim=1, keepdim=True).flatten(1), dim=1)

    def __repr__(self) -> str:
        return (
            f"FeatureDistiller(teacher_layers={self.teacher_layers}, "
            f"student_layers={self.student_layers}, loss={self.loss_type})"
        )
