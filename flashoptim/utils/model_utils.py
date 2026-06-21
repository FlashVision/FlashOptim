"""Model inspection utilities — parameter count, FLOPs, size, sparsity."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count the total number of parameters in a model.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only parameters with requires_grad=True.

    Returns:
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_flops(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 640, 640),
    device: Optional[str] = None,
) -> float:
    """Estimate FLOPs for a single forward pass.

    Uses a hook-based approach to accumulate FLOPs from Conv2d and Linear
    layers. This is an approximation; for exact counts, use a dedicated
    profiler like fvcore or thop.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape (B, C, H, W).
        device: Device for the dummy input.

    Returns:
        Estimated FLOPs count.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    total_flops = [0]
    hooks = []

    def conv_hook(module, input, output):
        batch_size = input[0].size(0)
        out_channels = output.size(1)
        out_h, out_w = output.size(2), output.size(3)
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        total_flops[0] += batch_size * out_channels * out_h * out_w * kernel_ops * 2

    def linear_hook(module, input, output):
        batch_size = input[0].size(0)
        total_flops[0] += batch_size * module.in_features * module.out_features * 2

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    dummy = torch.randn(*input_size, device=device)
    with torch.no_grad():
        model(dummy)

    for hook in hooks:
        hook.remove()

    return float(total_flops[0])


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes (parameter memory only).

    Args:
        model: PyTorch model.

    Returns:
        Model size in MB.
    """
    total_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (total_bytes + buffer_bytes) / (1024 * 1024)


def get_sparsity(model: nn.Module) -> Dict[str, float]:
    """Compute per-layer and global sparsity (fraction of zero weights).

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with per-layer sparsity and a 'global' key for overall sparsity.
    """
    total_zeros = 0
    total_elements = 0
    layer_sparsity: Dict[str, float] = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            zeros = (weight == 0).sum().item()
            elements = weight.numel()
            total_zeros += zeros
            total_elements += elements
            layer_sparsity[name] = zeros / elements if elements > 0 else 0.0

    layer_sparsity["global"] = total_zeros / total_elements if total_elements > 0 else 0.0
    return layer_sparsity
