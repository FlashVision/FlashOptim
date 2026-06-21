#!/usr/bin/env python3
"""Example: Post-Training Quantization with FlashOptim.

Demonstrates how to quantize a model using PTQuantizer for INT8 deployment.

Usage:
    python examples/quantize_model.py
"""

import torch
import torch.nn as nn

from flashoptim.quantization import PTQuantizer, Calibrator
from flashoptim.utils.model_utils import count_parameters, get_model_size_mb


def build_demo_model() -> nn.Module:
    """Build a simple CNN for demonstration."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    )


def main():
    print("=" * 60)
    print("FlashOptim — Post-Training Quantization Example")
    print("=" * 60)

    model = build_demo_model()
    model.eval()

    print(f"\nOriginal Model:")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Size:       {get_model_size_mb(model):.2f} MB")

    quantizer = PTQuantizer(
        dtype="int8",
        per_channel=True,
        symmetric=True,
        observer="minmax",
    )
    print(f"\nQuantizer: {quantizer}")

    calibrator = Calibrator(num_samples=100, batch_size=32)
    print(f"Calibrator: num_samples={calibrator.num_samples}")

    print("\nNote: Full quantization pipeline coming in FlashOptim v1.1.")
    print("This example demonstrates the API surface and configuration.")

    dummy_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\nModel output shape: {output.shape}")
    print("Done!")


if __name__ == "__main__":
    main()
