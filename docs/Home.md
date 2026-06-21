# FlashOptim Documentation

Welcome to the **FlashOptim** documentation — a comprehensive model optimization toolkit for FlashVision models.

## Overview

FlashOptim provides state-of-the-art model compression and optimization techniques:

- **Quantization** — Reduce model precision (INT8, FP16) for faster inference
- **Pruning** — Remove redundant weights and channels
- **Distillation** — Transfer knowledge from large to small models
- **NAS** — Automatically search for efficient architectures
- **LoRA** — Low-Rank Adaptation for efficient fine-tuning
- **Export** — Deploy to ONNX, TensorRT, OpenVINO

## Navigation

| Page | Description |
|------|-------------|
| [Installation](Installation.md) | Setup and installation guide |
| [Quick Start](Quick-Start.md) | Get running in minutes |
| [Quantization](Quantization.md) | PTQ and QAT guide |
| [Pruning](Pruning.md) | Weight and channel pruning |
| [Distillation](Distillation.md) | Knowledge distillation |
| [NAS](NAS.md) | Neural Architecture Search |
| [FAQ](FAQ.md) | Frequently asked questions |

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.8 (recommended for GPU acceleration)
