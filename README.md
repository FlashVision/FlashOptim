<p align="center">
  <h1 align="center">⚡ FlashOptim</h1>
  <p align="center">
    <strong>Model Optimization Toolkit for FlashVision</strong>
  </p>
  <p align="center">
    Quantization • Pruning • Distillation • Neural Architecture Search • Deployment
  </p>
</p>

<p align="center">
  <a href="https://github.com/FlashVision/FlashOptim/actions"><img src="https://img.shields.io/github/actions/workflow/status/FlashVision/FlashOptim/ci.yml?style=flat-square&logo=github" alt="CI"></a>
  <a href="https://pypi.org/project/flashoptim/"><img src="https://img.shields.io/pypi/v/flashoptim?style=flat-square&logo=pypi" alt="PyPI"></a>
  <a href="https://pypi.org/project/flashoptim/"><img src="https://img.shields.io/pypi/pyversions/flashoptim?style=flat-square&logo=python" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License"></a>
  <a href="https://github.com/FlashVision/FlashOptim"><img src="https://img.shields.io/github/stars/FlashVision/FlashOptim?style=flat-square&logo=github" alt="Stars"></a>
</p>

---

## 🚀 What is FlashOptim?

**FlashOptim** is a comprehensive model optimization toolkit designed for [FlashVision](https://github.com/FlashVision) models. It provides state-of-the-art techniques to compress, accelerate, and deploy deep learning models efficiently on edge devices, mobile platforms, and cloud infrastructure.

### Key Features

| Feature | Description |
|---------|-------------|
| **Quantization** | Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) — INT8, FP16, mixed precision |
| **Pruning** | Unstructured (magnitude), Structured (channel/filter), Lottery Ticket Hypothesis |
| **Distillation** | Knowledge distillation (logit-level), Feature distillation, Self-distillation |
| **NAS** | Neural Architecture Search with configurable search spaces and strategies |
| **LoRA** | Low-Rank Adaptation and QLoRA for efficient fine-tuning |
| **Export** | ONNX, TensorRT, OpenVINO, CoreML export with optimization |
| **Auto-Optimizer** | One-click optimization pipeline with automatic method selection |
| **Benchmarking** | Latency, throughput, memory, and accuracy profiling |

---

## 📦 Installation

### From PyPI (recommended)

```bash
pip install flashoptim
```

### From Source

```bash
git clone https://github.com/FlashVision/FlashOptim.git
cd FlashOptim
pip install -e ".[all]"
```

### With Optional Dependencies

```bash
pip install flashoptim[export]        # ONNX export support
pip install flashoptim[quantization]  # Quantization extras
pip install flashoptim[analytics]     # Visualization & profiling
pip install flashoptim[all]           # Everything
pip install flashoptim[dev]           # Development tools
```

---

## ⚡ Quick Start

### Quantize a Model (INT8)

```python
from flashoptim import FlashOptim, PTQuantizer

model = FlashOptim("pretrained/model.pth")

quantizer = PTQuantizer(dtype="int8", calibration_samples=500)
quantized_model = quantizer.quantize(model, calibration_data="data/calibration/")

quantized_model.export("optimized/model_int8.onnx")
```

### Prune a Model

```python
from flashoptim import FlashOptim, UnstructuredPruner

model = FlashOptim("pretrained/model.pth")

pruner = UnstructuredPruner(sparsity=0.5, method="magnitude")
pruned_model = pruner.prune(model)

pruned_model.export("optimized/model_pruned.onnx")
```

### Knowledge Distillation

```python
from flashoptim import FlashOptim, KnowledgeDistiller, Trainer

teacher = FlashOptim("pretrained/teacher_large.pth")
student = FlashOptim("pretrained/student_small.pth")

distiller = KnowledgeDistiller(temperature=4.0, alpha=0.7)
trainer = Trainer(distiller=distiller, epochs=50)
trainer.train(teacher=teacher, student=student, data="data/train/")
```

### Auto-Optimize (One-Click)

```python
from flashoptim.solutions import AutoOptimizer

optimizer = AutoOptimizer(target="edge")  # "edge", "mobile", "server"
optimized = optimizer.optimize(model)
print(optimizer.get_report())
```

---

## 🖥️ CLI Usage

```bash
# Quantize a model
flashoptim quantize --config configs/flashoptim_quantize_int8.yaml

# Prune a model
flashoptim prune --config configs/flashoptim_prune_unstructured.yaml

# Knowledge distillation
flashoptim distill --config configs/flashoptim_distill_det.yaml

# Neural Architecture Search
flashoptim nas --config configs/flashoptim_nas_search.yaml

# Export optimized model
flashoptim export --model optimized/model.pth --format onnx

# Benchmark
flashoptim benchmark --model optimized/model.onnx --device cpu
```

---

## 📁 Project Structure

```
FlashOptim/
├── configs/          # YAML configuration files
├── docker/           # Docker support
├── docs/             # Documentation
├── examples/         # Runnable example scripts
├── flashoptim/       # Core library
│   ├── cfg/          # Configuration management
│   ├── data/         # Data loading & calibration
│   ├── engine/       # Training, validation, export
│   ├── models/       # Model architectures
│   ├── losses/       # Loss functions
│   ├── quantization/ # Quantization methods
│   ├── pruning/      # Pruning methods
│   ├── distillation/ # Distillation methods
│   ├── nas/          # Neural Architecture Search
│   ├── solutions/    # High-level optimization solutions
│   ├── analytics/    # Benchmarking & profiling
│   └── utils/        # Utilities
└── tests/            # Unit tests
```

---

## 📊 Benchmarks

| Model | Method | Size Reduction | Latency Speedup | Accuracy Drop |
|-------|--------|---------------|-----------------|---------------|
| FlashDet-S | INT8 PTQ | 4× | 2.3× | < 0.5% |
| FlashDet-M | 50% Pruning | 2× | 1.8× | < 1.0% |
| FlashDet-L | Distillation | 3× | 2.5× | < 0.3% |
| FlashDet-S | Auto-Optimize | 5× | 3.1× | < 1.0% |

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [ONNX](https://onnx.ai/) for model interoperability
- [FlashVision](https://github.com/FlashVision) ecosystem

---

<p align="center">
  Made with ❤️ by the <a href="https://github.com/FlashVision">FlashVision</a> team
</p>
