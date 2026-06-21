# Quick Start

Get started with FlashOptim in minutes.

## 1. Quantize a Model (Fastest Optimization)

```python
from flashoptim import FlashOptim, PTQuantizer

# Load your model
model = FlashOptim("pretrained/model.pth")

# Apply INT8 quantization
quantizer = PTQuantizer(dtype="int8", calibration_samples=500)
quantized = quantizer.quantize(model, calibration_data="data/calibration/")

# Export
quantized.export("optimized/model_int8.onnx")
print(f"Size reduction: {quantized.compression_ratio}x")
```

## 2. Prune a Model

```python
from flashoptim import FlashOptim, UnstructuredPruner

model = FlashOptim("pretrained/model.pth")

pruner = UnstructuredPruner(sparsity=0.5, method="magnitude")
pruned = pruner.prune(model)

pruned.export("optimized/model_pruned.onnx")
```

## 3. CLI Workflow

```bash
# Quantize
flashoptim quantize --config configs/flashoptim_quantize_int8.yaml

# Prune
flashoptim prune --config configs/flashoptim_prune_unstructured.yaml

# Benchmark the result
flashoptim benchmark --model optimized/model_int8.onnx --device cpu
```

## 4. Auto-Optimize (Recommended for Beginners)

```python
from flashoptim import AutoOptimizer

optimizer = AutoOptimizer(
    model_path="pretrained/model.pth",
    target="edge",
    constraints={"latency_ms": 10}
)
result = optimizer.run()
result.export("optimized/model_auto.onnx")
```

## Next Steps

- [Quantization Guide](Quantization.md) — Deep dive into PTQ and QAT
- [Pruning Guide](Pruning.md) — Structured and unstructured pruning
- [Distillation Guide](Distillation.md) — Knowledge transfer techniques
- [NAS Guide](NAS.md) — Automated architecture search
