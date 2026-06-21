# Frequently Asked Questions

## General

### What is FlashOptim?

FlashOptim is a model optimization toolkit for compressing and accelerating deep learning models. It supports quantization, pruning, distillation, and neural architecture search.

### Which models are supported?

FlashOptim works with any PyTorch model, with first-class support for FlashVision detection and classification models.

### What hardware is required?

- **Training/Optimization**: GPU recommended (CUDA 11.8+)
- **Inference**: CPU, GPU, or edge devices (via ONNX/TensorRT export)

---

## Quantization

### PTQ vs QAT — which should I use?

- **PTQ** is faster (no training required) but may lose 1-2% accuracy
- **QAT** requires training but typically recovers accuracy to within 0.5%
- Start with PTQ; use QAT if accuracy drop is unacceptable

### How many calibration samples do I need?

Typically 100-500 representative samples are sufficient. More samples help with histogram-based calibration.

---

## Pruning

### Does unstructured pruning actually speed up inference?

Not without sparse hardware or sparse inference engines. Use structured pruning for guaranteed speedup on standard hardware.

### Can I combine pruning with quantization?

Yes! Apply pruning first, fine-tune, then quantize. This gives both sparsity and reduced precision.

---

## Distillation

### Does the teacher need to be the same architecture?

No. Feature distillation with projection layers can handle different architectures. Logit distillation works regardless of architecture.

### What temperature should I use?

Start with T=4. Higher temperatures (5-10) work better when the teacher is much larger.

---

## NAS

### How long does a NAS search take?

Depends on strategy and search space:
- Random search: minutes to hours
- Evolutionary: hours to days
- Use proxy tasks to speed up evaluation

---

## Deployment

### What export formats are supported?

- ONNX (recommended)
- TensorRT (planned)
- OpenVINO (planned)
- CoreML (planned)

### How do I benchmark my optimized model?

```bash
flashoptim benchmark --model optimized/model.onnx --device cpu --warmup 10 --runs 100
```
