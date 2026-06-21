# Changelog

All notable changes to FlashOptim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] — 2025-06-21

### Implemented (previously stubbed)
- **PTQ**: Full Post-Training Quantization with `torch.ao.quantization` (INT8, FP16)
- **QAT**: Quantization-Aware Training with fake-quant insertion, training loop, and conversion
- **Calibrator**: Complete calibration pipeline with forward hooks for activation statistics
- **HistogramObserver**: Entropy and MSE threshold selection methods
- **Trainer**: Full training loop with AMP, warmup, cosine annealing, gradient accumulation
- **Validator**: Model evaluation with accuracy, latency, size metrics; model comparison
- **Predictor**: Inference with PyTorch and ONNX models, benchmarking
- **Exporter**: ONNX export via `torch.onnx.export` with simplification and validation
- **ModelCheckpoint callback**: Actual checkpoint saving on epoch end
- **Structured pruning**: Channel/filter pruning via `torch.nn.utils.prune`
- **Lottery Ticket**: Iterative magnitude pruning with weight rewinding
- **Importance scoring**: Taylor, gradient, and activation-based importance methods
- **Knowledge distillation**: Full training loop with KD loss
- **Feature distillation**: Automatic projector building for channel alignment
- **Self-distillation**: Auxiliary head creation and self-distill training loop
- **LoRA/QLoRA**: QLoRA with simulated INT4/INT8 quantization of base weights
- **CLI commands**: All 6 commands (`quantize`, `prune`, `distill`, `nas`, `export`, `benchmark`) wired to real implementations
- **AutoOptimizer**: Quantization step now calls PTQuantizer
- **Profiler**: Fixed per-layer timing with pre+post hooks
- **mAP metric**: Proper IoU-based matching with 11-point interpolation

### Fixed
- Checkpoint loading now handles `state_dict` and `model_state_dict` keys with clear error messages
- Profiler timing hooks now correctly measure elapsed time (was measuring zero)
- `compute_map` now uses actual bounding box IoU matching instead of naive counting

---

## [1.0.0] — 2024-12-01

### Added
- Initial release of FlashOptim
- Project structure with quantization, pruning, distillation, NAS modules
- Unstructured pruning (magnitude-based) — fully implemented
- MinMaxObserver and HistogramObserver (percentile method)
- LoRA adapter implementation
- AutoOptimizer framework with pruning support
- `Calibrator` hook registration and statistics collection helpers
- `KnowledgeDistiller.compute_loss()` with KL-div, MSE, cosine
- `FeatureDistiller.compute_loss()` with feature matching
- `SelfDistiller.compute_self_distill_loss()`
- `Validator.model_size_mb()` and `count_parameters()` utilities
- `EarlyStopping` callback
- CLI framework with `version`, `settings`, `check` commands
- Docker support with GPU-enabled Dockerfile
- GitHub Actions CI pipeline
- Documentation and examples

---

## [Unreleased]

### Planned
- TensorRT and OpenVINO export backends
- Mixed-precision quantization strategies
- Hardware-aware NAS constraints
- Distributed optimization support
- Model zoo with pre-optimized checkpoints
