# Quantization

Quantization reduces model precision from FP32 to lower bit-widths (INT8, FP16), dramatically reducing model size and inference latency.

## Post-Training Quantization (PTQ)

PTQ quantizes a pre-trained model without retraining. It requires a small calibration dataset.

```python
from flashoptim import FlashOptim, PTQuantizer

model = FlashOptim("pretrained/model.pth")

quantizer = PTQuantizer(
    dtype="int8",
    calibration_samples=500,
    per_channel=True,
    symmetric=True,
)
quantized = quantizer.quantize(model, calibration_data="data/calibration/")
```

### Calibration Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `minmax` | Track min/max activation ranges | Speed |
| `histogram` | Entropy-based calibration | Accuracy |
| `percentile` | Clip outliers at percentile | Robustness |

## Quantization-Aware Training (QAT)

QAT simulates quantization during training, enabling the model to adapt to reduced precision.

```python
from flashoptim import FlashOptim, QATTrainer

model = FlashOptim("pretrained/model.pth")

trainer = QATTrainer(
    dtype="int8",
    epochs=10,
    lr=0.001,
)
qat_model = trainer.train(model, train_data="data/train/", val_data="data/val/")
```

## Mixed Precision

```python
quantizer = PTQuantizer(
    dtype="mixed",
    sensitive_layers=["backbone.layer1", "head.cls"],  # Keep FP32
)
```

## CLI Usage

```bash
flashoptim quantize --config configs/flashoptim_quantize_int8.yaml
```

## Tips

- Use at least 100-500 calibration samples representative of your data
- Histogram calibration is slower but often more accurate
- QAT typically recovers 0.5-1% accuracy compared to PTQ
- Per-channel quantization is more accurate than per-tensor
