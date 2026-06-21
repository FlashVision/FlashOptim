# Knowledge Distillation

Transfer knowledge from a large teacher model to a smaller student model.

## Logit-Level Distillation

Classic knowledge distillation using softened output logits.

```python
from flashoptim import FlashOptim, KnowledgeDistiller, Trainer

teacher = FlashOptim("pretrained/teacher_large.pth")
student = FlashOptim("pretrained/student_small.pth")

distiller = KnowledgeDistiller(
    temperature=4.0,
    alpha=0.7,  # Weight of distillation loss vs task loss
)
trainer = Trainer(distiller=distiller, epochs=100)
trained_student = trainer.train(teacher=teacher, student=student, data="data/train/")
```

## Feature-Level Distillation

Match intermediate feature representations between teacher and student.

```python
from flashoptim import FeatureDistiller

distiller = FeatureDistiller(
    teacher_layers=["backbone.layer3", "backbone.layer4"],
    student_layers=["backbone.layer2", "backbone.layer3"],
    loss_type="mse",
)
```

## Self-Distillation

Distill knowledge within the same model across layers or augmentations.

```python
from flashoptim.distillation import SelfDistiller

distiller = SelfDistiller(
    method="layer",  # "layer" or "augmentation"
    teacher_layer="backbone.layer4",
    student_layer="backbone.layer2",
)
```

## Configuration

```yaml
distillation:
  method: knowledge
  temperature: 4.0
  alpha: 0.7
  teacher_path: pretrained/teacher_large.pth
  student_path: pretrained/student_small.pth
```

## CLI Usage

```bash
flashoptim distill --config configs/flashoptim_distill_det.yaml
```

## Tips

- Higher temperature (3-10) softens probabilities more, transferring "dark knowledge"
- Alpha balances task loss vs distillation loss (0.5-0.9 typical)
- Feature distillation helps when teacher and student have different architectures
- Combine with pruning or quantization for maximum compression
