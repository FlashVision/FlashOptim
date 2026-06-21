# Pruning

Pruning removes redundant parameters from a model to reduce size and computation.

## Unstructured Pruning

Removes individual weights based on importance (magnitude).

```python
from flashoptim import FlashOptim, UnstructuredPruner

model = FlashOptim("pretrained/model.pth")

pruner = UnstructuredPruner(
    sparsity=0.5,
    method="magnitude",
    iterative=True,
    iterations=3,
)
pruned = pruner.prune(model)
```

### Methods

| Method | Description |
|--------|-------------|
| `magnitude` | Remove smallest absolute weights |
| `random` | Random weight removal (baseline) |
| `gradient` | Remove by gradient magnitude |

## Structured Pruning

Removes entire channels or filters for actual speedup without sparse hardware.

```python
from flashoptim import FlashOptim, StructuredPruner

model = FlashOptim("pretrained/model.pth")

pruner = StructuredPruner(
    sparsity=0.3,
    criterion="l1_norm",
    granularity="channel",
)
pruned = pruner.prune(model)
```

## Lottery Ticket Hypothesis

Find sparse subnetworks that train to full accuracy from initialization.

```python
from flashoptim.pruning import LotteryTicketPruner

pruner = LotteryTicketPruner(
    target_sparsity=0.8,
    rounds=5,
    rewind_epoch=2,
)
ticket = pruner.find_ticket(model, train_data="data/train/")
```

## Iterative Pruning with Fine-tuning

```python
from flashoptim import UnstructuredPruner, Trainer

pruner = UnstructuredPruner(sparsity=0.5)
pruned = pruner.prune(model)

trainer = Trainer(epochs=10, lr=0.001)
finetuned = trainer.train(pruned, data="data/train/")
```

## CLI Usage

```bash
flashoptim prune --config configs/flashoptim_prune_unstructured.yaml
```
