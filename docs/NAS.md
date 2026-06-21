# Neural Architecture Search (NAS)

Automatically discover efficient model architectures under hardware constraints.

## Overview

FlashOptim NAS searches for optimal architectures that balance accuracy, latency, and model size.

## Basic Usage

```python
from flashoptim import SearchSpace, Searcher

search_space = SearchSpace(
    channels=[16, 32, 64, 128, 256],
    kernel_sizes=[3, 5, 7],
    depths=[1, 2, 3, 4],
    operations=["conv", "dwconv", "mbconv", "skip"],
)

searcher = Searcher(
    strategy="evolutionary",
    population_size=50,
    generations=30,
)

best_arch = searcher.search(
    search_space=search_space,
    train_data="data/train/",
    val_data="data/val/",
    constraints={"max_flops": 1e9, "max_params": 5e6},
)
```

## Search Strategies

| Strategy | Description | Speed | Quality |
|----------|-------------|-------|---------|
| `random` | Random sampling | Fast | Baseline |
| `evolutionary` | Genetic algorithm | Medium | Good |
| `bayesian` | Bayesian optimization | Slow | Best |

## Constraints

```python
constraints = {
    "max_flops": 1.0e9,       # Maximum FLOPs
    "max_params": 5.0e6,      # Maximum parameters
    "max_latency_ms": 10.0,   # Target latency
    "min_accuracy": 0.85,     # Minimum accuracy threshold
}
```

## Hardware-Aware NAS

```python
searcher = Searcher(
    strategy="evolutionary",
    hardware="gpu",  # "gpu", "cpu", "mobile", "edge"
    latency_lookup_table="latency_lut.json",
)
```

## CLI Usage

```bash
flashoptim nas --config configs/flashoptim_nas_search.yaml
```

## Tips

- Start with random search to understand the search space
- Use proxy tasks (fewer epochs) for faster evaluation
- Hardware-aware search provides more realistic results
- Population size of 50+ recommended for evolutionary search
