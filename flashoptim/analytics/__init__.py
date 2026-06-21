"""Analytics, benchmarking, profiling, and visualization."""

from flashoptim.analytics.benchmark import Benchmark
from flashoptim.analytics.profiler import Profiler
from flashoptim.analytics.plots import (
    plot_training_curves,
    plot_optimization_comparison,
    plot_sparsity_map,
)

__all__ = [
    "Benchmark",
    "Profiler",
    "plot_training_curves",
    "plot_optimization_comparison",
    "plot_sparsity_map",
]
