"""FlashOptim — Model optimization toolkit for quantization, pruning, distillation, and NAS."""

__version__ = "1.1.0"

from flashoptim.models.optimized_model import FlashOptim
from flashoptim.models.lora import apply_lora, apply_qlora, merge_lora_weights
from flashoptim.engine.trainer import Trainer
from flashoptim.engine.validator import Validator
from flashoptim.engine.predictor import Predictor
from flashoptim.engine.exporter import Exporter
from flashoptim.cfg import get_config
from flashoptim.quantization import PTQuantizer, QATTrainer
from flashoptim.pruning import UnstructuredPruner, StructuredPruner
from flashoptim.distillation import KnowledgeDistiller, FeatureDistiller
from flashoptim.nas import SearchSpace, Searcher
from flashoptim.solutions import AutoOptimizer, DeploymentProfiler
from flashoptim.analytics import Benchmark

__all__ = [
    "FlashOptim", "Trainer", "Validator", "Predictor", "Exporter",
    "apply_lora", "apply_qlora", "merge_lora_weights", "get_config",
    "PTQuantizer", "QATTrainer",
    "UnstructuredPruner", "StructuredPruner",
    "KnowledgeDistiller", "FeatureDistiller",
    "SearchSpace", "Searcher",
    "AutoOptimizer", "DeploymentProfiler",
    "Benchmark",
    "__version__",
]
