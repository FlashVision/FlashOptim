"""Dataclass-based configuration for FlashOptim pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""

    method: str = "ptq"
    dtype: str = "int8"
    calibration_samples: int = 500
    per_channel: bool = True
    symmetric: bool = True
    model_path: str = ""
    sensitive_layers: List[str] = field(default_factory=list)


@dataclass
class PruningConfig:
    """Configuration for pruning."""

    method: str = "unstructured"
    sparsity: float = 0.5
    criterion: str = "magnitude"
    iterative: bool = True
    iterations: int = 3
    finetune_epochs: int = 10
    model_path: str = ""
    granularity: str = "weight"


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""

    method: str = "knowledge"
    temperature: float = 4.0
    alpha: float = 0.7
    teacher_path: str = ""
    student_path: str = ""
    feature_layers: List[str] = field(default_factory=list)
    loss_type: str = "kl_div"


@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search."""

    strategy: str = "evolutionary"
    population_size: int = 50
    generations: int = 30
    mutation_prob: float = 0.1
    crossover_prob: float = 0.5
    model_path: str = ""
    max_flops: float = 1.0e9
    max_params: float = 5.0e6
    min_accuracy: float = 0.85
    target_latency_ms: float = 10.0


@dataclass
class DataConfig:
    """Configuration for data loading."""

    train_images: str = ""
    val_images: str = ""
    calibration_images: str = ""
    num_workers: int = 4
    batch_size: int = 32


@dataclass
class TrainingConfig:
    """Configuration for training/fine-tuning."""

    epochs: int = 100
    lr: float = 0.01
    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 0.0001
    scheduler: str = "cosine"
    warmup_epochs: int = 5


@dataclass
class ExportConfig:
    """Configuration for model export."""

    output: str = "optimized/model.onnx"
    simplify: bool = True
    opset_version: int = 17
    format: str = "onnx"


@dataclass
class Config:
    """Main FlashOptim configuration container."""

    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    nas: NASConfig = field(default_factory=NASConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A populated Config instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        config = cls()

        if "quantization" in raw:
            config.quantization = QuantizationConfig(**raw["quantization"])
        if "pruning" in raw:
            config.pruning = PruningConfig(**raw["pruning"])
        if "distillation" in raw:
            config.distillation = DistillationConfig(**raw["distillation"])
        if "nas" in raw:
            config.nas = NASConfig(**raw["nas"])
        if "data" in raw:
            config.data = DataConfig(**raw["data"])
        if "training" in raw:
            config.training = TrainingConfig(**raw["training"])
        if "export" in raw:
            config.export = ExportConfig(**raw["export"])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary."""
        from dataclasses import asdict

        return asdict(self)


def get_config(path: Optional[str | Path] = None) -> Config:
    """Load a FlashOptim configuration.

    Args:
        path: Optional path to a YAML config file. Returns default config if None.

    Returns:
        A Config instance.
    """
    if path is None:
        return Config()
    return Config.from_yaml(path)
