"""Knowledge distillation methods for model compression."""

from flashoptim.distillation.knowledge_distill import KnowledgeDistiller
from flashoptim.distillation.feature_distill import FeatureDistiller
from flashoptim.distillation.self_distill import SelfDistiller

__all__ = ["KnowledgeDistiller", "FeatureDistiller", "SelfDistiller"]
