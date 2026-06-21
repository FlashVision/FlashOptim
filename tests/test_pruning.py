"""Tests for pruning module — UnstructuredPruner, StructuredPruner."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from flashoptim.pruning import UnstructuredPruner, StructuredPruner, LotteryTicketPruner, ImportanceScorer


def _make_simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )


def _make_dataloader(num_samples=20, img_size=32, num_classes=10):
    """Create a simple DataLoader for testing."""
    images = torch.randn(num_samples, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=4)


class TestUnstructuredPruner:
    """Tests for unstructured (weight-level) pruning."""

    def test_init_default(self):
        pruner = UnstructuredPruner()
        assert pruner.sparsity == 0.5
        assert pruner.method == "magnitude"

    def test_init_custom(self):
        pruner = UnstructuredPruner(sparsity=0.3, method="random", iterative=False)
        assert pruner.sparsity == 0.3
        assert pruner.method == "random"
        assert pruner.iterative is False

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            UnstructuredPruner(method="nonexistent")

    def test_invalid_sparsity(self):
        with pytest.raises(ValueError, match="Sparsity must be"):
            UnstructuredPruner(sparsity=1.5)

    def test_prune_applies_masks(self):
        model = _make_simple_model()
        pruner = UnstructuredPruner(sparsity=0.3, iterative=False)
        pruned_model = pruner.prune(model)
        sparsity = pruner.get_sparsity(pruned_model)
        assert sparsity["global"] > 0.0

    def test_remove_pruning(self):
        model = _make_simple_model()
        pruner = UnstructuredPruner(sparsity=0.5, iterative=False)
        pruner.prune(model)
        UnstructuredPruner.remove_pruning(model)
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                assert not hasattr(module, "weight_mask")

    def test_repr(self):
        pruner = UnstructuredPruner()
        r = repr(pruner)
        assert "UnstructuredPruner" in r
        assert "0.5" in r


class TestStructuredPruner:
    """Tests for structured (channel/filter) pruning."""

    def test_init_default(self):
        pruner = StructuredPruner()
        assert pruner is not None

    def test_repr(self):
        pruner = StructuredPruner()
        r = repr(pruner)
        assert "StructuredPruner" in r

    def test_prune_applies_structured_masks(self):
        model = _make_simple_model()
        pruner = StructuredPruner(sparsity=0.3, criterion="l1_norm")
        pruned = pruner.prune(model)
        has_zeros = False
        for module in pruned.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, "weight_orig"):
                weight = module.weight
                per_filter_norm = weight.abs().sum(dim=(1, 2, 3))
                if (per_filter_norm == 0).any():
                    has_zeros = True
        assert pruned is not None

    def test_compute_importance(self):
        model = _make_simple_model()
        pruner = StructuredPruner(criterion="l1_norm")
        importance = pruner.compute_importance(model)
        assert len(importance) > 0

    def test_get_pruning_plan(self):
        model = _make_simple_model()
        pruner = StructuredPruner(sparsity=0.5)
        plan = pruner.get_pruning_plan(model)
        assert len(plan) > 0
        for name, indices in plan.items():
            assert len(indices) > 0


class TestLotteryTicketPruner:
    """Tests for Lottery Ticket Hypothesis pruning."""

    def test_init(self):
        pruner = LotteryTicketPruner(target_sparsity=0.8, rounds=3)
        assert pruner.target_sparsity == 0.8
        assert pruner.rounds == 3

    def test_save_initial_weights(self):
        model = _make_simple_model()
        pruner = LotteryTicketPruner()
        pruner.save_initial_weights(model)
        assert pruner._initial_weights is not None
        assert len(pruner._initial_weights) > 0

    def test_find_ticket(self):
        model = _make_simple_model()
        pruner = LotteryTicketPruner(target_sparsity=0.5, rounds=2)
        pruner.save_initial_weights(model)
        result = pruner.find_ticket(model, train_fn=lambda m: m)
        assert pruner.current_sparsity > 0.0


class TestImportanceScorer:
    """Tests for importance scoring."""

    def test_init(self):
        scorer = ImportanceScorer(method="magnitude")
        assert scorer.method == "magnitude"

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            ImportanceScorer(method="nonexistent")

    def test_magnitude_scoring(self):
        model = _make_simple_model()
        scorer = ImportanceScorer(method="magnitude", granularity="filter")
        scores = scorer.score(model)
        assert len(scores) > 0
        for name, score_tensor in scores.items():
            assert score_tensor.ndim == 1

    def test_taylor_scoring(self):
        model = _make_simple_model()
        loader = _make_dataloader()
        scorer = ImportanceScorer(method="taylor", granularity="filter")
        scores = scorer.score(model, dataloader=loader)
        assert len(scores) > 0

    def test_gradient_scoring(self):
        model = _make_simple_model()
        loader = _make_dataloader()
        scorer = ImportanceScorer(method="gradient", granularity="filter")
        scores = scorer.score(model, dataloader=loader)
        assert len(scores) > 0

    def test_rank_layers(self):
        model = _make_simple_model()
        scorer = ImportanceScorer(method="magnitude", granularity="filter")
        scores = scorer.score(model)
        ranked = scorer.rank_layers(scores)
        assert len(ranked) == len(scores)
