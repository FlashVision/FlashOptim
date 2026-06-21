"""Tests for SparseGPT, Wanda, and N:M sparsity pruning methods."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from flashoptim.pruning import SparseGPTPruner, WandaPruner, NMSparsityPruner


def _make_simple_model():
    """Create a simple model for pruning testing."""
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )


def _make_calibration_loader(n_samples=20, input_dim=32):
    """Create calibration DataLoader."""
    data = torch.randn(n_samples, input_dim)
    labels = torch.randint(0, 10, (n_samples,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=4)


class TestSparseGPTPruner:
    """Tests for SparseGPT pruning."""

    def test_init_default(self):
        pruner = SparseGPTPruner()
        assert pruner.sparsity == 0.5

    def test_invalid_sparsity(self):
        with pytest.raises(ValueError, match="Sparsity must be"):
            SparseGPTPruner(sparsity=1.5)

    def test_prune(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        pruner = SparseGPTPruner(sparsity=0.3, block_size=16, calibration_samples=10)
        pruned = pruner.prune(model, loader)
        assert pruned is not None
        x = torch.randn(2, 32)
        out = pruned(x)
        assert out.shape == (2, 10)

    def test_sparsity_achieved(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        pruner = SparseGPTPruner(sparsity=0.5, block_size=16, calibration_samples=10)
        pruner.prune(model, loader)
        sparsity = SparseGPTPruner.get_sparsity(model)
        assert sparsity["global"] > 0.0

    def test_stats(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        pruner = SparseGPTPruner(sparsity=0.3, block_size=16, calibration_samples=10)
        pruner.prune(model, loader)
        stats = pruner.stats
        assert len(stats) > 0

    def test_repr(self):
        pruner = SparseGPTPruner(sparsity=0.5)
        r = repr(pruner)
        assert "SparseGPTPruner" in r


class TestWandaPruner:
    """Tests for Wanda pruning."""

    def test_init_default(self):
        pruner = WandaPruner()
        assert pruner.sparsity == 0.5
        assert pruner.pruning_scope == "row"

    def test_invalid_sparsity(self):
        with pytest.raises(ValueError, match="Sparsity must be"):
            WandaPruner(sparsity=0.0)

    def test_invalid_scope(self):
        with pytest.raises(ValueError, match="Unknown scope"):
            WandaPruner(pruning_scope="invalid")

    def test_prune_row_scope(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        pruner = WandaPruner(sparsity=0.5, pruning_scope="row", calibration_samples=10)
        pruned = pruner.prune(model, loader)
        x = torch.randn(2, 32)
        out = pruned(x)
        assert out.shape == (2, 10)

    def test_prune_layer_scope(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        pruner = WandaPruner(sparsity=0.3, pruning_scope="layer", calibration_samples=10)
        pruned = pruner.prune(model, loader)
        x = torch.randn(2, 32)
        out = pruned(x)
        assert out.shape == (2, 10)

    def test_prune_global_scope(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        pruner = WandaPruner(sparsity=0.3, pruning_scope="global", calibration_samples=10)
        pruned = pruner.prune(model, loader)
        x = torch.randn(2, 32)
        out = pruned(x)
        assert out.shape == (2, 10)

    def test_sparsity_achieved(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        pruner = WandaPruner(sparsity=0.5, calibration_samples=10)
        pruner.prune(model, loader)
        sparsity = WandaPruner.get_sparsity(model)
        assert sparsity["global"] > 0.0

    def test_nm_pruning(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        pruner = WandaPruner(sparsity=0.5, use_nm=True, n=2, m=4, calibration_samples=10)
        pruned = pruner.prune(model, loader)
        x = torch.randn(2, 32)
        out = pruned(x)
        assert out.shape == (2, 10)

    def test_stats(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        pruner = WandaPruner(sparsity=0.5, calibration_samples=10)
        pruner.prune(model, loader)
        stats = pruner.stats
        assert len(stats) > 0

    def test_repr(self):
        pruner = WandaPruner(sparsity=0.5)
        r = repr(pruner)
        assert "WandaPruner" in r


class TestNMSparsityPruner:
    """Tests for N:M structured sparsity pruning."""

    def test_init_default(self):
        pruner = NMSparsityPruner()
        assert pruner.n == 2
        assert pruner.m == 4
        assert pruner.sparsity == 0.5

    def test_invalid_criterion(self):
        with pytest.raises(ValueError, match="Unknown criterion"):
            NMSparsityPruner(criterion="invalid")

    def test_invalid_nm(self):
        with pytest.raises(ValueError, match="N must be < M"):
            NMSparsityPruner(n=4, m=4)

    def test_prune_magnitude(self):
        model = _make_simple_model()
        pruner = NMSparsityPruner(n=2, m=4, criterion="magnitude")
        pruned = pruner.prune(model)
        x = torch.randn(2, 32)
        out = pruned(x)
        assert out.shape == (2, 10)

    def test_prune_wanda(self):
        model = _make_simple_model()
        loader = _make_calibration_loader()
        pruner = NMSparsityPruner(n=2, m=4, criterion="wanda", calibration_samples=10)
        pruned = pruner.prune(model, loader)
        x = torch.randn(2, 32)
        out = pruned(x)
        assert out.shape == (2, 10)

    def test_sparsity_pattern(self):
        model = _make_simple_model()
        pruner = NMSparsityPruner(n=2, m=4)
        pruner.prune(model)
        sparsity = NMSparsityPruner.get_sparsity(model)
        assert sparsity["global"] >= 0.4  # Should be close to 0.5

    def test_verify_pattern(self):
        model = _make_simple_model()
        pruner = NMSparsityPruner(n=2, m=4)
        pruner.prune(model)
        verification = pruner.verify_pattern(model)
        assert len(verification) > 0

    def test_masks_stored(self):
        model = _make_simple_model()
        pruner = NMSparsityPruner(n=2, m=4)
        pruner.prune(model)
        assert len(pruner.masks) > 0

    def test_1_4_sparsity(self):
        model = _make_simple_model()
        pruner = NMSparsityPruner(n=1, m=4)
        pruned = pruner.prune(model)
        x = torch.randn(2, 32)
        out = pruned(x)
        assert out.shape == (2, 10)
        assert pruner.sparsity == 0.25

    def test_repr(self):
        pruner = NMSparsityPruner(n=2, m=4)
        r = repr(pruner)
        assert "NMSparsityPruner" in r
        assert "2" in r
        assert "4" in r
