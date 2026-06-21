"""Comprehensive test suite for FlashOptim.

Covers quantization (PTQ, QAT, GPTQ, AWQ, SmoothQuant), pruning
(unstructured, structured, SparseGPT, Wanda, N:M sparsity, lottery ticket),
knowledge distillation, feature distillation, torch.compile, NAS,
auto-optimizer, deployment profiler, registry, CLI, engine, edge cases,
and integration.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal model for testing optimization techniques."""

    def __init__(self, in_features=16, num_classes=5):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class _LinearModel(nn.Module):
    """Simple linear model for distillation tests."""

    def __init__(self, in_f=16, out_f=5):
        super().__init__()
        self.fc1 = nn.Linear(in_f, 32)
        self.fc2 = nn.Linear(32, out_f)

    def forward(self, x):
        if x.dim() == 4:
            x = x.mean(dim=(2, 3))
        if x.shape[-1] != 16:
            x = nn.functional.adaptive_avg_pool1d(x.unsqueeze(1), 16).squeeze(1)
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def tiny_model():
    return _TinyModel()


@pytest.fixture
def linear_model():
    return _LinearModel()


@pytest.fixture
def small_input():
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def calibration_loader():
    """Simple calibration loader."""
    data = [(torch.randn(2, 3, 32, 32), torch.randint(0, 5, (2,))) for _ in range(5)]
    return data


# ===========================================================================
# 1. Model / Component classes
# ===========================================================================


class TestFlashOptimModel:
    def test_instantiation(self):
        from flashoptim.models.optimized_model import FlashOptim

        inner = _TinyModel()
        model = FlashOptim(model=inner, device="cpu")
        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        from flashoptim.models.optimized_model import FlashOptim

        inner = _TinyModel()
        model = FlashOptim(model=inner, device="cpu")
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out is not None


class TestCompressedBackbone:
    def test_import(self):
        from flashoptim.models.backbone.compressed import CompressedBackbone  # noqa: F401


class TestLightweightHead:
    def test_import(self):
        from flashoptim.models.head.lightweight_head import LightweightHead  # noqa: F401


class TestEfficientNeck:
    def test_import(self):
        from flashoptim.models.neck.efficient_neck import EfficientNeck  # noqa: F401


# ===========================================================================
# 2. Registry
# ===========================================================================


class TestRegistry:
    def test_registry_import(self):
        from flashoptim.registry import Registry  # noqa: F401

    def test_register_and_build(self):
        from flashoptim.registry import Registry

        reg = Registry("test_optim")

        @reg.register("TestClass")
        class TestClass:
            def __init__(self, v=1):
                self.v = v

        obj = reg.build("TestClass", v=99)
        assert obj.v == 99

    def test_list(self):
        from flashoptim.registry import Registry

        reg = Registry("list_test")

        @reg.register("A")
        class A:
            pass

        assert "A" in reg.list()


# ===========================================================================
# 3. CLI
# ===========================================================================


class TestCLI:
    def test_cli_import(self):
        from flashoptim.cli import main  # noqa: F401

    def test_version(self):
        import flashoptim

        assert flashoptim.__version__ == "1.1.0"


# ===========================================================================
# 4. Engine
# ===========================================================================


class TestEngine:
    def test_trainer_import(self):
        from flashoptim.engine.trainer import Trainer  # noqa: F401

    def test_validator_import(self):
        from flashoptim.engine.validator import Validator  # noqa: F401

    def test_predictor_import(self):
        from flashoptim.engine.predictor import Predictor  # noqa: F401

    def test_exporter_import(self):
        from flashoptim.engine.exporter import Exporter  # noqa: F401

    def test_callbacks_import(self):
        from flashoptim.engine.callbacks import Callback, CallbackManager, EarlyStopping  # noqa: F401


# ===========================================================================
# 5. Quantization — PTQ
# ===========================================================================


class TestPTQuantizer:
    def test_instantiation(self):
        from flashoptim.quantization.ptq import PTQuantizer

        q = PTQuantizer(dtype="int8")
        assert q.dtype == "int8"

    def test_fp16_quantization(self, tiny_model):
        from flashoptim.quantization.ptq import PTQuantizer

        q = PTQuantizer(dtype="fp16")
        quantized = q.quantize(tiny_model)
        assert quantized is not None
        # fp16 model params should be half precision
        for p in quantized.parameters():
            assert p.dtype == torch.float16

    def test_invalid_dtype_raises(self):
        from flashoptim.quantization.ptq import PTQuantizer

        with pytest.raises(ValueError):
            PTQuantizer(dtype="int4")

    def test_quantize_with_calibration_loader(self, tiny_model, calibration_loader):
        from flashoptim.quantization.ptq import PTQuantizer

        q = PTQuantizer(dtype="int8", calibration_samples=5)
        quantized = q.quantize(tiny_model, calibration_loader=calibration_loader)
        assert quantized is not None

    def test_repr(self):
        from flashoptim.quantization.ptq import PTQuantizer

        q = PTQuantizer(dtype="int8", per_channel=True)
        assert "int8" in repr(q)


# ===========================================================================
# 6. Quantization — QAT
# ===========================================================================


class TestQATTrainer:
    def test_instantiation(self):
        from flashoptim.quantization.qat import QATTrainer

        qat = QATTrainer()
        assert qat is not None


# ===========================================================================
# 7. Quantization — GPTQ
# ===========================================================================


class TestGPTQQuantizer:
    def test_instantiation(self):
        from flashoptim.quantization.gptq import GPTQQuantizer

        gptq = GPTQQuantizer()
        assert gptq is not None


# ===========================================================================
# 8. Quantization — AWQ
# ===========================================================================


class TestAWQQuantizer:
    def test_instantiation(self):
        from flashoptim.quantization.awq import AWQQuantizer

        awq = AWQQuantizer()
        assert awq is not None


# ===========================================================================
# 9. Quantization — SmoothQuant
# ===========================================================================


class TestSmoothQuantizer:
    def test_instantiation(self):
        from flashoptim.quantization.smoothquant import SmoothQuantizer

        sq = SmoothQuantizer()
        assert sq is not None


# ===========================================================================
# 10. Pruning — Unstructured
# ===========================================================================


class TestUnstructuredPruner:
    def test_instantiation(self):
        from flashoptim.pruning.unstructured import UnstructuredPruner

        p = UnstructuredPruner(sparsity=0.5)
        assert p.sparsity == 0.5

    def test_prune_model(self, tiny_model):
        from flashoptim.pruning.unstructured import UnstructuredPruner

        p = UnstructuredPruner(sparsity=0.3, method="magnitude", iterative=False)
        pruned = p.prune(tiny_model)
        assert pruned is not None

    def test_get_sparsity(self, tiny_model):
        from flashoptim.pruning.unstructured import UnstructuredPruner

        p = UnstructuredPruner(sparsity=0.5, iterative=False)
        p.prune(tiny_model)
        stats = p.get_sparsity(tiny_model)
        assert "global" in stats
        assert stats["global"] > 0.0

    def test_remove_pruning(self, tiny_model):
        from flashoptim.pruning.unstructured import UnstructuredPruner

        p = UnstructuredPruner(sparsity=0.3, iterative=False)
        p.prune(tiny_model)
        cleaned = p.remove_pruning(tiny_model)
        assert cleaned is not None

    def test_random_method(self, tiny_model):
        from flashoptim.pruning.unstructured import UnstructuredPruner

        p = UnstructuredPruner(sparsity=0.3, method="random", iterative=False)
        pruned = p.prune(tiny_model)
        assert pruned is not None

    def test_invalid_sparsity(self):
        from flashoptim.pruning.unstructured import UnstructuredPruner

        with pytest.raises(ValueError):
            UnstructuredPruner(sparsity=1.5)

    def test_invalid_method(self):
        from flashoptim.pruning.unstructured import UnstructuredPruner

        with pytest.raises(ValueError):
            UnstructuredPruner(method="invalid")

    def test_iterative_pruning(self, tiny_model):
        from flashoptim.pruning.unstructured import UnstructuredPruner

        p = UnstructuredPruner(sparsity=0.5, iterative=True, iterations=3)
        pruned = p.prune(tiny_model)
        stats = p.get_sparsity(pruned)
        assert stats["global"] > 0.0


# ===========================================================================
# 11. Pruning — Structured
# ===========================================================================


class TestStructuredPruner:
    def test_instantiation(self):
        from flashoptim.pruning.structured import StructuredPruner

        p = StructuredPruner()
        assert p is not None


# ===========================================================================
# 12. Pruning — SparseGPT
# ===========================================================================


class TestSparseGPTPruner:
    def test_instantiation(self):
        from flashoptim.pruning.sparsegpt import SparseGPTPruner

        p = SparseGPTPruner()
        assert p is not None


# ===========================================================================
# 13. Pruning — Wanda
# ===========================================================================


class TestWandaPruner:
    def test_instantiation(self):
        from flashoptim.pruning.wanda import WandaPruner

        p = WandaPruner()
        assert p is not None


# ===========================================================================
# 14. Pruning — N:M Sparsity
# ===========================================================================


class TestNMSparsityPruner:
    def test_instantiation(self):
        from flashoptim.pruning.nm_sparsity import NMSparsityPruner

        p = NMSparsityPruner()
        assert p is not None


# ===========================================================================
# 15. Pruning — Lottery Ticket
# ===========================================================================


class TestLotteryTicketPruner:
    def test_instantiation(self):
        from flashoptim.pruning.lottery_ticket import LotteryTicketPruner

        p = LotteryTicketPruner()
        assert p is not None


# ===========================================================================
# 16. Knowledge Distillation
# ===========================================================================


class TestKnowledgeDistiller:
    def test_instantiation(self):
        from flashoptim.distillation.knowledge_distill import KnowledgeDistiller

        kd = KnowledgeDistiller(temperature=4.0, alpha=0.7)
        assert kd.temperature == 4.0

    def test_compute_kl_loss(self):
        from flashoptim.distillation.knowledge_distill import KnowledgeDistiller

        kd = KnowledgeDistiller(loss_type="kl_div")
        student = torch.randn(4, 10)
        teacher = torch.randn(4, 10)
        loss = kd.compute_loss(student, teacher)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_compute_mse_loss(self):
        from flashoptim.distillation.knowledge_distill import KnowledgeDistiller

        kd = KnowledgeDistiller(loss_type="mse")
        student = torch.randn(4, 10)
        teacher = torch.randn(4, 10)
        loss = kd.compute_loss(student, teacher)
        assert loss.item() >= 0.0

    def test_compute_cosine_loss(self):
        from flashoptim.distillation.knowledge_distill import KnowledgeDistiller

        kd = KnowledgeDistiller(loss_type="cosine")
        student = torch.randn(4, 10)
        teacher = torch.randn(4, 10)
        loss = kd.compute_loss(student, teacher)
        assert loss.item() >= 0.0

    def test_with_task_loss(self):
        from flashoptim.distillation.knowledge_distill import KnowledgeDistiller

        kd = KnowledgeDistiller(alpha=0.5)
        student = torch.randn(4, 5)
        teacher = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        loss = kd.compute_loss(student, teacher, targets=targets, task_loss_fn=nn.CrossEntropyLoss())
        assert loss.item() > 0.0

    def test_invalid_loss_type(self):
        from flashoptim.distillation.knowledge_distill import KnowledgeDistiller

        kd = KnowledgeDistiller(loss_type="invalid")
        with pytest.raises(ValueError):
            kd.compute_loss(torch.randn(2, 5), torch.randn(2, 5))

    def test_distill_pipeline(self):
        from flashoptim.distillation.knowledge_distill import KnowledgeDistiller

        teacher = _LinearModel()
        student = _LinearModel()
        kd = KnowledgeDistiller(temperature=3.0, alpha=0.7)

        train_data = [(torch.randn(4, 16), torch.randint(0, 5, (4,))) for _ in range(3)]
        trained = kd.distill(teacher, student, train_loader=train_data, epochs=2, device="cpu")
        assert isinstance(trained, nn.Module)


# ===========================================================================
# 17. Feature Distillation
# ===========================================================================


class TestFeatureDistiller:
    def test_instantiation(self):
        from flashoptim.distillation.feature_distill import FeatureDistiller

        fd = FeatureDistiller()
        assert fd is not None


# ===========================================================================
# 18. torch.compile Integration
# ===========================================================================


class TestTorchCompiler:
    def test_instantiation(self):
        from flashoptim.compilation.torch_compile import TorchCompiler

        c = TorchCompiler(mode="default", backend="inductor")
        assert c.mode == "default"

    def test_compile_disabled(self, tiny_model):
        from flashoptim.compilation.torch_compile import TorchCompiler

        c = TorchCompiler(disable=True)
        compiled = c.compile(tiny_model)
        assert compiled is tiny_model

    def test_compile_model(self, tiny_model):
        from flashoptim.compilation.torch_compile import TorchCompiler

        c = TorchCompiler(mode="default", backend="eager")
        compiled = c.compile(tiny_model)
        assert compiled is not None
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = compiled(x)
        assert out.shape == (2, 5)

    def test_warmup(self, tiny_model):
        from flashoptim.compilation.torch_compile import TorchCompiler

        c = TorchCompiler(disable=True)
        compiled = c.compile(tiny_model)
        x = torch.randn(2, 3, 32, 32)
        c.warmup(compiled, x, warmup_iterations=2)

    def test_benchmark(self, tiny_model):
        from flashoptim.compilation.torch_compile import TorchCompiler

        c = TorchCompiler(disable=True)
        compiled = c.compile(tiny_model)
        x = torch.randn(1, 3, 32, 32)
        stats = c.benchmark(compiled, x, n_iterations=5, warmup_iterations=2)
        assert "mean_ms" in stats
        assert "throughput_fps" in stats
        assert stats["mean_ms"] > 0

    def test_invalid_mode(self):
        from flashoptim.compilation.torch_compile import TorchCompiler

        with pytest.raises(ValueError):
            TorchCompiler(mode="invalid_mode")

    def test_invalid_backend(self):
        from flashoptim.compilation.torch_compile import TorchCompiler

        with pytest.raises(ValueError):
            TorchCompiler(backend="invalid_backend")

    def test_get_backend_info(self):
        from flashoptim.compilation.torch_compile import TorchCompiler

        info = TorchCompiler.get_backend_info()
        assert "torch_version" in info
        assert "compile_available" in info

    def test_compile_function(self):
        from flashoptim.compilation.torch_compile import TorchCompiler

        c = TorchCompiler(disable=True)

        def my_fn(x):
            return x * 2

        compiled_fn = c.compile_function(my_fn)
        result = compiled_fn(torch.tensor(3.0))
        assert result.item() == 6.0


# ===========================================================================
# 19. NAS — Search Space
# ===========================================================================


class TestSearchSpace:
    def test_instantiation(self):
        from flashoptim.nas.search_space import SearchSpace

        ss = SearchSpace()
        assert ss.num_stages == 5

    def test_sample(self):
        from flashoptim.nas.search_space import SearchSpace

        ss = SearchSpace()
        arch = ss.sample()
        assert "channels" in arch
        assert "kernel_sizes" in arch
        assert "depths" in arch
        assert "operations" in arch
        assert len(arch["channels"]) >= 2

    def test_encode_decode(self):
        from flashoptim.nas.search_space import SearchSpace

        ss = SearchSpace()
        arch = ss.sample()
        encoding = ss.encode(arch)
        decoded = ss.decode(encoding)
        assert decoded == arch

    def test_mutate(self):
        from flashoptim.nas.search_space import SearchSpace

        ss = SearchSpace()
        arch = ss.sample()
        mutated = ss.mutate(arch, prob=1.0)
        assert "channels" in mutated

    def test_crossover(self):
        from flashoptim.nas.search_space import SearchSpace

        ss = SearchSpace()
        a = ss.sample()
        b = ss.sample()
        child = ss.crossover(a, b)
        assert "channels" in child
        assert len(child["channels"]) >= 1

    def test_invalid_encoding_length(self):
        from flashoptim.nas.search_space import SearchSpace

        ss = SearchSpace()
        with pytest.raises(ValueError):
            ss.decode([0, 1, 2])  # Not divisible by 4


# ===========================================================================
# 20. NAS — Searcher
# ===========================================================================


class TestSearcher:
    def test_instantiation(self):
        from flashoptim.nas.search_space import SearchSpace
        from flashoptim.nas.searcher import Searcher

        ss = SearchSpace()
        searcher = Searcher(ss, strategy="random", max_evals=5)
        assert searcher.strategy == "random"

    def test_random_search(self):
        from flashoptim.nas.search_space import SearchSpace
        from flashoptim.nas.searcher import Searcher

        ss = SearchSpace()
        searcher = Searcher(ss, strategy="random", max_evals=3)

        class MockEval:
            def evaluate(self, arch):
                return {"score": np.random.random()}

        best = searcher.search(MockEval())
        assert best is not None
        assert searcher.best_score > 0

    def test_evolutionary_search(self):
        from flashoptim.nas.search_space import SearchSpace
        from flashoptim.nas.searcher import Searcher

        ss = SearchSpace()
        searcher = Searcher(ss, strategy="evolutionary", population_size=5, generations=2)

        class MockEval:
            def evaluate(self, arch):
                return {"score": np.random.random()}

        best = searcher.search(MockEval())
        assert best is not None
        assert len(searcher.history) > 0

    def test_invalid_strategy(self):
        from flashoptim.nas.search_space import SearchSpace
        from flashoptim.nas.searcher import Searcher

        ss = SearchSpace()
        with pytest.raises(ValueError):
            Searcher(ss, strategy="invalid")


# ===========================================================================
# 21. Solutions — Auto-Optimizer
# ===========================================================================


class TestAutoOptimizer:
    def test_instantiation(self):
        from flashoptim.solutions.auto_optimizer import AutoOptimizer

        opt = AutoOptimizer(target="edge")
        assert opt.target == "edge"

    def test_optimize_edge(self, tiny_model):
        from flashoptim.solutions.auto_optimizer import AutoOptimizer

        opt = AutoOptimizer(target="edge", prune=True, quantize=False)
        opt.optimize(tiny_model)
        report = opt.get_report()
        assert "steps" in report
        assert any(s["step"] == "pruning" for s in report["steps"])

    def test_optimize_server(self, tiny_model):
        from flashoptim.solutions.auto_optimizer import AutoOptimizer

        opt = AutoOptimizer(target="server", prune=False, quantize=False)
        opt.optimize(tiny_model)
        report = opt.get_report()
        assert report["target"] == "server"

    def test_optimize_mobile(self, tiny_model):
        from flashoptim.solutions.auto_optimizer import AutoOptimizer

        opt = AutoOptimizer(target="mobile", quantize=False)
        opt.optimize(tiny_model)
        report = opt.get_report()
        assert report["original_params"] > 0

    def test_invalid_target(self):
        from flashoptim.solutions.auto_optimizer import AutoOptimizer

        with pytest.raises(ValueError):
            AutoOptimizer(target="unknown")

    def test_get_report(self, tiny_model):
        from flashoptim.solutions.auto_optimizer import AutoOptimizer

        opt = AutoOptimizer(target="edge", quantize=False)
        opt.optimize(tiny_model)
        report = opt.get_report()
        assert "original_size_mb" in report
        assert "final_size_mb" in report


# ===========================================================================
# 22. Solutions — Deployment Profiler
# ===========================================================================


class TestDeploymentProfiler:
    def test_instantiation(self):
        from flashoptim.solutions.deployment_profiler import DeploymentProfiler

        dp = DeploymentProfiler(device="cpu", input_size=(3, 32, 32), batch_sizes=[1, 2])
        assert dp.device == "cpu"

    def test_profile(self, tiny_model):
        from flashoptim.solutions.deployment_profiler import DeploymentProfiler

        dp = DeploymentProfiler(
            device="cpu",
            input_size=(3, 32, 32),
            batch_sizes=[1, 2],
            warmup_runs=2,
            benchmark_runs=5,
        )
        result = dp.profile(tiny_model)
        assert "total_params" in result
        assert "latency_ms" in result
        assert "throughput_fps" in result
        assert "memory" in result

    def test_suggest_optimizations(self, tiny_model):
        from flashoptim.solutions.deployment_profiler import DeploymentProfiler

        dp = DeploymentProfiler(
            device="cpu",
            input_size=(3, 32, 32),
            batch_sizes=[1],
            warmup_runs=1,
            benchmark_runs=2,
        )
        result = dp.profile(tiny_model)
        suggestions = dp.suggest_optimizations(result)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_compare(self, tiny_model):
        from flashoptim.solutions.deployment_profiler import DeploymentProfiler

        dp = DeploymentProfiler(
            device="cpu",
            input_size=(3, 32, 32),
            batch_sizes=[1],
            warmup_runs=1,
            benchmark_runs=3,
        )
        result1 = dp.profile(tiny_model)
        result2 = dp.profile(tiny_model)
        comparison = dp.compare(result1, result2)
        assert "param_reduction" in comparison
        assert "size_reduction" in comparison


# ===========================================================================
# 23. Losses
# ===========================================================================


class TestLosses:
    def test_distillation_loss_import(self):
        from flashoptim.losses.distillation_loss import DistillationLoss, KLDivergenceLoss  # noqa: F401

    def test_regularization_loss_import(self):
        from flashoptim.losses.regularization_loss import SparsityLoss, L1RegularizationLoss  # noqa: F401


# ===========================================================================
# 24. Edge Cases
# ===========================================================================


class TestEdgeCases:
    def test_ptq_fp16_empty_model(self):
        from flashoptim.quantization.ptq import PTQuantizer

        class EmptyModel(nn.Module):
            def forward(self, x):
                return x

        q = PTQuantizer(dtype="fp16")
        quantized = q.quantize(EmptyModel())
        assert quantized is not None

    def test_pruner_model_without_conv_or_linear(self):
        from flashoptim.pruning.unstructured import UnstructuredPruner

        class NoPrunableModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(3)

            def forward(self, x):
                return self.bn(x)

        p = UnstructuredPruner(sparsity=0.5)
        pruned = p.prune(NoPrunableModel())
        assert pruned is not None

    def test_knowledge_distill_no_loader_raises(self):
        from flashoptim.distillation.knowledge_distill import KnowledgeDistiller

        kd = KnowledgeDistiller()
        with pytest.raises(ValueError):
            kd.distill(_LinearModel(), _LinearModel())

    def test_search_space_custom_ops(self):
        from flashoptim.nas.search_space import SearchSpace

        ss = SearchSpace(operations=["conv", "skip"])
        arch = ss.sample()
        assert all(op in ["conv", "skip"] for op in arch["operations"])

    def test_torch_compiler_reset_dynamo(self):
        from flashoptim.compilation.torch_compile import TorchCompiler

        TorchCompiler.reset_dynamo()


# ===========================================================================
# 25. Integration — end-to-end
# ===========================================================================


class TestIntegration:
    def test_prune_then_profile(self, tiny_model):
        """Prune model, then profile it."""
        from flashoptim.pruning.unstructured import UnstructuredPruner
        from flashoptim.solutions.deployment_profiler import DeploymentProfiler

        pruner = UnstructuredPruner(sparsity=0.5, iterative=False)
        pruned = pruner.prune(tiny_model)
        pruner.remove_pruning(pruned)

        dp = DeploymentProfiler(
            device="cpu",
            input_size=(3, 32, 32),
            batch_sizes=[1],
            warmup_runs=1,
            benchmark_runs=3,
        )
        result = dp.profile(pruned)
        assert result["sparsity"] > 0.0

    def test_distill_then_quantize(self):
        """Distill a model, then quantize the student."""
        from flashoptim.distillation.knowledge_distill import KnowledgeDistiller
        from flashoptim.quantization.ptq import PTQuantizer

        teacher = _LinearModel()
        student = _LinearModel()
        kd = KnowledgeDistiller(temperature=3.0, alpha=0.7)

        train_data = [(torch.randn(4, 16), torch.randint(0, 5, (4,))) for _ in range(3)]
        trained_student = kd.distill(teacher, student, train_loader=train_data, epochs=2, device="cpu")

        q = PTQuantizer(dtype="fp16")
        quantized = q.quantize(trained_student)
        assert quantized is not None

    def test_nas_then_optimize(self):
        """Run NAS, then auto-optimize the found architecture concept."""
        from flashoptim.nas.search_space import SearchSpace
        from flashoptim.nas.searcher import Searcher
        from flashoptim.solutions.auto_optimizer import AutoOptimizer

        ss = SearchSpace()
        searcher = Searcher(ss, strategy="random", max_evals=3)

        class MockEval:
            def evaluate(self, arch):
                return {"score": np.random.random()}

        best_arch = searcher.search(MockEval())
        assert best_arch is not None

        model = _TinyModel()
        opt = AutoOptimizer(target="edge", quantize=False)
        opt.optimize(model)
        report = opt.get_report()
        assert report["original_params"] > 0

    def test_compile_and_benchmark(self, tiny_model):
        """Compile model and run benchmark."""
        from flashoptim.compilation.torch_compile import TorchCompiler

        c = TorchCompiler(disable=True)
        compiled = c.compile(tiny_model)
        x = torch.randn(2, 3, 32, 32)
        stats = c.benchmark(compiled, x, n_iterations=3, warmup_iterations=1)
        assert stats["mean_ms"] > 0
        assert stats["throughput_fps"] > 0
