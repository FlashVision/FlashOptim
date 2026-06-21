"""FlashOptim Command-Line Interface.

Usage:
    flashoptim version
    flashoptim settings
    flashoptim check
    flashoptim quantize --config <path>
    flashoptim prune --config <path>
    flashoptim distill --config <path>
    flashoptim nas --config <path>
    flashoptim export --model <path> --format <fmt>
    flashoptim benchmark --model <path> --device <dev>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


BANNER = r"""
  _____ _           _      ___        _   _
 |  ___| | __ _ ___| |__  / _ \ _ __ | |_(_)_ __ ___
 | |_  | |/ _` / __| '_ \| | | | '_ \| __| | '_ ` _ \
 |  _| | | (_| \__ \ | | | |_| | |_) | |_| | | | | | |
 |_|   |_|\__,_|___/_| |_|\___/| .__/ \__|_|_| |_| |_|
                                |_|
"""


def cmd_version(args: argparse.Namespace) -> None:
    """Print FlashOptim version."""
    from flashoptim import __version__

    print(f"FlashOptim v{__version__}")


def cmd_settings(args: argparse.Namespace) -> None:
    """Display current settings and environment info."""
    import torch

    from flashoptim import __version__

    print(BANNER)
    print(f"  FlashOptim:  v{__version__}")
    print(f"  PyTorch:     v{torch.__version__}")
    print(f"  CUDA:        {'v' + torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
    print(f"  Python:      {sys.version.split()[0]}")
    print(f"  Platform:    {sys.platform}")
    print()


def cmd_check(args: argparse.Namespace) -> None:
    """Run system checks and verify installation."""
    import importlib

    print("FlashOptim System Check")
    print("=" * 40)

    checks = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("yaml", "PyYAML"),
        ("PIL", "Pillow"),
        ("tqdm", "tqdm"),
    ]

    optional = [
        ("onnx", "ONNX"),
        ("onnxruntime", "ONNXRuntime"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
    ]

    all_ok = True
    for module, name in checks:
        try:
            mod = importlib.import_module(module)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {name:15s} v{ver}")
        except ImportError:
            print(f"  [FAIL] {name:15s} NOT INSTALLED")
            all_ok = False

    print()
    print("Optional Dependencies:")
    for module, name in optional:
        try:
            mod = importlib.import_module(module)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {name:15s} v{ver}")
        except ImportError:
            print(f"  [--] {name:15s} not installed")

    print()
    if all_ok:
        print("All required dependencies OK!")
    else:
        print("Some dependencies are missing. Run: pip install flashoptim[all]")


def cmd_quantize(args: argparse.Namespace) -> None:
    """Run quantization pipeline."""
    import yaml

    print(f"Running quantization with config: {args.config}")
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    quant_cfg = config.get("quantization", config)
    dtype = quant_cfg.get("dtype", "int8")
    model_path = quant_cfg.get("model", config.get("model", {}).get("path"))
    calibration_data = quant_cfg.get("calibration_data")
    output_path = quant_cfg.get("output", "quantized_model.pth")

    print(f"  Dtype: {dtype}")

    import torch

    from flashoptim.quantization import PTQuantizer

    quantizer = PTQuantizer(
        dtype=dtype,
        calibration_samples=quant_cfg.get("calibration_samples", 500),
        per_channel=quant_cfg.get("per_channel", True),
        symmetric=quant_cfg.get("symmetric", True),
        observer=quant_cfg.get("observer", "minmax"),
    )

    if model_path:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(model, dict) and "model" in model:
            model = model["model"]
        quantized = quantizer.quantize(model, calibration_data=calibration_data)
        torch.save(quantized, output_path)
        print(f"  Saved quantized model to: {output_path}")
    else:
        print("  Error: No model path specified in config")
        sys.exit(1)


def cmd_prune(args: argparse.Namespace) -> None:
    """Run pruning pipeline."""
    import yaml

    print(f"Running pruning with config: {args.config}")
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    prune_cfg = config.get("pruning", config)
    method = prune_cfg.get("method", "unstructured")
    sparsity = prune_cfg.get("sparsity", 0.5)
    model_path = prune_cfg.get("model", config.get("model", {}).get("path"))
    output_path = prune_cfg.get("output", "pruned_model.pth")

    print(f"  Method:   {method}")
    print(f"  Sparsity: {sparsity}")

    import torch

    from flashoptim.pruning import StructuredPruner, UnstructuredPruner

    if model_path:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(model, dict) and "model" in model:
            model = model["model"]

        if method == "structured":
            pruner = StructuredPruner(sparsity=sparsity)
        else:
            pruner = UnstructuredPruner(sparsity=sparsity)

        pruned = pruner.prune(model)
        torch.save(pruned, output_path)
        print(f"  Saved pruned model to: {output_path}")
    else:
        print("  Error: No model path specified in config")
        sys.exit(1)


def cmd_distill(args: argparse.Namespace) -> None:
    """Run distillation pipeline."""
    import yaml

    print(f"Running distillation with config: {args.config}")
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    distill_cfg = config.get("distillation", config)
    method = distill_cfg.get("method", "knowledge")
    temperature = distill_cfg.get("temperature", 4.0)
    alpha = distill_cfg.get("alpha", 0.7)

    print(f"  Method:      {method}")
    print(f"  Temperature: {temperature}")

    from flashoptim.distillation import KnowledgeDistiller

    distiller = KnowledgeDistiller(temperature=temperature, alpha=alpha)
    print(f"  Distiller created: {distiller}")
    print("  Note: Provide train_loader via Python API for actual training.")


def cmd_nas(args: argparse.Namespace) -> None:
    """Run Neural Architecture Search."""
    import yaml

    print(f"Running NAS with config: {args.config}")
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    nas_cfg = config.get("nas", config)
    strategy = nas_cfg.get("strategy", "random")
    print(f"  Strategy: {strategy}")
    print("  NAS execution requires Python API with search space definition.")


def cmd_export(args: argparse.Namespace) -> None:
    """Export an optimized model."""
    import torch

    from flashoptim.engine.exporter import Exporter

    print(f"Exporting model: {args.model}")
    print(f"  Format: {args.format}")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    model = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if isinstance(model, dict) and "model" in model:
        model = model["model"]

    output_path = model_path.with_suffix(f".{args.format}")
    exporter = Exporter(model=model)
    result = exporter.export(output_path, format=args.format)
    print(f"  Exported to: {result}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark an optimized model."""
    print(f"Benchmarking model: {args.model}")
    print(f"  Device: {args.device}")

    from flashoptim.engine.predictor import Predictor

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    predictor = Predictor(model_path=model_path, device=args.device)
    results = predictor.benchmark(runs=args.runs)

    print(f"\nBenchmark Results ({args.runs} runs):")
    print(f"  Mean latency: {results['mean_ms']:.3f} ms")
    print(f"  Std latency:  {results['std_ms']:.3f} ms")
    print(f"  Throughput:   {results['fps']:.1f} FPS")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="flashoptim",
        description="FlashOptim — Model optimization toolkit for FlashVision",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # version
    subparsers.add_parser("version", help="Show FlashOptim version")

    # settings
    subparsers.add_parser("settings", help="Display settings and environment")

    # check
    subparsers.add_parser("check", help="Check installation and dependencies")

    # quantize
    p_quant = subparsers.add_parser("quantize", help="Quantize a model")
    p_quant.add_argument("--config", type=str, required=True, help="Path to quantization config YAML")

    # prune
    p_prune = subparsers.add_parser("prune", help="Prune a model")
    p_prune.add_argument("--config", type=str, required=True, help="Path to pruning config YAML")

    # distill
    p_distill = subparsers.add_parser("distill", help="Run knowledge distillation")
    p_distill.add_argument("--config", type=str, required=True, help="Path to distillation config YAML")

    # nas
    p_nas = subparsers.add_parser("nas", help="Run Neural Architecture Search")
    p_nas.add_argument("--config", type=str, required=True, help="Path to NAS config YAML")

    # export
    p_export = subparsers.add_parser("export", help="Export an optimized model")
    p_export.add_argument("--model", type=str, required=True, help="Path to optimized model")
    p_export.add_argument("--format", type=str, default="onnx", choices=["onnx", "tensorrt", "openvino", "coreml"])

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark an optimized model")
    p_bench.add_argument("--model", type=str, required=True, help="Path to model file")
    p_bench.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p_bench.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    p_bench.add_argument("--runs", type=int, default=100, help="Benchmark iterations")

    return parser


def main() -> None:
    """FlashOptim CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "version": cmd_version,
        "settings": cmd_settings,
        "check": cmd_check,
        "quantize": cmd_quantize,
        "prune": cmd_prune,
        "distill": cmd_distill,
        "nas": cmd_nas,
        "export": cmd_export,
        "benchmark": cmd_benchmark,
    }

    if args.command is None:
        print(BANNER)
        parser.print_help()
        return

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
