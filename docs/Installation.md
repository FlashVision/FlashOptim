# Installation

## Prerequisites

- Python 3.8 or higher
- pip (latest version recommended)
- CUDA 11.8+ (for GPU acceleration, optional)

## Install from PyPI

```bash
pip install flashoptim
```

## Install from Source

```bash
git clone https://github.com/FlashVision/FlashOptim.git
cd FlashOptim
pip install -e .
```

## Optional Dependencies

```bash
# ONNX export support
pip install flashoptim[export]

# Quantization extras
pip install flashoptim[quantization]

# Visualization and profiling
pip install flashoptim[analytics]

# All optional dependencies
pip install flashoptim[all]

# Development tools (testing, linting)
pip install flashoptim[dev]
```

## Environment Setup Script

For a complete development setup:

```bash
bash setup_env.sh
```

This will:
1. Create a virtual environment
2. Install all dependencies
3. Set up pre-commit hooks

## Verify Installation

```bash
flashoptim version
flashoptim check
```

## Docker

```bash
docker build -t flashoptim -f docker/Dockerfile .
docker run --gpus all -it flashoptim --help
```

## Troubleshooting

- **CUDA not found**: Ensure NVIDIA drivers and CUDA toolkit are installed
- **Import errors**: Try `pip install -e ".[all]"` for all dependencies
- **Permission denied**: Use `pip install --user` or a virtual environment
