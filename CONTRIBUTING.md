# Contributing to FlashOptim

Thank you for your interest in contributing to FlashOptim! This guide will help you get started.

---

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/<your-username>/FlashOptim.git
cd FlashOptim
```

### 2. Set Up Development Environment

```bash
bash setup_env.sh
# or manually:
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,all]"
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

---

## Development Workflow

### Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Maximum line length: 120 characters
- Follow existing code patterns and naming conventions
- Add type hints to all public functions

### Running Tests

```bash
pytest tests/ -v
```

### Running Linter

```bash
ruff check flashoptim/
ruff format flashoptim/
```

---

## Pull Request Process

1. Ensure all tests pass and linting is clean
2. Update documentation if adding new features
3. Add an entry to `CHANGELOG.md` under `[Unreleased]`
4. Write a clear PR description explaining the change
5. Request review from a maintainer

---

## What to Contribute

- **Bug fixes** — Always welcome!
- **New optimization methods** — Quantization, pruning, distillation techniques
- **Export backends** — TensorRT, OpenVINO, CoreML support
- **Documentation** — Tutorials, examples, API docs
- **Tests** — Improve coverage
- **Benchmarks** — Performance comparisons

---

## Code of Conduct

Be respectful, constructive, and inclusive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

---

## Questions?

Open an issue on GitHub or start a discussion. We're happy to help!
