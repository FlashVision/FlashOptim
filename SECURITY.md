# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.1.x   | :white_check_mark: |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability within FlashOptim, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities.
2. Email security concerns to the maintainers directly.
3. Include a detailed description of the vulnerability and steps to reproduce.
4. Allow up to 72 hours for an initial response.

## Security Considerations

- FlashOptim loads model checkpoints using `torch.load()`. Only load models from trusted sources, as malicious pickle files can execute arbitrary code.
- When using the ONNX export/validation features, ensure onnxruntime is up to date.
- CLI commands that load YAML configs use `yaml.safe_load()` to prevent code injection.

## Best Practices

- Pin dependency versions in production deployments.
- Use `weights_only=True` with `torch.load()` when loading only state dicts from untrusted sources.
- Keep PyTorch and ONNX runtime updated to receive security patches.
