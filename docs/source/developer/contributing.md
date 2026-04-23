# Contributing

We welcome contributions to HybridInference!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/hybridInference.git
   cd hybridInference
   ```

3. Set up development environment:
   ```bash
   make setup-dev
   ```

## Code Quality Standards

This project follows industry best practices:

- **Code Style**: Google Python Style Guide (formatted with ruff)
- **Linting**: ruff with extensive rule sets
- **Type Checking**: mypy with strict mode
- **Documentation**: Google-style docstrings (pydocstyle)

### Pre-commit Hooks

Pre-commit hooks run automatically on git commit:

```bash
# Install pre-commit hooks (done by make setup-dev)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run quality checks:
   ```bash
   make all  # Format, lint, typecheck, and test
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. Push to your fork and submit a pull request

## Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
uv run pytest test/test_routing.py
```

## Documentation

Update documentation when adding new features:

```bash
# Build documentation locally
cd docs
make html

# View in browser
open build/html/index.html
```

## Pull Request Guidelines

- Write clear, descriptive commit messages
- Update documentation for new features
- Add tests for new functionality
- Ensure all CI checks pass
- Keep PRs focused on a single feature or fix
