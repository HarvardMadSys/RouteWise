# hybridInference

A high-performance hybrid inference server providing local deployment and offline API access to various LLM providers.

**[User Documentation](https://doc.freeinference.org/)** | **[Developer Documentation](https://internaldoc.freeinference.org/)** | [Quick Start](#quick-start-with-uv-recommended)

## Project Structure

```
hybridInference/
├── serving/        # FastAPI gateway, adapters, observability, storage
├── routing/        # Routing manager and execution strategies
├── config/         # Model + routing configuration files
├── infrastructure/ # Systemd units, observability manifests, deployment assets
├── scripts/        # Operational and perf tooling
├── docs/           # Architecture and integration guides
├── var/            # Runtime artifacts (e.g., SQLite logs)
├── test/           # Test suite
└── client/         # Client tooling (loaders, runners, metrics)
```

For service-specific deployment and routing details, refer to `docs/openrouter.md`, `docs/freeinference.md`, and `docs/routing.md`. For extension guides (adding models or new providers), see `docs/adding_models.md`.

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or conda

### Quick Start with uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/HarvardSys/hybridInference.git
cd hybridInference

# Set up development environment
make setup-dev

# Or manually:
uv venv -p 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### Alternative: conda Setup

```bash
# Create and activate conda environment
conda create -n hybrid_inference python=3.10 -y
conda activate hybrid_inference

# Install dependencies from pyproject.toml
pip install -e .
```

## Package Management

This project uses `pyproject.toml` for dependency management (PEP 517/518 standard).

### Adding Dependencies

```bash
# Add runtime dependency
uv add fastapi httpx pydantic

# Add development dependency
uv add --group dev ruff mypy pydocstyle pytest

# Update a package
uv add fastapi --upgrade

# Sync all dependencies
uv sync
```

### Development Workflow

```bash
# Format code
make format

# Run linters
make lint

# Type checking
make typecheck

# Run tests
make test

# Run all checks
make check

# Clean build artifacts
make clean
```

## Configuration

### 1. Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

**IMPORTANT: Security Configuration**

Edit `.env` and configure:

1. **API Keys** (for external providers):
```env
OPENAI_API_KEY=your-actual-openai-api-key
LLAMA_API_KEY=your-actual-llama-api-key
GEMINI_API_KEY=your-actual-gemini-api-key
```

2. **Database Credentials** (required, no defaults):
```env
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_secure_password
```

⚠️ **Security Note**: Database credentials are **required** and have no default values. Services will fail to start without proper configuration.

### 2. Config Local Model

**Configure in `config/models.yaml`:**
```yaml
route:
  - kind: vllm
    base_url: <your-local-model-url>  # for example, http://localhost:8001/v1
    provider_model_id: "<your-model-id>"  # must match served model name
```

**For SGLang:** Use `base_url: http://localhost:30000` and `kind: sglang` or `kind: openai_compat`

## Code Quality Standards

This project follows industry best practices:

- **Code Style**: Google Python Style Guide (formatted with ruff)
- **Linting**: ruff with extensive rule sets
- **Type Checking**: mypy with strict mode
- **Documentation**: Google-style docstrings (pydocstyle)
- **Pre-commit Hooks**: Automated quality checks
- **Security**: Secret scanning with gitleaks

### Pre-commit Hooks

Pre-commit hooks run automatically on git commit:

```bash
# Install pre-commit hooks (done by make setup-dev)
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Skip hooks temporarily
git commit --no-verify
```

## Testing

```bash
# Run all tests
make test

# Verbose output
make test-verbose

# With coverage
make test-cov

# Specific test file
uv run pytest test/test_routing.py

# Run tests with markers
uv run pytest -m "not slow"  # Skip slow tests
uv run pytest -m integration  # Only integration tests
```

## Documentation

- **User Documentation**: [https://doc.freeinference.org/](https://doc.freeinference.org/) - API reference, quick start guides, and usage examples
- **Developer Documentation**: [https://internaldoc.freeinference.org/](https://internaldoc.freeinference.org/) - Architecture, deployment, and contribution guides

### Building Documentation Locally

```bash
cd docs
make html

# View the documentation
open build/html/index.html  # macOS
# or
xdg-open build/html/index.html  # Linux
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Run `make all` to format and validate code
4. Submit a pull request

## License

Proprietary - All rights reserved
