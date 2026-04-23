# Installation

Detailed installation instructions for HybridInference.

## System Requirements

- Python 3.10 or higher
- GPU support (recommended for local inference)
- Linux or macOS (Windows via WSL2)

## Installation Methods

### Using uv (Recommended)

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

### Using conda

```bash
# Create and activate conda environment
conda create -n hybrid_inference python=3.10 -y
conda activate hybrid_inference

# Install dependencies
pip install -e .
```

## Configuration

### Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Required configuration:

1. **API Keys** (for external providers):
   ```bash
   OPENAI_API_KEY=your-actual-openai-api-key
   LLAMA_API_KEY=your-actual-llama-api-key
   GEMINI_API_KEY=your-actual-gemini-api-key
   ```

2. **Database Credentials**:
   ```bash
   DB_NAME=your_database_name
   DB_USER=your_database_user
   DB_PASSWORD=your_secure_password
   ```

## Verification

Run tests to verify installation:

```bash
make test
```

## Troubleshooting

### Common Issues

- **Import errors**: Ensure you've activated the virtual environment
- **Database connection**: Verify PostgreSQL is running and credentials are correct
- **GPU issues**: Check CUDA installation and driver compatibility
