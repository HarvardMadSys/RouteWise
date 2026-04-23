# OpenRouter-Compatible API Gateway

A FastAPI-based gateway that serves OpenRouter-compatible traffic, fans out to local and remote LLM adapters, and exposes observability interfaces for operations. The application listens on port 80 in production (via `systemd`) and can also run on a developer-selectable port for local work.

## Architecture

```
hybridInference/
├── docs/                       # Deployment and integration guides
├── serving/
│   ├── servers/
│   │   ├── app.py              # FastAPI entry point (exposes /v1/*)
│   │   ├── bootstrap.py        # Service bootstrap: models, routing, DB, rate limits
│   │   └── routers/            # API routers (health, models, completions, admin)
│   ├── adapters/               # Provider adapters (local VLLM, DeepSeek, Gemini, Llama, ...)
│   ├── storage/                # Database loggers (SQLite/PostgreSQL)
│   ├── observability/          # Metrics export (Prometheus, traces)
│   └── utils/                  # Logging, configuration helpers
├── routing/                    # Routing manager and execution strategies
├── config/
│   ├── models.yaml             # Canonical model definitions + adapters
│   └── routing.yaml (optional) # Weighted routing configuration
├── infrastructure/systemd/     # Production unit files (FastAPI on port 80)
└── var/db/openrouter_logs.db   # Default SQLite request log (created at runtime)
```

### Key Components
- **FastAPI app (`serving.servers.app:create_app`)**: Hosts OpenRouter-compatible endpoints plus admin and metrics routes.
- **Bootstrap (`serving.servers.bootstrap`)**: Loads environment, registers models, applies routing weights, wires database logging, and configures rate limits.
- **Adapters (`serving.adapters.*`)**: Translate requests to providers such as local VLLM, DeepSeek, Gemini, and Llama API.
- **Routing (`routing.*`)**: Supports fixed-ratio and future strategies for splitting traffic across adapters.
- **Observability (`serving.observability.metrics`)**: Prometheus metrics and structured request logging.

## Features

- **OpenRouter API compatibility**: Implements `/v1/chat/completions`, `/v1/models`, and related schemas.
- **Hybrid routing**: Combine local VLLM workers with hosted APIs; supports hard/soft offload.
- **Resilient adapters**: Automatic retry/fallback when a provider returns errors.
- **Usage accounting**: Prompt/completion token tracking and persisted request logs.
- **Streaming responses**: Server-Sent Events (SSE) for incremental output.
- **Observability hooks**: Prometheus metrics endpoint and structured request logs (SQLite/PostgreSQL).

## Development Setup

### Prerequisites
- Python 3.10 or newer
- [uv](https://github.com/astral-sh/uv) (recommended) or conda

### Create Environment
```bash
# Clone and bootstrap
git clone <repository-url>
cd hybridInference
uv venv -p 3.10
source .venv/bin/activate
uv sync
```

### Local Environment Variables
Create `.env` from the template:
```bash
cp .env.example .env
```
Populate it with provider credentials and runtime configuration:
```bash
LOCAL_BASE_URL=https://freeinference.org/v1
OFFLOAD=0
DEEPSEEK_API_KEY=your-deepseek-api-key
GEMINI_API_KEY=your-gemini-api-key
LLAMA_API_KEY=your-llama-api-key
LLAMA_BASE_URL=https://your-llama-api-base/v1
USE_SQLITE_LOG=true
```

### Run Locally
```bash
# Development server with reload on port 8080
uvicorn serving.servers.app:app --reload --host 0.0.0.0 --port 8080

# Alternate: respect PORT env var
PORT=9000 uvicorn serving.servers.app:app --host 0.0.0.0 --port $PORT
```

When the app starts it will:
1. Load environment variables (dotenv).
2. Register models from `config/models.yaml`.
3. Apply routing overrides from `config/routing.yaml` if present.
4. Initialize the database logger (SQLite under `var/db` by default).
5. Configure per-provider rate limits when API keys are supplied.

### Quick Checks
```bash
# Health
curl http://localhost:8080/health

# Models (OpenRouter schema)
curl http://localhost:8080/v1/models | jq

# Chat completion
env \
  http_proxy= \
  curl -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
          "model": "llama-4-scout",
          "messages": [{"role": "user", "content": "Ping"}],
          "max_tokens": 64
        }'
```

## Production Deployment

We run the service directly on port 80 under `systemd`, letting Cloudflare terminate TLS at the edge. The repository ships a maintained unit file at `infrastructure/systemd/hybrid_inference.service`.

```bash
# Copy the unit file
sudo cp infrastructure/systemd/hybrid_inference.service \
        /etc/systemd/system/freeinference.service

# Reload systemd and enable on boot
sudo systemctl daemon-reload
sudo systemctl enable freeinference.service
sudo systemctl start freeinference.service
sudo systemctl status freeinference.service
```

Runtime operations:
- Restart: `sudo systemctl restart freeinference.service`
- Logs: `journalctl -u freeinference.service -f`
- Health: `curl https://freeinference.org/health`

## API Surface

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/v1/models` | Enumerate available models with OpenRouter metadata |
| POST | `/v1/chat/completions` | OpenRouter/OpenAI-compatible chat completion |
| GET | `/health` | Liveness and dependency checks |
| GET | `/metrics` | Prometheus metrics (requires auth upstream) |
| GET | `/routing` | Current routing weights (admin scope) |
| GET | `/stats` | Aggregated usage statistics |

### Example Requests
```bash
# Streaming response
env \
  http_proxy= \
  curl -N -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
          "model": "deepseek-chat",
          "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Describe the architecture."}
          ],
          "stream": true,
          "temperature": 0.7,
          "max_tokens": 256
        }'
```

## Logging and Metrics

- **SQLite (default)**: When `USE_SQLITE_LOG=true`, logs persist to `var/db/openrouter_logs.db`. Override with `SQLITE_DB_PATH` or `OPENROUTER_SQLITE_DB`.
- **PostgreSQL**: Set `USE_SQLITE_LOG=false` and `DATABASE_URL=<dsn>` to stream logs into PostgreSQL for analytics.
- **Metrics**: `/metrics` exposes Prometheus counters/latencies. Enable scraping through infrastructure (e.g., Prometheus + Grafana).

Inspect logs locally:
```bash
python scripts/view_logs.py
sqlite3 var/db/openrouter_logs.db 'SELECT model_id, COUNT(*) FROM api_logs GROUP BY model_id;'
```

## Testing

```bash
# Fast unit/integration tests
pytest -m "not external" -q

# Focused server tests
pytest test/servers/test_bootstrap.py -q
```

## Troubleshooting

- **Port already in use**: `sudo lsof -ti :80 | xargs sudo kill -9`
- **Missing models**: Verify `config/models.yaml` contains the expected entries and that `LOCAL_BASE_URL` is reachable.
- **No logs written**: Confirm `USE_SQLITE_LOG` and filesystem permissions for `var/db/`.
- **Provider rate limiting**: Adjust `GEMINI_TPM_LIMIT`, `DEEPSEEK_TPM_LIMIT`, or equivalent environment variables as needed.

## Related Docs

- [FreeInference Deployment](freeinference.md): Deployment history and rationale for running FastAPI directly on port 80.
- [Routing](routing.md): Detailed routing manager configuration and strategy extension guide.
- [Adding Models](adding-models.md): How to add new models (YAML) and integrate new providers (adapter) in one place.
