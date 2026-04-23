# Hybrid Inference Routing System

The routing system implements a two-layer architecture for intelligent traffic distribution:

## Architecture

### Decision Layer (`routing/manager.py` + `routing/strategies.py`)
Reads `config/routing.yaml` and computes weight distributions between local and remote deployments. Currently supports a fixed-ratio strategy with plans for expansion.

### Execution Layer (`routing/executor.py`)
Performs weighted random selection based on computed weights and provides automatic fallback to alternative adapters on request failure.

## Features

- **Fixed-ratio routing**: Configurable traffic split between local and remote deployments
- **Health monitoring** (optional): Simple health checks with automatic weight adjustment
- **Automatic fallback**: Seamless failover when primary adapter fails
- **Environment variable support**: Configuration with `${VAR}` and `${VAR:-default}` syntax

## Configuration

See the [Configuration guide](configuration.md) for detailed options and examples.

### Required Files
- `config/models.yaml`: Registers available models and adapters

### Optional Files
- `config/routing.yaml`: Configures local/remote deployment split and health checking

### Example Configuration (60/40 split):

```yaml
routing_strategy: fixed
routing_parameter:
  local_fraction: 0.6
timeout: 2
health_check: 30
logging:
  output: output.log
local_deployment:
  - endpoint: ${LOCAL_BASE_URL:-http://localhost:8000}
    models:
      - llama-3.3-70b-instruct
      - llama-4-scout
remote_deployment:
  - endpoint: ${LLAMA_BASE_URL}
    models:
      - llama-3.3-70b-instruct
```

## Running the Server

```bash
# Development: run FastAPI app with routing enabled
uvicorn serving.servers.app:app --host 0.0.0.0 --port 8080

# Respect production environment (see openrouter.md for systemd deployment)
PORT=9000 uvicorn serving.servers.app:app --host 0.0.0.0 --port $PORT
```

When the application starts, `serving.servers.bootstrap` loads `config/models.yaml` and optionally `config/routing.yaml`. If `routing.yaml` is present the `RoutingManager` applies the configured weights; otherwise default weights from `models.yaml` are used.

## API Endpoints

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completion with automatic routing
- `GET /routing` - View current routing configuration and weights
- `GET /health` - Health check endpoint

## Extending the System

### Adding New Strategies

1. Create a new strategy class in `routing/strategies.py`:
```python
class RoundRobinStrategy:
    def assign(self, local: List, remote: List) -> Dict[object, float]:
        # Implementation
```

2. Update `routing/manager.py` to use the new strategy based on `routing_strategy` config.

### Health Monitoring

Health checks are optional and can be enabled by setting `health_check > 0` in the configuration. The system performs simple GET requests to `/health` endpoints and adjusts weights accordingly.

## Migration Notes

For users migrating from older versions:
- The old `deployment.example.yaml` format is deprecated
- Use the simplified `config/routing.yaml` structure shown above
- Legacy `RoutingStrategy/select_deployment` patterns have been replaced with the current `FixedRatioStrategy.assign()` approach
