
# Configuration Guide

The system uses two configuration files for runtime setup:

- `config/models.yaml` (required): Registers available models and their backend adapters
- `config/routing.yaml` (optional): Configures traffic distribution between local and remote deployments

These files are independent: `models.yaml` provides the candidate set of adapters, while `routing.yaml` adjusts weights on top of registered adapters. Without `routing.yaml`, the system uses default weights from `models.yaml` or environment variables (typically 1.0).

## 1. Environment Variables and Priority

### Variable Substitution
The system supports environment variable placeholders in YAML:
- `${VAR}`: Reads `VAR` from environment
- `${VAR:-default}`: Reads `VAR` from environment, uses `default` if not found

### Configuration Priority
Environment variables (`.env` or system) > `routing.yaml`/`models.yaml` > code defaults

## 2. models.yaml (Required)

Registers models and adapters at startup. Each entry describes a model identifier and its backend configuration (base_url, api_key, capabilities, aliases, etc.).

Example:

```yaml
# config/models.yaml
models:
  - id: llama-3.3-70b-instruct
    name: Llama 3.3 70B Instruct
    provider: llama
    base_url: ${LLAMA_BASE_URL}
    api_key: ${LLAMA_API_KEY}
    context_length: 131072
    max_output_length: 8192
    supports_tools: true
    supports_structured_output: true
    supported_params: [temperature, top_p, top_k, min_p, max_tokens, stop, seed]
    aliases: ["llama-3.3-70b-instruct"]
    route:
      - kind: llama
        weight: 1.0
        base_url: ${LLAMA_BASE_URL}
        api_key: ${LLAMA_API_KEY}

  - id: llama-4-scout
    name: Llama 4 Scout
    provider: vllm
    base_url: ${LOCAL_BASE_URL}
    provider_model_id: "/models/meta-llama_Llama-4-Scout-17B-16E"  # backend expects this id
    context_length: 262144
    max_output_length: 16384
    supports_tools: true
    supports_structured_output: true
    supported_params: [temperature, top_p, top_k, min_p, max_tokens, stop, seed]
    aliases: ["/models/meta-llama_Llama-4-Scout-17B-16E"]
    route:
      - kind: vllm
        weight: 1.0
        base_url: ${LOCAL_BASE_URL}
```

### Key Points:
- `id`: Public model ID exposed by the API (what clients use to call the model)
- `provider_model_id`: The actual model name sent to the backend provider (e.g., vLLM/freeinference's `/models/...`). If omitted, uses `id`
- `aliases`: Additional public aliases that are registered alongside `id` to point to the same adapter
- `provider`: Determines adapter type (`llama`, `vllm`, `deepseek`, `gemini`, etc.)
- `/v1/models` endpoint dynamically generates its response from registered adapters

## 3. routing.yaml (Optional)

Controls traffic distribution between local and remote deployments with optional health monitoring.

### Fixed-Ratio Strategy
- Set `routing_strategy: fixed`
- Control local traffic percentage via `routing_parameter.local_fraction` (0.0–1.0)
- Weights are distributed equally within local and remote groups

### Health Checking (Optional)
- `health_check: N`: Sends GET request to `/health` every N seconds
- Unhealthy endpoints temporarily get weight 0
- Endpoints recover automatically when health checks succeed
- Set to 0 or omit to disable health checking

### Example: Hybrid Deployment (60% local / 40% remote)

```yaml
# config/routing.yaml
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

### How It Works:
- At startup, `RoutingManager` applies 60/40 weights to registered adapters
- If local endpoint becomes unhealthy, weights adjust automatically (0% local, 100% remote)
- System falls back gracefully to maintain service availability

### Local-Only Deployment

Simply omit `routing.yaml` to use default weights from `models.yaml` (typically 1.0 for all adapters).

## 4. Running the System

### Set Environment Variables:
  ```bash
  export LOCAL_BASE_URL=http://localhost:8000
  export LLAMA_BASE_URL=https://api.llama.com/compat/v1
  export LLAMA_API_KEY=sk-...
  ```

### Start the Server:
```bash
python -m serving.servers.app
# Or use uvicorn/pm2/supervisor for production
# uvicorn serving.servers.app:app --host 0.0.0.0 --port 8080
```

### Verify Operation:
- `GET /v1/models` - Returns available models
- `POST /v1/chat/completions` - Routes requests based on configured ratios
- `GET /routing` - Shows current routing configuration

## 5. FAQ

**Q: What if routing.yaml conflicts with models.yaml?**
A: `routing.yaml` only adjusts weights; it doesn't add/remove adapters. The candidate set comes from `models.yaml` and environment variables.

**Q: How to disable health checks?**
A: Set `health_check: 0` or omit the field entirely.

**Q: Can I use other routing strategies?**
A: Currently only `fixed` is built-in. You can add new strategies in `routing/strategies.py` and configure them in `routing.yaml`.

**Q: What happens during failover?**
A: The system automatically tries alternative adapters when the primary fails, ensuring continuous service availability.
