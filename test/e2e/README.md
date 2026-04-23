# E2E Tests for Local Deployment Models

End-to-end testing suite for hybridInference local model deployment, covering direct local access, gateway routing, and hybrid (local + remote) routing strategies.

---

## Quick Start (TL;DR)

**Just want to run tests? Here's the fastest way:**

```bash
cd test/e2e

# 1. Verify local models are running
curl http://localhost:8001/v1/models  # Llama 3.2 3B
curl http://localhost:8003/v1/models  # Qwen3 Coder 30B

# 2. Run all tests (fully automated)
make test-all-auto
```

**That's it!** Gateway will auto-start, tests will run, and everything cleans up automatically.

<details>
<summary>Step-by-step guide (click to expand)</summary>

### Step 1: Test local models directly
```bash
make test-phase1
```

### Step 2: Test gateway routing (automated)
```bash
make test-phase2-auto
```

### Step 3: Test hybrid routing (automated)
```bash
make test-phase3-auto
```

**Need to debug?** Use manual mode:
```bash
# Terminal 1: Start gateway
make start-gateway-phase3

# Terminal 2: Run tests
make test-phase3
```

</details>

---

## Overview

The E2E test suite is organized into three progressive phases:

| Phase | Description | Dependencies | Test File |
|-------|-------------|--------------|-----------|
| **Phase 1** | Direct local model testing | Local models (8001, 8003) | `test_phase1_direct_local.py` |
| **Phase 2** | Gateway routing to local models | Phase 1 + Gateway | `test_phase2_gateway_routing.py` |
| **Phase 3** | Hybrid routing (40% local + 60% remote) | Phase 2 + Chutes API | `test_phase3_hybrid_routing.py` |

---

## Command Reference

### Automated Testing (Recommended)

| Command | Description | Prerequisites |
|---------|-------------|---------------|
| `make test-phase1` | Test local models directly | Local models running on 8001, 8003 |
| `make test-phase2-auto` | Test gateway routing (auto start/stop) | Local models running |
| `make test-phase3-auto` | Test hybrid routing (auto start/stop) | Local models + `CHUTES_API_KEY` in `.env` |
| `make test-all-auto` | Run all phases (fully automated) | All of the above |

### Manual Testing (for debugging)

| Command | Description | When to use |
|---------|-------------|-------------|
| `make start-gateway-phase2` | Start gateway in foreground (Phase 2) | When you need to see gateway logs |
| `make start-gateway-phase3` | Start gateway in foreground (Phase 3) | When debugging gateway issues |
| `make test-phase2` | Run Phase 2 tests (requires gateway) | After starting gateway manually |
| `make test-phase3` | Run Phase 3 tests (requires gateway) | After starting gateway manually |
| `make stop-gateway` | Stop any running gateway | Clean up after manual testing |

### Utilities

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make clean` | Clean up test artifacts and logs |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUTES_API_KEY` | From `.env` | Chutes API key (required for Phase 3) |
| `CHUTES_BASE_URL` | From `.env` | Chutes API base URL |
| `GATEWAY_PORT` | `10081` | Gateway port (can override) |

**Example:**
```bash
# Use custom port
GATEWAY_PORT=8080 make test-phase2-auto

# Override Chutes URL
CHUTES_BASE_URL=https://custom.api make test-phase3-auto
```

---

## Configuration

All configuration is centralized in `test/fixtures/e2e_config.yaml`:

```yaml
local_servers:          # Phase 1: Direct local model configs
  llama_3_2_3b: ...
  qwen3_coder_30b: ...

models:                 # Phase 2/3: Gateway model registry
  - id: llama-3.2-3b-local-test
  - id: qwen3-coder-30b
  - id: qwen3-coder-30b-hybrid    # 40% local + 60% Chutes
  - id: qwen3-coder-30b-local-only
  - id: qwen3-coder-30b-remote-only

test_config:            # Test parameters
  gateway: ...
  routing_distribution: ...
  defaults: ...
```

## Phase Details

### Phase 1: Direct Local Model Testing

**Purpose:** Validate local SGLang/vLLM servers work correctly without gateway.

**Tests:**
- ✅ `/v1/models` endpoint (OpenAI compatibility)
- ✅ Non-streaming chat completions
- ✅ Streaming chat completions
- ✅ Model-specific validation (code generation for Qwen3)

**Run:**
```bash
make test-phase1

# Or directly
uv run pytest test_phase1_direct_local.py -v -s
```

**Expected Output:**
```
test_phase1_direct_local.py::test_models_endpoint[llama-3.2-3b-local] PASSED
test_phase1_direct_local.py::test_models_endpoint[qwen3-coder-30b-local] PASSED
test_phase1_direct_local.py::test_non_streaming_completion[llama-3.2-3b-local] PASSED
test_phase1_direct_local.py::test_non_streaming_completion[qwen3-coder-30b-local] PASSED
...
```

---

### Phase 2: Gateway Routing to Local Models

**Purpose:** Validate gateway correctly routes requests to local models.

**Tests:**
- ✅ Gateway health check
- ✅ Model registry exposure
- ✅ Non-streaming routing (per model)
- ✅ Streaming routing (per model)
- ✅ Error handling (invalid model, malformed requests)

**Setup:**
```bash
# Terminal 1: Start gateway
make start-gateway-phase2

# Terminal 2: Run tests
make test-phase2
```

**Expected Output:**
```
test_phase2_gateway_routing.py::test_local_model_non_streaming[llama-3.2-3b-local-test] PASSED
test_phase2_gateway_routing.py::test_local_model_non_streaming[qwen3-coder-30b] PASSED
test_phase2_gateway_routing.py::test_local_model_streaming[llama-3.2-3b-local-test] PASSED
test_phase2_gateway_routing.py::test_local_model_streaming[qwen3-coder-30b] PASSED
...
```

---

### Phase 3: Hybrid Routing (40% Local + 60% Remote)

**Purpose:** Validate hybrid routing distributes requests between local and remote backends.

**Tests:**
- ✅ Hybrid model registration
- ✅ Basic completion with hybrid routing
- ✅ Streaming with hybrid routing
- ✅ Routing distribution (100 requests, validate success rate)
- ✅ Comparison (local-only vs hybrid vs remote-only)

**Setup (Automated Mode - Recommended):**
```bash
# Ensure CHUTES_API_KEY is in ../../.env file
# Then just run:
make test-phase3-auto
```

**Setup (Manual Mode - for debugging):**
```bash
# Set Chutes API key
export CHUTES_API_KEY=your-key

# Terminal 1: Start gateway
make start-gateway-phase3

# Terminal 2: Run tests
make test-phase3
```

**Expected Output:**
```
test_phase3_hybrid_routing.py::test_hybrid_model_available PASSED
test_phase3_hybrid_routing.py::test_hybrid_basic_completion PASSED
test_phase3_hybrid_routing.py::test_hybrid_streaming PASSED
test_phase3_hybrid_routing.py::test_routing_distribution PASSED
  Total Requests:    100
  Successful:        100 (100.0%)
  Failed:            0
...
```

---

## Troubleshooting

### Local Models Not Reachable

**Symptom:** Tests skipped with "Server not reachable"

**Solution:**
```bash
# Check if models are running
curl http://localhost:8001/v1/models
curl http://localhost:8003/v1/models

# Start models if needed (use your SGLang/vLLM startup scripts)
```

### Gateway Not Starting (Automated Mode)

**Symptom:** Tests fail immediately after "Starting gateway in background"

**Solution:**
```bash
# Check gateway logs
cat /tmp/hybridInference_gateway_10081.log

# Common issues:
# 1. Port already in use
make stop-gateway

# 2. Missing dependencies
cd ../.. && uv sync

# 3. Config file not found
ls -la test/fixtures/e2e_config.yaml
```

### Gateway Not Starting (Manual Mode)

**Symptom:** "Address already in use" or port conflict

**Solution:**
```bash
# Check what's using the port
lsof -i :10081

# Kill existing process
make stop-gateway

# Or use different port
GATEWAY_PORT=10082 make start-gateway-phase2
```

### Phase 3 Remote-Only Fails

**Symptom:** `qwen3-coder-30b-remote-only` returns HTTP 500

**Solution:**
- This is expected if Chutes API key is invalid/expired
- Hybrid model should still work (has local fallback)
- Verify API key: `echo $CHUTES_API_KEY`

### Tests Fail with "Model not available"

**Symptom:** Model ID not found in gateway

**Solution:**
```bash
# Verify gateway loaded correct config
curl http://localhost:10081/v1/models | jq '.data[].id'

# Should see (Phase 2):
# - llama-3.2-3b-local-test
# - qwen3-coder-30b

# Or (Phase 3):
# - qwen3-coder-30b-hybrid
# - qwen3-coder-30b-local-only
# - qwen3-coder-30b-remote-only
```

---

## Test Results Interpretation

### Success Criteria

- **Phase 1:** All local models respond correctly (7+ tests pass)
- **Phase 2:** Gateway routes to both local models (5+ tests pass)
- **Phase 3:** Hybrid routing works, 90%+ success rate (6+ tests pass)

### Performance Benchmarks

| Metric | Llama 3.2 3B | Qwen3 Coder 30B |
|--------|--------------|-----------------|
| Latency (non-streaming) | ~1s | ~3-5s |
| Streaming chunks | 8-70 | 100-150 |
| Context length | 131K | 32K |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Phase 1                              │
│  Test ──────────────────> Local Model (8001/8003)           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         Phase 2                              │
│  Test ──> Gateway ──────> Local Model (8001/8003)           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         Phase 3                              │
│                    ┌──> Local (8003) [40%]                   │
│  Test ──> Gateway ─┤                                         │
│                    └──> Chutes API [60%]                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Adding New Tests

### Add a New Local Model (Phase 1)

1. Update `test/fixtures/e2e_config.yaml`:
```yaml
local_servers:
  your_model:
    name: "your-model-local"
    url: "http://localhost:8004"
    enabled: true
    ...
```

2. Tests will automatically pick it up (parametrized)

### Add a New Gateway Model (Phase 2)

1. Update `test/fixtures/e2e_config.yaml`:
```yaml
models:
  - id: your-model-test
    phase: [2]
    route:
      - kind: openai_compat
        base_url: http://localhost:8004/v1
        ...
```

2. Add to `LOCAL_MODEL_IDS` in `test_phase2_gateway_routing.py`

---

## Related Documentation

- [Phase 1 Test Details](./test_phase1_direct_local.py)
- [Phase 2 Test Details](./test_phase2_gateway_routing.py)
- [Phase 3 Test Details](./test_phase3_hybrid_routing.py)
- [Main Project README](../../README.md)

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review test output for specific error messages
3. Verify all prerequisites are met
4. Check gateway logs: `/tmp/hybridInference_gateway_10081.log` (automated mode)
