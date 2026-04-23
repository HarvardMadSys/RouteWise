# Architecture Overview

HybridInference is designed as a modular, high-performance inference gateway.

## System Architecture

```
┌─────────────┐
│   Clients   │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  FastAPI Gateway│
│   (serving/)    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌─────────┐
│Routing │ │Adapters │
│Manager │ │         │
└────┬───┘ └────┬────┘
     │          │
     ▼          ▼
┌──────────────────┐
│   LLM Providers  │
│ ┌──────────────┐ │
│ │ Local vLLM   │ │
│ │ OpenAI API   │ │
│ │ Gemini API   │ │
│ └──────────────┘ │
└──────────────────┘
```

## Core Components

### Serving Layer (`serving/`)

The serving layer provides the FastAPI-based gateway:

- **Gateway**: HTTP API endpoints for inference requests
- **Adapters**: Provider-specific API adapters
- **Observability**: Logging, metrics, and tracing
- **Storage**: PostgreSQL integration for request/response logging

### Routing Layer (`routing/`)

Intelligent routing and load balancing:

- **Manager**: Routes requests to optimal providers
- **Strategies**: Different routing algorithms (round-robin, cost-based, latency-based)
- **Health Checks**: Monitor provider availability and performance

### Configuration (`config/`)

Centralized configuration management:

- Model configurations
- Provider settings
- Routing policies
- Feature flags

### Infrastructure (`infrastructure/`)

Deployment and observability:

- Systemd service definitions
- Prometheus metrics collection
- Grafana dashboards
- Alert manager rules

## Key Design Principles

1. **Modularity**: Clear separation between serving, routing, and provider layers
2. **Extensibility**: Easy to add new providers and routing strategies
3. **Observability**: Comprehensive logging and metrics at every layer
4. **Performance**: Optimized for low-latency, high-throughput inference
5. **Reliability**: Health checks, retries, and fallback mechanisms

## Data Flow

1. Client sends inference request to Gateway
2. Gateway validates and preprocesses request
3. Routing Manager selects optimal provider
4. Adapter translates request to provider-specific format
5. Provider processes inference
6. Response is logged to PostgreSQL
7. Metrics are exported to Prometheus
8. Response is returned to client
