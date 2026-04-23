# Deployment Guide

Guide for deploying HybridInference in production.

## Production Deployment

### Using systemd

The recommended way to deploy HybridInference is using systemd.

1. **Install dependencies:**

   ```bash
   cd hybridInference
   uv venv -p 3.10
   source .venv/bin/activate
   uv sync
   ```

2. **Create systemd unit file:**

   ```bash
   sudo cp infrastructure/systemd/hybrid_inference.service /etc/systemd/system/
   ```

3. **Configure environment:**

   Edit `/etc/systemd/system/hybrid_inference.service` and update:
   - `WorkingDirectory`
   - `User`
   - Environment variables

4. **Start the service:**

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable hybrid_inference.service
   sudo systemctl start hybrid_inference.service
   ```

5. **Check status:**

   ```bash
   sudo systemctl status hybrid_inference.service
   journalctl -u hybrid_inference.service -f
   ```

### Environment Variables

Required environment variables for production:

```bash
# API Keys
DEEPSEEK_API_KEY=your-key
GEMINI_API_KEY=your-key
LLAMA_API_KEY=your-key

# Database
DB_NAME=hybridinference
DB_USER=postgres
DB_PASSWORD=your-secure-password
DB_HOST=localhost
DB_PORT=5432

# Local vLLM (optional)
LOCAL_BASE_URL=http://localhost:8000/v1

# Rate limiting (optional)
RATE_LIMIT_PER_MINUTE=100
```

### Health Checks

Monitor service health:

```bash
curl http://localhost:80/health
```

### Logs

View logs:

```bash
# Follow logs
journalctl -u hybrid_inference.service -f

# View recent logs
journalctl -u hybrid_inference.service -n 100
```

## Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics`:

```bash
curl http://localhost:80/metrics
```

Key metrics:
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency
- `model_requests_total` - Requests per model
- `model_errors_total` - Errors per model

### Grafana Dashboards

Import the dashboard from `infrastructure/grafana/`.

## Database Setup

### PostgreSQL

1. **Create database:**

   ```sql
   CREATE DATABASE hybridinference;
   CREATE USER hybridinference WITH PASSWORD 'your-password';
   GRANT ALL PRIVILEGES ON DATABASE hybridinference TO hybridinference;
   ```

2. **Configure connection:**

   Update `.env` with database credentials.

See the Database guide in this section: [Database](database.md).

## Troubleshooting

### Service won't start

Check logs:
```bash
journalctl -u hybrid_inference.service -n 50
```

Common issues:
- Missing API keys
- Database connection failed
- Port already in use

### High latency

Check:
- Database performance
- Provider API latency
- Resource usage (CPU/memory)

### Rate limiting

Adjust rate limits in configuration:
```yaml
rate_limits:
  requests_per_minute: 100
  tokens_per_minute: 100000
```
