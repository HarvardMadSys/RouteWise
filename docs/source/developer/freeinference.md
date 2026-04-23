# FreeInference Deployment

## FastAPI + systemd (current)

We serve OpenRouter-compatible traffic directly through a FastAPI application listening on port 80. Removing Nginx reduces operational overhead, keeps debugging straightforward, and lets `systemd` own the lifecycle of the gateway process.

### Overview

```bash
┌─────────────┐      ┌─────────────────┐      ┌────────────────────┐
│  OpenRouter │─────▶│ FastAPI Gateway │─────▶│ Model Executors... │
└─────────────┘      └─────────────────┘      └────────────────────┘
```

- FastAPI binds to `0.0.0.0:80` and exposes `/v1` endpoints consumed by OpenRouter clients.
- The gateway handles request authentication, routing, and backpressure before invoking the selected model adapter.
- `systemd` supervises the process, ensuring automatic restarts after crashes or host reboots.

### Deployment Steps

1. **Install runtime dependencies**

   Ensure Python environment and model weights are ready. Confirm the FastAPI entry point (`serving.servers.bootstrap:app`) is reachable via `uvicorn` or the configured launcher script.

2. **Create the unit file**

   ```bash
   sudo tee /etc/systemd/system/freeinference.service <<'UNIT'
   [Unit]
   Description=FreeInference FastAPI service
   After=network-online.target
   Wants=network-online.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/hybridInference
   ExecStart=/usr/bin/env uvicorn serving.servers.bootstrap:app --host 0.0.0.0 --port 80
   Restart=always
   RestartSec=5
   Environment=PYTHONUNBUFFERED=1

   [Install]
   WantedBy=multi-user.target
   UNIT
   ```

   Replace `User`, `WorkingDirectory`, and `Environment` entries as needed for the target host.
   The repository carries a maintained version of this unit at `infrastructure/systemd/hybrid_inference.service`; copy or symlink it into `/etc/systemd/system/freeinference.service` during deploys.

3. **Reload and enable the service**

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable freeinference.service
   sudo systemctl start freeinference.service
   sudo systemctl status freeinference.service
   ```

### Runtime Operations

- Restart on demand: `sudo systemctl restart freeinference.service`
- Follow logs: `journalctl -u freeinference.service -f`
- Health check: `curl https://freeinference.org/health`
- List registered models: `curl https://freeinference.org/v1/models | jq`

### Why We Dropped Nginx

- FastAPI already terminates HTTP and exposes the required OpenRouter-compatible endpoints.
- Nginx added another moving part, increasing failover complexity and opaque error handling.
- Debugging latency or request routing is simpler when traffic is handled in a single process.

## Legacy Architectures

### Nginx (v2, abandoned)

We briefly fronted FastAPI (running on port 8080) with vanilla Nginx that exposed `http://freeinference.org` on port 80 and terminated TLS for the public endpoint. Once Cloudflare took over edge SSL duties, the extra hop mostly added deployment and observability complexity without material benefit, so the setup was removed.

### Nginx + Lua via OpenResty (v1, abandoned)

We previously relied on OpenResty (Nginx + Lua) to provide a production routing tier across multiple LLM backends. The stack handled model mapping, load balancing, health checks, and error handling. We keep the installation notes for posterity.

#### Overview

```bash
┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Client    │─────▶│  OpenResty       │─────▶│  Backend 1      │
│  (API Call) │      │  (Router)        │      │  (Qwen@8000)    │
└─────────────┘      │                  │      └─────────────────┘
                     │  - Model Mapping │
                     │  - Load Balancing│      ┌─────────────────┐
                     │  - Health Checks │─────▶│  Backend 2      │
                     │  - Error Handling│      │  (Llama@8001)   │
                     └──────────────────┘      └─────────────────┘
```

#### Installation Notes

```bash
# Add repository
wget -O - https://openresty.org/package/pubkey.gpg | sudo apt-key add -
echo "deb http://openresty.org/package/ubuntu $(lsb_release -sc) main" | \
    sudo tee /etc/apt/sources.list.d/openresty.list

# Install
sudo apt-get update
sudo apt-get install openresty
```

```bash
# Create directory
sudo mkdir -p /usr/local/openresty/nginx/conf/sites-available
sudo mkdir -p /usr/local/openresty/nginx/conf/sites-enabled

# Copy Config file
sudo cp <your config file> /usr/local/openresty/nginx/conf/sites-available/vllm

# Enable the site
sudo ln -s /usr/local/openresty/nginx/conf/sites-available/vllm \
           /usr/local/openresty/nginx/conf/sites-enabled/vllm
```

```bash
http {
    # ... Others ...

    # Lua settings
    lua_package_path "/usr/local/openresty/lualib/?.lua;;";
    lua_shared_dict model_cache 10m;

    # Include Site Configuration
    include /usr/local/openresty/nginx/conf/sites-enabled/*;
}
```

```bash
# test openresty config
sudo openresty -t

# Start
sudo systemctl start openresty

# Enable auto-start
sudo systemctl enable openresty

# reload openresty
sudo openresty -s reload
```

```bash
# check service status
curl https://freeinference.org/health

# list all models
curl https://freeinference.org/v1/models | jq

# Chat with Qwen3-Coder
curl -X POST http://freeinference.org/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/models/Qwen_Qwen3-Coder-480B-A35B-Instruct-FP8", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'

# Chat with llama4-scout
curl -X POST http://freeinference.org/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/models/meta-llama_Llama-4-Scout-17B-16E", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'
```

### Nginx (v0, abandoned)

```bash
sudo vim /etc/nginx/sites-available/vllm
sudo nginx -t
sudo systemctl reload nginx

# to test the endpoint
curl https://freeinference.org/v1/models
```
