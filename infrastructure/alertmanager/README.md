# Alertmanager

Alertmanager receives alerts from Prometheus and routes them to your notification channels (Slack, Email, etc.). This folder contains an example configuration you can copy and adapt.

## Files
- `alertmanager.yml.example` — Template config demonstrating common receivers and routing.

## Quick Start (local)

1) Create your config from the template:

```bash
cp infrastructure/alertmanager/alertmanager.yml.example \
   infrastructure/alertmanager/alertmanager.yml
```

2) Configure a receiver (Slack or Email). Example snippets:

Slack:
```yaml
route:
  receiver: slack-default
receivers:
  - name: slack-default
    slack_configs:
      - send_resolved: true
        api_url: "https://hooks.slack.com/services/XXX/YYY/ZZZ"  # replace
        channel: "#alerts"
```

Email (SMTP):
```yaml
route:
  receiver: email-default
receivers:
  - name: email-default
    email_configs:
      - to: "alerts@example.com"
        from: "noreply@example.com"
        smarthost: "smtp.example.com:587"
        auth_username: "smtp-user"
        auth_password: "${SMTP_PASSWORD}"  # mount via env/file; do not commit secrets
```

3) Run Alertmanager with Docker:

```bash
docker run --rm -p 9093:9093 \
  -v $(pwd)/infrastructure/alertmanager:/etc/alertmanager \
  prom/alertmanager:latest \
  --config.file=/etc/alertmanager/alertmanager.yml
```

4) Point Prometheus at Alertmanager (edit `infrastructure/prometheus/prometheus.yml` and uncomment):

```yaml
alerting:
  alertmanagers:
    - static_configs:
        - targets: ["localhost:9093"]
```

## Verify
- Open `http://localhost:9093` for the Alertmanager UI.
- Health endpoints: `/-/ready` and `/-/healthy`.
- Trigger a test alert by temporarily lowering a threshold in a rule under `infrastructure/prometheus/rules/`.

## Tips
- Grouping and routing: group by labels such as `alertname`, `env`, `service` to reduce noise.
- Templating: use Alertmanager templates to include labels like `route`, `provider`, `model` in messages.
- Secrets: never commit secrets; prefer env vars or mounted files for webhook URLs and SMTP credentials.
- Reverse proxy: set `--web.external-url` (or `external_url` in config) if exposed behind a proxy.
- Ownership: add `external_labels` in Prometheus (e.g., `env`, `instance`) to help route/assign alerts.

## Related
- Prometheus rules live in `infrastructure/prometheus/rules/`.
- Prometheus setup and tips: `infrastructure/prometheus/README.md`.
