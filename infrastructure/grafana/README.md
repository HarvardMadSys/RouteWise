# Grafana Dashboards

This folder contains ready-to-import Grafana dashboards for the hybrid inference service.

## Import Steps
- Open Grafana → Dashboards → Import
- Upload `dashboards/hybrid_inference_overview.json`
- Select your Prometheus data source when prompted
- Save the dashboard

## Notes
- The dashboard expects Prometheus metrics exposed at `/metrics` as configured under `infrastructure/prometheus/`.
- Variables `route`, `method`, `stream`, `provider`, and `model` help slice views by key labels.
- Feel free to clone and adapt the dashboard for environment-specific needs.
