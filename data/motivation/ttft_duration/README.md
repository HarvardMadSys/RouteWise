# TTFT Duration Share Data

Sanitized admin User Requests exports used by `plots/motivation/plot_ttft_duration_share.py`.

The raw exports were downloaded on 2026-05-15 for the 2026-03-01 to 2026-05-15 window. These repo-local files keep only the fields needed to reproduce the figure:

- `timestamp`
- `model_id`
- `provider`
- `ttft_ms`
- `latency_ms`
- `prompt_tokens`
- `completion_tokens`
- `reasoning_tokens`
- `cache_read_tokens`
- `cache_write_tokens`
- `total_tokens`
- `status_code`

The plot script applies the paper filters:

- successful requests only
- positive `ttft_ms`, `latency_ms`, `prompt_tokens`, and `completion_tokens`
- `latency_ms >= ttft_ms`
- agentic workload: `completion_tokens <= 0.01 * prompt_tokens`
- provider-specific traces:
  - GPT-5.4 with `provider=openai`
  - Claude Opus 4.7 with `provider=anthropic`, excluding two `ttft_ms > 20000` outliers
  - MiniMax-M2.5 with `provider=minimax`

Reproduce the paper figure with:

```bash
uv run python plots/motivation/plot_ttft_duration_share.py
```
