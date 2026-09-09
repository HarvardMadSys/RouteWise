# Data

| Path | Contents | Used by |
|---|---|---|
| `real_eval_records/` | 24-hour real-provider replay records (10 policies × 14,233 requests) | Figures 1, 6, 7 — see its README |
| `freeinference.jsonl` | De-identified PROD agentic-workload trace (24,035 requests, 7 days) | Figure 8 |
| `drift_source/` | Provider TTFT measurements over 50 days (Llama-3.3-70B, gpt-4o-mini) | Figure 2 |
| `motivation/ttft_duration/` | Sanitized production request exports (GPT-5.4, Claude Opus 4.7, MiniMax-M2.5) | Figure 3 |
| `fixtures/` | 120-request synthetic trace for the smoke test | `scripts/artifact_smoke_test.sh` |

The BurstGPT and ShareGPT sources behind the 30-day simulation workload are
downloaded from their public hosts by `scripts/prepare_workload.py` (SHA-256
pinned) and are not redistributed here.

## PROD trace (`freeinference.jsonl`)

A sampled 7-day trace from our production router, serving mostly agentic
applications (paper Table 1: ~24K requests, ~48K-token mean prompt, 65%
prefix-cache hits). One JSON object per line:

| Field | Meaning |
|---|---|
| `timestamp` | Request arrival, ISO-8601 UTC |
| `request_id` | Opaque request identifier |
| `model_id`, `provider` | Model served and upstream provider |
| `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `total_tokens` | Token counts reported by the provider |
| `cache_read_tokens`, `cache_write_tokens` | Prefix-cache token counters reported by the provider (`null` when not reported) |
| `cost_usd`, `latency_ms`, `ttft_ms` | Observed charge and latencies |
| `status_code` | HTTP status; the simulator loader skips rows ≥ 400 |
| `user_id` | Stable per-account pseudonym (`account-NN`), used only as the prefix-cache locality key |

De-identification: account names, e-mail addresses, and error strings were
removed; account identifiers were replaced with per-account pseudonyms that
preserve grouping. The trace never contained prompt or response text. The
simulator consumes only the numeric fields above, so the replay is identical
to one on the internal file (21,678 requests enter the simulation after the
loader drops failed and zero-token rows). `SHA256SUMS` covers the file.

Produced from the internal export with `scripts/export_ae_data.py prod-trace`.
