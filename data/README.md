# Data

| Path | Contents | Used by |
|---|---|---|
| `real_eval_records/` | 24-hour real-provider replay records (10 policies × 14,233 requests) | Figures 1 and 6; Figure 7 provenance is documented separately |
| `real_eval_records_m3/` | 24-hour real-provider replay records on MiniMax-M3 (11 policies × 14,233 requests, 2026-09-15) | Paper revision; not a paper figure |
| `freeinference.jsonl` | De-identified PROD agentic-workload trace (24,035 requests, 7 days) | Figure 8 |
| `figure8_reference_summary.csv` | Archived aggregates for the eight Figure 8 policies | Checked after an independent simulation |
| `drift_source/` | Provider TTFT measurements over 50 days (Llama-3.3-70B, gpt-4o-mini) | Figure 2 |
| `motivation/ttft_duration/` | Sanitized production request exports (GPT-5.4, Claude Opus 4.7, MiniMax-M2.5) | Figure 3 |
| `fixtures/` | 120-request synthetic trace for the smoke test | `scripts/artifact_smoke_test.sh` |

The BurstGPT and ShareGPT sources behind the 30-day simulation workload are
downloaded from their public hosts by `scripts/prepare_workload.py` (SHA-256
pinned) and are not redistributed here.

## PROD trace (`freeinference.jsonl`)

A sampled trace spanning 7.0444 days from our production router, serving
mostly agentic applications. It contains 24,035 raw records. The loader
excludes 2,238 failed requests (HTTP status ≥ 400) and 119 remaining
zero-token requests, leaving **21,678 simulation requests**. Their mean
prompt length is 48,219.9 tokens and 65.947% have positive cache-read token
counts. These filtered statistics use a different population from the
paper's rounded, approximately 24K-request workload description.
One JSON object per line:

| Field | Meaning |
|---|---|
| `timestamp` | Request arrival, ISO-8601 UTC |
| `request_id` | Opaque request identifier |
| `model_id`, `provider` | Model served and upstream provider |
| `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `total_tokens` | Token counts reported by the provider |
| `cache_read_tokens`, `cache_write_tokens` | Prefix-cache token counters reported by the provider (`null` when not reported) |
| `cost_usd`, `latency_ms`, `ttft_ms` | Observed charge and latencies |
| `status_code` | HTTP status; the simulator loader skips rows ≥ 400 |
| `user_id` | Stable per-account pseudonym (`account-NN`), preserving account grouping |

De-identification: account names, e-mail addresses, and error strings were
removed; account identifiers were replaced with per-account pseudonyms that
preserve grouping. The trace never contained prompt or response text. The
loader receives the same timestamps, lengths, cache counters, and other
simulation inputs as the internal export. Cache discounts use the retained
`cache_read_tokens` counters; absent counters are treated as cache misses.
Account pseudonyms do not reconstruct missing cache hits. `SHA256SUMS`
covers the trace and the Figure 8 reference summary.

Produced from the internal export with `scripts/export_ae_data.py prod-trace`.

`figure8_reference_summary.csv` projects the policy, request-count, cost,
TTFT, SLO, hedge-rate, and provider-count columns from the archived
2026-05-14 simulation summary. Only the policy-name spelling changes
(`ablation_lp_only_pN` to `ablation_lp_only_alphaN`); numeric values are
unchanged. The reference is read only after a fresh simulation completes.
See [the Figure 8 source notes](../experiments/simulation/PAPER_FIGURE8.md).
