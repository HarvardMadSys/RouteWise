# Real-Eval Monitor

Live monitor for `scripts/run_real_eval_8h_policy_processes.sh` runs. Reads each policy's `requests.csv` + `run.log` and prints health metrics. Read-only and safe to run repeatedly against an in-flight experiment.

## Usage

```bash
python3 scripts/monitor_real_eval.py <OUTPUT_BASE>
```

`<OUTPUT_BASE>` is the directory you passed as `OUTPUT_BASE=` to the launcher (the one that contains `policies.txt`, `run_env.txt`, and one subdir per policy).

### Flags

| flag | what it does |
|---|---|
| `--watch N` | Refresh every `N` seconds (Ctrl+C to stop). |
| `--policies-only` | Print only the per-policy summary table. |
| `--providers-only` | Print only the per-provider TTFT distribution table. |
| `--snapshot` | Freeze current state into `<OUTPUT_BASE>/snapshots/snapshot_<ts>/` (text + JSON + copies of each policy's `requests.csv` / `run.log` / `args.json`). |
| `--snapshot-label LABEL` | Append `_LABEL` to the snapshot directory name. |

`--policies-only` and `--providers-only` are mutually exclusive.

## Output

### Header
```
trace dataset      : 14233 requests over 8.1 hours
wall-clock elapsed : 2h49m17s  (cap 10h00m00s)
trace replayed     : 34.7% by trace time, 72.1% by per-policy request count
SLO threshold      : 3000 ms (ttft)
total cost (all)   : $7.4316
```

- `wall-clock elapsed` = `now − earliest ts across all policies' requests.csv`.
- `trace replayed by trace time` = `wall_elapsed / trace_time_sec` (intuitive at speedup=1).
- `trace replayed by per-policy request count` = `Σ rows / (trace_total × n_policies)`.

### Per-policy table
| column | meaning |
|---|---|
| `reqs` / `fail` | total CSV rows / rows with `status != success` |
| `cost$` | sum of `billed_cost_usd` |
| `mean / p50 / p90 / p95 / p99` | `ttft_ms` distribution over successful requests (linear-interpolated percentiles) |
| `SLO viol` | fraction with `ttft_ms > SLO_MS` OR failed |
| `tier mix` | share of rows by `tier` (`api` / `quota` / `concurrency`); top 3 |
| `top providers` | share by normalized `actual_provider`; top 3 with `+N` for the rest |
| `hedge` / `win` | rows where `hedge_triggered=1` / subset where `hedge_winner=backup` |
| `429s` | rows with `rate_limited=1` |

For the OpenRouter sentinel baselines (`or_auto`, `or_sort_latency`, `or_sort_cost`), `actual_provider` is `__or_auto__@DeepInfra` etc. — the monitor strips the sentinel and shows the OR-selected sub-provider (`DeepInfra`, `Friendli`, …). Tier is inferred as `api` for those rows since the runner restricts the sentinel allowlist to API-tier providers.

### Per-provider TTFT table
Aggregates every CSV across all policies, grouped by normalized provider name. Useful for spotting slow or flaky providers regardless of which policy routed to them.

`OR_X` (pinned via `provider_hint`) and `X` (auto-selected by an OR sentinel) appear as separate rows even though they hit the same backend — comparing them sanity-checks the auto-routing.

## Snapshots

`--snapshot` writes to `<OUTPUT_BASE>/snapshots/snapshot_<YYYYMMDD_HHMMSS>[_<label>]/`:

```
summary.txt                # rendered text table
summary.json               # structured metrics (policies + providers_ttft)
<policy>/requests.csv      # frozen copy per policy
<policy>/run.log
<policy>/args.json
run_env.txt
policies.txt
initial_profile.json
policy_key_assignments.tsv
```

Take as many snapshots as you want during a run; each gets its own directory.

## Examples

```bash
# One-shot full snapshot view
python3 scripts/monitor_real_eval.py outputs/real_eval/run_xyz

# Live policies-only dashboard, refresh every 30s
python3 scripts/monitor_real_eval.py outputs/real_eval/run_xyz --policies-only --watch 30

# Freeze current state with a label
python3 scripts/monitor_real_eval.py outputs/real_eval/run_xyz --snapshot --snapshot-label midrun_3h
```

## Notes

- Latency / SLO use **TTFT** (`ttft_ms`), not e2e. TTFT is what routing controls; e2e is dominated by output-token count.
- The script never writes into the policy directories; only the `snapshots/` subdir.
- Missing files are tolerated — a policy that hasn't started dispatching yet shows up as a zero-filled row.
