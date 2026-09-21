# Artifact clarifications and errata

These notes accompany the EuroSys '27 AE artifact for *RouteWise:
Latency–Cost Optimization for Multi-Provider LLM Routing*. They address the
evaluated 14-page paper, SHA-256
`3e8e99eb2db98bac8b9c75cc1b4adc75b7ea13835027f304e6eccaf076e49ba8`,
and the evaluated artifact commit
`ae3a9988b5df6fd3396a41fd4ab53d1b7af1fadd`. They are an artifact erratum;
the evaluated paper is unchanged.

This follow-up preserves routing algorithms, simulator timing, input data,
the historical source archive, and numeric references. Its execution fix
initializes the Figure 8 cache inside the extracted snapshot before either
worker path. CI now includes a one-worker reference check alongside the
existing parallel check. The [figure evidence table](FIGURE_MAP.md) separates
recorded-data aggregation, simulation, redraws, and illustrative diagrams.

## Figure 8 algorithm variant

The default Figure 8 command executes historical source `99c5f3f`. Its
`concurrency_shadow_price` returns `L` for available concurrency. The
zero-cost rule in §3.2.3 and Equation (3) was implemented in subsequent
commit `86a26e2`; it changes some routing decisions. Thus Figure 8 evaluates
an earlier variant, and reproducing its archived values does not validate
those same values for the algorithm described in the text. This mismatch
also occurs in the July arXiv submission; it is not solely a difference
between that submission and the evaluated PDF.

The default command retains the earlier experiment. See
[PAPER_FIGURE8.md](../../experiments/simulation/PAPER_FIGURE8.md) for the
revision mapping, exact settings, source checksum, and numeric tolerances.
The raw PROD population is 24,035 records (the approximately 24K population
described beside Table 2); removing 2,238 failed and 119 remaining zero-token
records leaves 21,678 requests per simulated policy.

## Simulator output-length feedback

Both the evaluated simulator and the historical Figure 8 source compute a
request outcome and immediately call `policy.observe`, updating the
output-length estimator before routing the next arrival. They do not wait
for that response's simulated completion. With overlapping arrivals, later
routing decisions can therefore use output lengths that would not yet be
observable online. This is a limitation of the simulation, not a behavior
introduced by the reproduction wrapper.

The evaluator's limited PROD diagnostic delayed API/quota feedback only
until first-token arrival: it changed 9, 4, and 0 primary decisions at
α = 0, 0.25, and 0.5, respectively, with cost changes below 0.05% and
unchanged SLO rates. These are evaluator-reported sensitivity results,
not a full completion-time correction or a bound for the full BurstGPT
replay. This follow-up does not change feedback timing or claim that its
impact has been fully established.

## Figure 9a experiment scope

`scripts/run_output_length_prediction_ablation.py` defaults to an oracle
base predictor, with one fixed multiplier for each run:

```text
predicted output length = actual output length × (1 + error_pct / 100)
error_pct ∈ {-50, -25, 0, 25, 50, 100, 200}
```

The default sweep has three α values (0, 0.25, 0.5), each with LP-only
and LP+hedging policies: 42 configurations per seed. Its zero-bias baseline
is the oracle, not the online bucket-mean estimator. The released pipeline
measures sensitivity to systematic output-length bias, not independent
uniform random errors in [−50%, +200%] as described in §4.4.2. Accordingly,
it does not establish the claimed random-noise robustness of bucket-mean
prediction. No new random-noise experiment is included here.

The fixed-bias/oracle default is present in the historical wrapper added
at `b6ec95be0d6d709afc40f5f4508eee46c9858384` (June 22, 2026). This source
history identifies the released experiment's semantics; it does not
recover a complete original Figure 9a run manifest. The 10,000-request
prefix runs reported during evaluation establish functionality, not full
numeric reproduction of Figure 9.

## Theorem 3.3 erratum

The competitive-ratio guarantee as stated does not hold for the threshold
in Equation (1). We accept the counterexample: let Q = 100, L = 1, U = e,
and let 100 requests each have saved value 1.001 and consume one quota unit.
The initial threshold is 1, so the first request is admitted. The threshold
then becomes exp(1/100) ≈ 1.01005, exceeding 1.001, and the remaining
requests are rejected. The algorithm saves 1.001 while the optimum saves
100.1, giving a ratio of 100, greater than the stated
1 + ln(U/L) = 2.

Theorem 3.3 must therefore not be used as a guarantee for the released
threshold. In this artifact the exponential threshold is an empirically
evaluated heuristic; no replacement competitive-ratio theorem is claimed.
Empirical P10/P90 value estimates are also not guaranteed bounds on every
request, and predicted values need not equal realized saved values. A
saved-value bound would require care before translating it into a bound
on residual API spend. The reported measurements were obtained by
executing the routing heuristic; they neither establish nor rely on the
stated competitive-ratio bound.

## Recorded metrics and headline percentages

For the released real-provider records, mean and percentile TTFT use
successful requests with present, nonnegative TTFT. The SLO-violation rate
uses all 14,233 requests per policy: failures count as violations, as do
successful requests with TTFT above 3,000 ms. The two metrics intentionally
have different denominators. Costs aggregate all recorded requests and
add the prorated subscriptions.

| Metric | RouteWise α = 0 | OR-auto | Relative reduction |
|---|---:|---:|---:|
| Total cost (USD) | 2.1829646133 | 2.9693093100 | 26.48% |
| Mean TTFT (ms) | 923.6842503 | 2207.2699974 | 58.15% |
| SLO-violation rate (%) | 0.4496592 | 22.6445584 | 98.01% |

The reductions use `100 × (OR-auto − RouteWise) / OR-auto` on unrounded
aggregates. They supersede the evaluated paper's 26.6%, 58.4%, and 98.2%
figures; the underlying released records are unchanged. To compute them
directly after `uv run python scripts/reproduce_real_world.py`:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path("outputs/figures/real_world/real_world_summary.json").read_text())
by_policy = {row["policy"]: row for row in rows}
rw, baseline = by_policy["budget_range_alpha0_hedge"], by_policy["or_auto"]
for key in ("total_cost_usd", "ttft_mean_ms", "slo_violation_rate"):
    reduction = 100 * (baseline[key] - rw[key]) / baseline[key]
    print(f"{key}: {reduction:.2f}% relative reduction")
PY
```

For the separate 30-day simulation, LP-only α = 0 and Greedy-cost are
close, not identical. The evaluator's full RW8 run reported $233.473
versus $236.759 and mean TTFT 1545.182 versus 1543.868 ms, respectively.
The README's earlier wording that they "match" was too strong.

## Offline comparisons

The released `experiments/simulation/offline_oracle.py` assigns requests
at fixed arrival times without queueing, reordering, or preemption. Online
concurrency occupancy uses sampled service durations; the offline baseline
uses deterministic P50 TTFT and throughput. Single-window quota and some
fixed-start concurrency settings have exact implementations under their
own assumptions, while multi-window quota uses a value-greedy baseline
and joint exact solving is restricted by request count. These semantics
are not interchangeable with the paper's time-indexed ILP/Gurobi description.

Successful execution of `--policy offline` is therefore insufficient to
verify the reported 15.0%, 19.3%, and 6.0% distances to an optimum. The
original claim-to-configuration mapping, solver-status evidence, and a
comparison with matched realized durations remain unresolved. This
follow-up does not present those percentages as verified or infer a valid
bound from unmatched duration models.

## Cache and monetary accounting

With prefix-cache accounting enabled, both the Figure 8 snapshot and the
current helper apply a request's recorded cache-read tokens to candidate
API providers that offer cached-input rates. This reuses trace-observed
hits; it does not rebuild per-provider cache residency after changing the
routing policy. Account pseudonyms do not provide that missing state. No
cold-cache or locality-aware sensitivity result is supplied in this update.

For live records, `recorder.py` sums primary and backup billed costs.
The reported total is that sum plus fixed subscriptions. Profiling/probe
expense is tracked separately by the runner and is not included in this
total. When a canceled request's usage is not returned and billing
continues, `transports.py` can record zero with an unmeasured-cost marker;
that is an unknown charge, not a known zero charge. Reported usage costs
are retained when available, and some other missing usage is estimated
from token rates. `physical_cost_usd` is a separately recorded measure,
not an extra term to add again to the billed total.

The released CSVs omit the per-leg cost-source annotations and the probe
ledger. They do not establish probe overhead, the extent of unmeasured
cancellation charges, or an upper bound on those charges. The reproduced
totals validate the released accounting records, not a complete invoice
reconciliation. [Real-record data notes](../../data/real_eval_records/README.md)
describe the exported fields and the 24-hour billing window.

## Figure 2 sampling

The plot filters for `input_len = 10`, present timestamps and latencies,
and positive latencies, then sorts by timestamp. The released valid samples
predominantly occur in close pairs separated by about an hour, with
occasional gaps exceeding eleven hours. They do not support the caption's
regular five-minute sampling description.

| Series | Valid samples | Elapsed span (hours) | Median span of a full 100-sample window (hours) | Global P99 (ms) | Maximum rolling P99 (ms) |
|---|---:|---:|---:|---:|---:|
| Llama-3.3-70B | 2,419 | 1,183.96 | 52.29 | 1,189.32 | 5,305.81 |
| GPT-4o-mini | 2,442 | 1,182.46 | 50.72 | 2,805.10 | 4,731.81 |

The rolling P50/P99 window is 100 observations, not a fixed number of
hours. The median spans above use the timestamp of each window's last
sample minus its first. The released files do not establish whether
earlier collection, export, or subsampling produced the gaps, so no
specific cause is asserted. Global P99 remains computable from the
released measurements. Figure 7a separately remains a redraw from an
author-reconstructed snapshot, not recovered original profiling data.

## Other interpretation limits

The hedging success-probability expression assumes appropriate independence
of primary and backup latencies. Shared infrastructure or overload may
violate that assumption; the target is not a guaranteed SLO. The LP
constrains expected primary effective cost, not the final bill including
hedges. Proposition 3.4 should be read as existence of a sparse optimal
basic solution; ties can also admit dense optima. Uniform α spacing does
not imply uniform spacing of achieved latency and cost.

The real-provider baselines and RouteWise configurations ran in separate
24-hour windows. Provider drift can confound the comparison, and gains
relative to OpenRouter combine subscription access, routing, and hedging.
This update supplies no repeated/interleaved trial or correlation study.
Zero-cost greedy concurrency is an empirical policy choice, not a general
optimality result: an occupied slot can exclude a later more valuable
request. Quality comparisons remain conditional on matched model weights,
quantization, and serving behavior.

## Environments and validation

The artifact pins Python to `3.14` in `.python-version` and dependencies
in `uv.lock`; the Python patch release and OS can differ across installations.

| Check | Environment | Evidence and scope |
|---|---|---|
| Evaluated `ae3a998` CI | Ubuntu 24.04, Python 3.14, uv 0.9.7 | [Run 34542875011](https://github.com/HarvardMadSys/RouteWise/actions/runs/34542875011) passed smoke and committed-data figures, including Figure 8; a separate job built Docker and ran its smoke test |
| Reviewer Linux execution | Ubuntu 22.04.5, x86-64, Python 3.14.7, `uv sync --frozen` | Reviewer reported 680 tests, recorded-data figures, eight Figure 8 policies, and the full 13-policy, 1,813,565-request RW8 replay passing; Figure 9 used 10,000-request prefixes |
| This follow-up, local verification (2026-09-21) | macOS 15.7.7 (24G720), arm64, Python 3.14.0, uv 0.11.3, `uv sync --frozen` | Figure 8 one-worker and default four-worker checks against all eight archived rows; ten real-record reference checks; Figures 2/3 generated; smoke test passed (102 tests passed, 11 skipped for missing optional workloads, 2 deselected) |

The historical experiment's exact original OS and full environment manifest
have not been recovered in this follow-up. The tested environments above
must not be mistaken for that original environment. The macOS NumPy
code-signature failure reported by another evaluator is environment-specific;
the README's Ubuntu 24.04 Docker route is available without changing the
algorithm or dependencies. The Docker check above covers the smoke test,
not the full 30-day workload or full figure pipeline inside a container.

Validation commands for this patch, from the repository root:

```bash
uv sync --frozen
uv run python scripts/reproduce_figure8.py --jobs 1 --output-dir outputs/figure8-single-worker
uv run python scripts/reproduce_figure8.py --jobs 4
uv run python scripts/reproduce_real_world.py
uv run python plots/motivation/drift_wall_clock.py --source-dir data/drift_source --output-dir outputs/figures
uv run python -m plots.motivation.plot_ttft_background_panels --output-dir outputs/figures
bash scripts/artifact_smoke_test.sh
```

The Figure 8 checks require exact request/provider counts and relative/absolute
numeric tolerance `1e-9` for all eight policies. Each invocation extracts a
fresh source directory, so a previous checkout cache cannot mask the
single-worker failure. The same one-worker check is included in the CI
workflow. The evaluated CI link above certifies `ae3a998`; it predates this
follow-up and does not certify its changes.
