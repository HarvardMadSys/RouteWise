# OpenRouter Latency Experiments: Plan and Budget (Brief)

## Objective
Use OpenRouter as the primary (real) evaluation environment to validate our full latency-aware routing design for a single model, and produce paper-ready cost–latency results that support:

- **Heavy-tail and drift**: latency is heavy-tailed and non-stationary on OpenRouter, motivating **online profiling**.
- **SLO-first routing**: enforcing tail-latency SLOs with minimal cost overhead.
- **Smart hedging**: reducing tail latency (and SLO violations) by duplicating requests when a primary provider is likely to miss the SLO.

We do not require a separate motivation figure from an external benchmark; instead, the OpenRouter probing and online runs provide both (i) the empirical motivation (heavy tails, drift) and (ii) the end-to-end evaluation (policy comparisons).

## Scope (Initial)
- **Models**:
  - Primary: `meta-llama/llama-3.3-70b-instruct` (open-weight; many OpenRouter providers, including potential GPU/backbone heterogeneity).
  - Secondary (recommended): an OpenAI family model on OpenRouter (e.g., `openai/gpt-4o-mini` or `openai/gpt-4.1-mini`) to validate that our conclusions and policies are not specific to one model/provider ecosystem.
- **Request shape**: short prompt + small output (`max_tokens=8` or `16`) to primarily measure network/queueing/prefill effects.
- **Latency metric**: short-output end-to-end latency as a proxy for TTFT/prefill latency (explicitly documented as a proxy).
- **Providers (Llama)**: the set of providers available on OpenRouter for the target model (example list observed in our environment):
  - DeepInfra, Nebius, NovitaAI, Parasail, Crusoe
  - Cloudflare, Hyperbolic, Groq, Google Vertex, Cerebras
  - (If available) Chutes

For the OpenAI model, we will use the OpenRouter provider set exposed for that model (typically fewer providers). The exact provider list is discovered programmatically via probing.

## Key Hypotheses
1. **Non-stationarity**: rolling tail metrics (e.g., rolling P99) drift significantly over time; a global P99 is not reliable.
2. **SLO-first improves reliability**: SLO-first filtering reduces SLO violations relative to cheapest-only routing with moderate cost increase.
3. **Hedging is a robust safety net**: adaptive hedging reduces tail latency under drift without requiring manual per-provider tuning.

## Experiment Phases

### Phase 0: Configuration (No API cost)
- Choose:
  - Tail-latency SLO(s), e.g., `SLO ∈ {500, 1000, 2000} ms`.
  - `max_tokens` (8/16), prompt template, and timeout.
  - Rolling window size for profiling (e.g., 100–500 samples).

### Phase 1: Provider Discovery (Small API cost)
- Goal: identify which providers are actually selected by `route=auto`, and the observed distribution of actual providers.
- Procedure:
  - Send `M` requests with `route=auto` and record the **actual provider** in the response.
  - Output: `{provider: frequency}` plus basic latency stats per provider.

### Phase 2: Latency Probing (Main data collection; controllable cost)
- Goal: estimate per-provider latency distributions and their drift over time.
- Procedure:
  - For each provider `p` and for `auto`, send `N` probe requests.
  - Record: timestamp, requested route, actual provider, `e2e_ms`, (optional `ttft_ms` if streaming), `usage` tokens, status/error.
  - Run two tiers:
    - **Pilot**: `N=50` per provider (fast iteration).
    - **Paper-grade**: `N=500–1000` per provider (stabilize P99 estimates).

### Phase 3: Offline Policy Evaluation (No additional API cost)
- Build per-provider profiles from probing logs.
- Evaluate policies and generate figures:
  - Main figure: **Cost vs P99 latency** (Pareto scatter/curve).
  - Supporting: provider profile plots (CDF/CCDF, percentiles) and drift plots (rolling P99).

### Phase 4: Online Replay Validation (Required; additional API cost)
- Goal: demonstrate end-to-end gains in a true online setting (including drift and tail events), using either (i) an arrival trace replay or (ii) time-structured request streams (bursts + quiet periods).
- Run a limited set of policies to control spend:
  - OpenRouter default (`auto`)
  - Cheapest fixed
  - SLO-first (chosen SLO)
  - Hedging variant (optional)

## Routing Policies (Minimum Set)
- **Baselines**
  - OpenRouter default (`route=auto`)
  - Cheapest fixed provider
  - Fastest fixed provider (lowest estimated P99)
  - Always duplicate (send to two providers; upper-bound baseline)
- **Our methods**
  - SLO-first filtering (filter providers by `P99 <= SLO`, then pick cheapest)
  - Pareto/skyline selection (cost–P99 frontier + scalarization weight)
  - Smart hedging (duplicate only when the primary is likely to miss SLO)

## Metrics
- Latency: P50/P90/P99, SLO violation rate (fraction exceeding SLO).
- Cost: mean cost per request (USD/request), total cost, incremental cost vs baseline.
- Hedging: hedge rate, effective latency after hedging, extra cost due to duplication.
- Reliability: timeout/error rates by provider and by policy.

## Budget Estimation (Template)
Let:
- `P` = number of providers (excluding `auto`), total routes probed = `P + 1`.
- `M` = number of discovery requests (Phase 1).
- `N` = probes per route (Phase 2).
- `R` = number of replay requests (Phase 4, optional).
- `T_in`, `T_out` = average prompt/completion tokens per request.
- `p_in`, `p_out` = model price (USD per 1M tokens) for prompt/completion.

Per-request cost:
```
c_req = (T_in * p_in + T_out * p_out) / 1e6
```

Phase 1 (discovery) cost:
```
C_discover ≈ M * c_req
```

Phase 2 (probing) cost:
```
C_probe ≈ (P + 1) * N * c_req
```

Phase 4 (online replay) cost (optional):
```
C_replay ≈ R * c_req * (1 + hedge_rate)
```
Notes:
- For “always duplicate”, `hedge_rate ≈ 1` (cost ≈ 2×).
- For smart hedging, estimate `hedge_rate` from the probing-based policy simulation.

## Risk Controls
- Set a hard cap on total probes and total replay requests.
- Abort rules:
  - provider error rate > threshold (e.g., 5%) for sustained intervals
  - repeated 429 / rate-limiting without recovery
- Log everything needed for reproducibility (seed, timestamps, model id, parameters).

## Deliverables
- `cost_vs_p99_pareto.png` (main paper figure; policy tradeoff)
- `provider_cdf_llama.png` and `provider_cdf_openai.png` (two CDF/CCDF figures showing latency variation across providers and models)
- `rolling_p99_drift_time.png` (rolling tail latency over wall-clock time; drift evidence)
- `policy_results.csv` (all metrics for tables/appendix)
