# Latency Evaluation Design and Cost Budgeting (OpenRouter-Facing)

This document specifies an evaluation plan and a cost-estimation methodology for adding a
latency-aware dimension to our routing system, with an emphasis on a simple and persuasive
comparison against OpenRouter baselines.

## Goals

We aim to show that latency-aware routing can improve tail latency (e.g., P99) **without**
sacrificing the cost savings achieved by our existing cost-driven routing.

Key claims to support:

1. **Latency variability is large even at fixed token length**, motivating online profiling.
2. **Cost-only routing can violate tail SLOs** under provider queueing variance.
3. **SLO-first + hedging can reduce P99** at a modest incremental cost.
4. **OpenRouter-facing relevance**: improvements hold when the provider pool is OpenRouter’s.

## Experimental Setup (Single-Model, OpenRouter Provider Pool)

### Why Single-Model

We intentionally use a single model to avoid confounding from model-quality differences and to
focus on the provider selection problem: multiple providers serving the *same* model.

### Model Selection Criteria

Choose a model that satisfies:

1. **Multiple providers** (at least 3–5) on OpenRouter to enable meaningful comparison.
2. **Price variance** across providers to demonstrate cost-latency tradeoffs.
3. **Affordable** for large-scale replay (avoid frontier models like GPT-4o or Claude-3.5).

**Recommended candidates** (verify current availability on OpenRouter):

| Model | Typical Providers | Approx. Price | Notes |
|-------|-------------------|---------------|-------|
| `meta-llama/llama-3.1-8b-instruct` | 4–6 | $0.05–0.10/1M | Cheap, many providers |
| `mistralai/mistral-7b-instruct` | 3–5 | $0.05–0.15/1M | Good provider diversity |
| `qwen/qwen-2.5-7b-instruct` | 3–4 | $0.03–0.08/1M | Very cheap option |

To check current providers for a model:
```bash
curl -s https://openrouter.ai/api/v1/models \
  | jq '.data[] | select(.id == "meta-llama/llama-3.1-8b-instruct") | .providers'
```

### Provider Pool

Let `J` be the set of eligible providers for the selected model on OpenRouter.
We compare:

- **OpenRouter default** (baseline): `route=auto` (or the canonical OpenRouter configuration used by
  users).
- **Manual provider selection** (our methods): the router explicitly chooses a provider in `J`
  (and optionally triggers hedging).

### OpenRouter API Usage

**Specifying a provider explicitly:**

```python
import openai

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

response = client.chat.completions.create(
    model="meta-llama/llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "provider": {
            "order": ["Together"],  # Force specific provider
            "allow_fallbacks": False
        }
    }
)
```

**Getting provider from response:**

The response includes provider info in `response.model` (e.g., `meta-llama/llama-3.1-8b-instruct:together`).

**Baseline (route=auto):**

Simply omit the `provider` field to use OpenRouter's default routing.

### Workload Trace

We use an arrival trace (e.g., BurstGPT) and replay it online.
Because BurstGPT does not include prompt text, we generate synthetic prompts that match input token
length distributions. This is acceptable for latency evaluation when we control output length,
because latency is dominated by network + queueing + prefill for short outputs.

Recommended two regimes:

1. **Short-output regime (queueing-dominant)**:
   - Small `max_tokens` (e.g., 16–64).
   - Prompt: short instruction, ask for a concise response.
   - Purpose: isolate network/queueing/prefill variability.
2. **Long-output regime (decode-inclusive)**:
   - Larger `max_tokens` (e.g., 256–1024).
   - Purpose: test whether providers differ in TPS and whether hedging still helps.

## Online Latency Profiling

We maintain an online latency profile per provider using lightweight probing requests.

### Probing Design

- Probe prompt: 1–2-token instruction and a constraint “respond concisely”.
- Measure:
  - `TTFT` (time-to-first-token)
  - `E2E` (time-to-last-token)
  - `TPS` (estimated as `output_tokens / max(E2E - TTFT, eps)`)

### Sliding-Window Estimation

Maintain a sliding window (or EWMA) over the most recent `W` probes per provider.
From this window, compute:

- Empirical CDF / CCDF
- P50/P90/P99
- Bootstrap confidence intervals for P99 (paper-quality reporting)

This enables demonstrating *time-varying queueing effects* and motivates our “distribution-aware”
decision logic.

## Routing Policies to Evaluate

All policies operate on the same provider pool `J`.

### Baselines

1. **OpenRouter default**: `route=auto`.
2. **Cheapest fixed provider**: always use the lowest-cost provider in `J`.
3. **Fastest fixed provider**: always use the lowest-P99 provider (based on probing).
4. **Cost-only ranker**: choose provider by predicted cost only (ignoring latency distribution).
5. **Always-duplicate**: send requests to 2 providers in parallel and take the first response
   (expensive but strong latency baseline).

### Our Policies

1. **SLO-first filtering + cost ranking**
   - Filter providers with `P99_provider <= L` (or `P(T <= L) >= 0.99`) based on current profile.
   - Choose the cheapest among the remaining providers.
2. **Pareto / skyline selection**
   - Compute `(expected_cost, P99_latency)` per provider.
   - Restrict candidates to the Pareto frontier.
   - Choose based on a user knob (e.g., `lambda` in a scalarized objective).
3. **Smart hedging (optional but high impact)**
   - Start with the chosen (typically cheaper) provider.
   - If no completion by time `h`, send a duplicate to the fastest provider (or current best
     fallback).
   - Return the first completion and cancel the other if supported.

### Hedging Cancellation Semantics

**Important clarification on "cancel the other":**

OpenRouter (and most LLM APIs) do **not** support mid-stream request cancellation with billing refund.
Once a request is sent, you are billed for tokens generated up to the point of completion or client
disconnect.

**Practical hedging implementation:**

1. **Client-side cancellation**: When the first response arrives, simply close the HTTP connection
   to the slower provider. This stops further token generation but you are still billed for tokens
   already produced.

2. **Billing implications**:
   - If hedge triggers at time `h` and backup completes at time `t_b`:
     - Primary request billed for: tokens generated in `[0, t_b]` (partial)
     - Backup request billed for: full completion tokens
   - Worst case (both complete): pay for both full responses

3. **For cost estimation**, we define `kappa` (cancellation efficiency):
   - `kappa = 1.0`: Conservative—assume full duplicate is billed
   - `kappa = 0.5`: Moderate—assume ~50% of duplicate tokens billed (early cancellation)
   - `kappa = 0.0`: Optimistic—assume cancellation prevents billing (unrealistic)

   **Recommendation**: Use `kappa = 0.7` for realistic estimates.

## Metrics and Reporting

### Latency Metrics

Report per policy:

- TTFT: P50/P90/P99
- E2E: P50/P90/P99
- SLO violation rate: `P(T > L)`
- Error/timeout rate (provider failures)

### Cost Metrics

- Mean cost/request and total cost
- Cost breakdown by provider
- Hedging overhead:
  - duplicate rate
  - extra token usage due to hedging
  - additional dollar cost

### Statistical Rigor

- Bootstrap confidence intervals for P99 and violation rate.
- Time-of-day stratification to show non-stationarity (if probe logs support it).

## Cost Estimation and Budgeting

We separate cost into:

1. **Probing cost** (continuous, small).
2. **Replay cost** (experiment cost, potentially large).
3. **Hedging overhead** (policy-dependent).

All costs below assume per-token billing on OpenRouter (adjust if provider-specific billing differs).

### Notation

- `N`: number of replayed requests
- `n_in`, `n_out`: input/output tokens per request (bounded by `max_tokens`)
- `p_in,j`, `p_out,j`: provider `j` input/output price per token
- `pi_j`: fraction of traffic routed to provider `j`
- `rho`: probability of triggering a hedge (duplicate request)
- `j_f`: fallback provider used for hedging
- `b`: billing model for hedging
  - `b=conservative`: pay both attempts
  - `b=optimistic`: pay only the winning attempt (or assume cancels are not billed)

### Replay Cost (No Hedging)

Expected cost:

`E[Cost] = N * Σ_j pi_j * (E[n_in] * p_in,j + E[n_out] * p_out,j)`

Practical conservative bound using caps:

`Cost_max <= N * (E[n_in] * p_in,max + max_tokens * p_out,max)`

### Hedging Cost (Additional Overhead)

Conservative additional cost from hedging:

`ΔCost_hedge <= N * rho * (E[n_in] * p_in,j_f + E[n_out] * p_out,j_f)`

If cancellations reduce billed tokens, report both:

- Conservative: assume full duplicate is billed.
- Optimistic: assume only a fraction `kappa in [0,1]` of duplicate tokens are billed:
  `ΔCost_hedge ≈ N * rho * kappa * (...)`.

### Probing Cost

Let:

- `J`: number of providers
- `f`: probe frequency per provider (probes/hour)
- `H`: total hours
- `t_in_probe`, `t_out_probe`: tokens per probe
- `p_probe`: representative token price for probing (choose the same model)

Then:

`Cost_probe ≈ J * f * H * (t_in_probe + t_out_probe) * p_probe`

This is typically negligible compared to replay.

### Risk Controls (Avoid Wasting Cost)

1. Start with a **dry run** on `N=50–200` requests.
2. Enforce a hard budget cap `B` per run:
   - Stop early when cumulative billed cost exceeds `B`.
3. Use small `max_tokens` for the queueing-dominant regime.
4. Prefer a **cheap single model** with multiple OpenRouter providers to make large-scale replay
   affordable.

## Recommended Parameter Settings

### Probing Parameters

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `W` (window size) | 20–50 probes | Balance between responsiveness and stability |
| `f` (probe frequency) | 1–2 probes/min/provider | Avoid rate limiting; ~60–120 probes/hour |
| Probe `max_tokens` | 8–16 | Minimize cost while measuring TTFT reliably |

### Hedging Parameters

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `h` (hedge threshold) | P70–P80 of primary | Trigger hedge when primary is "slow" |
| Hedge provider | Lowest-P99 in pool | Use fastest provider as backup |

### SLO Settings

| Regime | Recommended SLO | Notes |
|--------|-----------------|-------|
| Short-output | 2–5 seconds E2E | Focus on TTFT + queueing |
| Long-output | 10–30 seconds E2E | Include decode time |

### Budget Allocation (Example: $50 total)

| Component | Allocation | Estimated Requests |
|-----------|------------|-------------------|
| Probing (Week 1) | $5 | ~50K probes @ $0.10/1K |
| Short-output replay | $20 | ~200K requests @ $0.10/1K |
| Long-output replay | $15 | ~30K requests @ $0.50/1K |
| Hedging experiments | $10 | +50% overhead on subset |

## Recommended Milestones

1. **Week 1: Plot real latency distributions**
   - TTFT/E2E ECDF and CCDF by provider; show variance at fixed token length.
2. **Week 2: OpenRouter online replay (short-output)**
   - Compare OpenRouter default vs SLO-first vs skyline; compute cost–P99 tradeoff.
3. **Week 3: Add hedging**
   - Compare against always-duplicate; quantify hedging overhead.
4. **Week 4: Paper integration**
   - One main plot: cost vs P99 Pareto curve with baselines.
   - One supporting plot: provider CCDF drift over time (motivates online profiling).
