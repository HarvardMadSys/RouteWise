# Online Experiment Plan: FreeInference Live Serving

This document summarizes the FreeInference dataset status and proposes a new online
experiment plan that upgrades from trace replay to live serving with real routing decisions.

---

## 1. FreeInference Dataset Status

### 1.1 Overview

| Metric | Value |
|--------|-------|
| Total requests | 420,373 |
| Date range | 2025-10-05 ~ 2026-01-02 (90 days, 62 active days) |
| Unique models | 14 |
| Prompt tokens (mean / median) | 1,212 / 326 |
| Completion tokens (mean / median) | 148 / 51 |

### 1.2 Model Distribution

| Model | Requests | Share |
|-------|----------|-------|
| llama-3.3-70b-instruct | 246,567 | 58.7% |
| llama-4-scout | 54,484 | 13.0% |
| gemini-2.5-flash | 42,665 | 10.1% |
| llama-4-maverick | 30,521 | 7.3% |
| gemini-2.5-flash-preview | 27,460 | 6.5% |
| glm-4.5 | 12,324 | 2.9% |
| MiniMax-m2 | 669 | 0.2% |
| Others | ~5,683 | 1.3% |

### 1.3 Daily Traffic Distribution

The traffic is **highly skewed**:

| Stat | Requests/day |
|------|-------------|
| Mean | 6,780 |
| Median | 270 |
| P25 | 38 |
| P75 | 3,345 |
| Max | 66,005 |

- 97% of total requests are concentrated in ~18 high-traffic days (>2K reqs/day).
- The remaining ~44 days have very low traffic (<270 reqs/day).

### 1.4 Per-Model Quota Scarcity Analysis

For the Primal-Dual routing algorithm to be meaningful, daily traffic must exceed the quota
threshold Q so that the algorithm faces a non-trivial allocation decision.

| Model | Active Days | Days > Q=300 | Days > Q=2000 | Days > Q=5000 |
|-------|------------|--------------|---------------|---------------|
| llama-3.3-70b | 38 | 19 (50%) | 14 (37%) | 12 (32%) |
| llama-4-scout | 21 | 8 (38%) | 4 (19%) | 3 (14%) |
| glm-4.5 | 12 | 2 (17%) | 1 (8%) | 0 (0%) |
| MiniMax-m2 | 11 | 0 (0%) | 0 (0%) | 0 (0%) |

**Key takeaway**: Only llama-3.3-70b and llama-4-scout have sufficient daily traffic
to create meaningful quota scarcity. GLM-4.5 and MiniMax traffic is too sparse for
quota-constrained experiments with Q >= 300.

### 1.5 Current Experiment Approach: Trace Replay

The current experiments use **offline trace replay**:
- Historical request logs are replayed in timestamp order.
- Online strategies (Greedy, Primal-Dual, LA-PD) make routing decisions sequentially.
- Offline Optimal runs with full hindsight as the upper bound.
- No real API calls are made; costs are computed using the pricing model.

This approach is sufficient for algorithm comparison but does **not** capture:
- Real-time latency dynamics (TTFT/TPS variation).
- Provider-side rate limiting and throttling behavior.
- Actual end-to-end serving performance under concurrent load.

---

## 2. Provider Rate Limit Profiling (Prerequisite)

Before running the online experiment, we need to **profile the rate limits of each
external provider** for our target models. The router itself is trivial (Python,
simple decision logic) and is not the bottleneck. The real constraint is how much
traffic each provider can absorb before returning 429s or degrading latency.

### 2.1 Motivation

- Each provider (FreeInference, Together, Groq, SambaNova, etc.) has its own
  undocumented or partially documented rate limits (RPM, TPM, concurrency).
- When our router sends traffic to a provider, we need to know the **actual
  sustainable request rate** so we don't hit rate limits during the experiment.
- Rate limit behavior varies by model — a provider may have high capacity for
  llama-3.3-70b but low capacity for a less popular model.
- Per Juncheng's suggestion: **limit to 1-2 models with high capacity across
  multiple providers** so we have enough headroom to route between them.

### 2.2 Model Selection

Focus on models available on multiple providers with relatively high capacity:

| Model | FreeInference | Together | Groq | SambaNova |
|-------|--------------|----------|------|-----------|
| llama-3.3-70b-instruct | Yes (59% of traffic) | Yes | Yes | Yes |
| llama-4-scout | Yes (13% of traffic) | Yes | TBD | TBD |

**Primary model**: llama-3.3-70b-instruct — highest traffic volume in our dataset,
available on all major providers, likely highest capacity allocation.

**Secondary model** (optional): llama-4-scout — second-highest traffic, provides
multi-model validation.

### 2.3 Test Plan

For each (model, provider) pair:

**Phase 1: Baseline Profiling**
- Send single sequential requests (50+ iterations).
- Measure baseline TTFT, TPS, total latency at zero contention.

**Phase 2: Rate Limit Discovery**
- Gradually ramp up request rate: 1, 2, 5, 10, 20, 50, 100 RPM.
- At each level, send a batch of requests and record:
  - Success rate (2xx vs. 429 / 503 / timeout)
  - TTFT distribution (P50, P95, P99)
  - Observed rate limit headers (X-RateLimit-Remaining, Retry-After, etc.)
- Identify the **max sustainable RPM** per provider per model.

**Phase 3: Sustained Load (at safe RPM)**
- Run 1-hour sustained traffic at 80% of the discovered max RPM.
- Monitor for:
  - Latency drift over time (does the provider throttle after sustained load?)
  - Intermittent 429 bursts or quota resets
  - Token-level throughput stability

### 2.4 Key Metrics to Collect

| Metric | Purpose |
|--------|---------|
| Max RPM per (model, provider) | Sets feasible request rate for experiment |
| 429 onset threshold | Determines safe operating point |
| Rate limit window & reset | Informs quota window alignment |
| TTFT at operating RPM | Validates latency model under real load |
| Concurrency limit (if applicable) | Sets S_C parameter C |

### 2.5 Expected Deliverables

1. A rate-limit profile table: max safe RPM for each (model, provider) pair.
2. Recommended model choice (1-2 models with best multi-provider capacity).
3. Go/no-go decision: do the selected providers support enough aggregate
   throughput for a meaningful online experiment?

---

## 3. Online Experiment Plan

### 3.1 Upgrade from Trace Replay to Live Serving

The advisor's suggestion: instead of replaying historical traces, run a **live online
experiment** where the Primal-Dual router makes real routing decisions against live
FreeInference and pay-per-token APIs.

**Architecture:**

```
User requests (replayed or live)
       |
       v
+------------------+
| RouteWise Router |  <-- Primal-Dual / LA-PD decision engine
+------------------+
       |
   +---+---+
   |       |
   v       v
 S_Q      S_A
(FreeInference)  (Together / Groq / etc.)
```

The key difference from trace replay: the router **actually sends requests** to the
selected provider and observes real latency, real token counts, and real errors.

### 3.2 Why FreeInference as S_Q

FreeInference provides free inference for selected models. In our framework:
- **S_Q (Quota subscription)**: FreeInference has implicit daily/hourly usage limits.
  We can also artificially cap usage at Q requests/day to create controlled scarcity.
- **S_A (Pay-per-token)**: Together AI, Groq, etc. serve as the pay-per-token fallback.
- **Cost metric**: Dollars saved = requests routed to S_Q * (API price per request).

### 3.3 Model Selection (Per Juncheng's Advice)

Limit to **1-2 models with high capacity across multiple providers** to ensure we
have enough headroom for routing between S_Q and S_A without hitting rate limits:

- **Primary: llama-3.3-70b-instruct** — 59% of FreeInference traffic, available on
  FreeInference / Together / Groq / SambaNova. Highest provider capacity.
- **Secondary (optional): llama-4-scout** — 13% of traffic, validates multi-model
  generalization.

This avoids the risk of picking a low-capacity model (e.g., GLM-4.5 or MiniMax)
where provider rate limits would constrain the experiment before quota scarcity
becomes the binding factor.

### 3.4 Traffic Source Strategy

**Problem**: Raw FreeInference traffic is too bursty and sparse on most days.

**Solution**: Use **controlled replay with rate scaling**:
1. Select the 18 high-traffic days (>2K reqs/day) from historical data.
2. Replay these requests in real-time against live providers.
3. Cap replay rate at the safe RPM discovered in Section 2.
4. Set Q (daily quota) to create meaningful scarcity:
   - Q = 300 (Base): ~50% of high-traffic days are quota-constrained.
   - Q = 2000 (Plus): ~30% of days are quota-constrained.
   - Q = 5000 (Pro): ~25% of days are quota-constrained.

### 3.5 Experiment Design

| Parameter | Setting |
|-----------|---------|
| Duration | 7 days (selecting 7 high-traffic days from trace) |
| Models | llama-3.3-70b-instruct (primary) |
| S_Q provider | FreeInference (with quota cap Q) |
| S_A provider | Together AI (pay-per-token, rate-limit safe) |
| Quota levels | Q = {300, 2000, 5000} |
| Request rate | Capped at safe RPM from rate-limit profiling |
| Strategies | Greedy, Primal-Dual (EMA), LA-PD (P10), Offline Optimal (post-hoc) |
| Metrics | Total cost ($), Savings (%), Quota utilization, TTFT (P50/P99) |

### 3.6 Online-Specific Measurements

Beyond cost savings, the online experiment enables:

1. **End-to-end latency comparison**: Real TTFT when routing to FreeInference vs.
   Together AI. Validates that cost savings don't come at unacceptable latency cost.
2. **Provider reliability**: Measure real error rates, retries, and timeouts.
3. **Decision latency overhead**: Time spent in the routing decision itself
   (should be <1ms).
4. **Prediction accuracy in production**: Compare predicted output tokens (P10)
   against actual completions in a live setting.

### 3.7 Execution Steps

1. **[Rate-Limit Profiling]** Profile provider rate limits for llama-3.3-70b on
   FreeInference and Together AI (Section 2).
2. **[Instrumentation]** Add request-level logging: timestamp, routing decision,
   provider, TTFT, TPS, tokens, cost, error code.
3. **[Quota Manager]** Implement a real-time quota tracker that enforces Q per day
   (not just simulated).
4. **[Router Integration]** Connect Primal-Dual strategy to live HTTP clients for
   FreeInference and Together AI.
5. **[Dry Run]** Run 1-day experiment with Q=2000 to validate the pipeline.
6. **[Full Experiment]** Run 7-day experiments at each quota level.
7. **[Post-hoc Analysis]** Compute offline optimal on the collected trace to get
   the savings upper bound.

### 3.8 Expected Results

Based on trace replay results (from `stage1_comparison.json`):

| Dataset | Plan | Greedy Savings | PD Savings | Optimal Savings |
|---------|------|---------------|------------|-----------------|
| BurstGPT | Base (Q=300) | 1.4% | 2.5% | 5.4% |
| BurstGPT | Plus (Q=2000) | 10.0% | 16.0% | 22.7% |
| BurstGPT | Pro (Q=5000) | 20.2% | 29.8% | 38.5% |
| FreeInference | Base (Q=300) | 37.4% | 42.8% | 56.8% |
| FreeInference | Plus (Q=2000) | 46.7% | 54.6% | 63.1% |
| FreeInference | Pro (Q=5000) | 38.6% | 46.2% | 56.9% |

The online experiment should produce similar savings ratios, with the added benefit
of real latency measurements to demonstrate the system works end-to-end.

---

## 4. Timeline

| Week | Task |
|------|------|
| Week 1 | Provider rate-limit profiling (llama-3.3-70b on FreeInference + Together) |
| Week 2 | Build live router + instrumentation pipeline |
| Week 3 | Dry run (1-day, single quota level) |
| Week 4 | Full 7-day experiment (all 3 quota levels) |
| Week 5 | Analysis, plots, paper integration |
