# RouteWise: Cut Your Inference Costs by 20-40% with Subscription Procurement

## The Problem

OpenRouter currently pays every inference provider per-token. This is simple, but expensive.

However, open-weight model inference has a key economic property: **GPU capacity costs are mostly fixed.** Whether a provider serves 1K or 100K tokens/sec on a provisioned GPU cluster, the hardware cost stays the same. This creates room for a new procurement model: **subscription-based capacity reservation** -- a fixed monthly fee for a guaranteed daily quota of token-normalized credits, with zero marginal cost within that capacity.

The catch: subscriptions have limited capacity. If you route requests naively (first-come-first-served), you waste scarce quota credits on cheap "Hello world" requests while expensive coding tasks still hit the pay-per-token API. Greedy routing captures only **27-50%** of the potential savings.

## The Proposal

We propose a **wholesale subscription procurement layer** for OpenRouter:

1. **OpenRouter negotiates subscription deals** with 1-2 inference providers (e.g., Chutes for DeepSeek models) -- a fixed monthly fee for a daily quota of token-normalized credits
2. **Our routing algorithm** decides, for each incoming request, whether to route it to a subscription slot or the pay-per-token API
3. **Both parties benefit** -- the provider gets guaranteed revenue and capacity planning certainty; OpenRouter gets lower costs

We provide the routing algorithm and the economic framework. OpenRouter provides the traffic.

## Why It Works: The Economics

We analyzed 13.7M real requests from a Chutes DeepSeek-V3.1 trace (30 days). Key findings:

**Request costs are highly skewed** (Gini = 0.54). A small fraction of requests (long coding/reasoning tasks) account for most of the cost. Smart routing exploits this by reserving subscription slots for high-value requests.

| Daily Quota (% of avg) | Savings with Smart Routing | Savings with Greedy | Smart Routing Advantage |
|---|---|---|---|
| 10% | 38% | 11% | **3.6x** |
| 30% | 67% | 32% | **2.1x** |
| 50% | 83% | 52% | **1.6x** |

**Smart routing is 1.6-3.6x better than greedy.** Without it, subscription procurement doesn't make economic sense. With it, the savings are substantial.

### Win-Win Pricing Framework

We model this as a **Stackelberg contract game** with Nash bargaining for the fee split:

- **Participation constraints** (when does a deal make sense for both sides?):
  - OpenRouter: subscription fee < savings from smart routing (V_A(Q))
  - Provider: subscription fee > outside option (existing per-token revenue on the covered tranche + incremental capacity cost)
- **Fee determination**: Nash bargaining (equal surplus split) within the feasible zone

The feasible zone depends on two scenario parameters:
- **alpha_tranche**: provider's current share of the subscription-covered traffic (0 = pure new traffic, 1 = pure cannibalization)
- **Delta_K**: incremental capacity reservation cost

Using the Chutes trace data (Q = 50% of daily volume):

| Scenario | OpenRouter Savings | Win-Win Zone Width |
|---|---|---|
| Provider has no existing traffic (alpha=0) | **41%** | $31K/mo |
| Provider has 30% existing share (alpha=0.3) | **29%** | $22K/mo |
| Provider has 50% existing share (alpha=0.5) | **21%** | $16K/mo |

Even in conservative scenarios, there is a significant pricing zone where both parties are better off.

**Key insight**: The routing algorithm's quality (gamma = savings capture ratio) directly determines the size of the feasible contract set. Better routing expands the win-win zone.

## What We're Asking

**A 1-model pilot**, starting with DeepSeek-V3.1 (or DeepSeek-R1) on Chutes:

- **OpenRouter's effort**: Negotiate a subscription with Chutes (daily quota of token-normalized credits, fixed monthly fee); integrate our routing logic (a lightweight threshold-based decision per request -- dozens of lines of code)
- **Timeline**: 1 month to see results
- **Risk**: Minimal. If savings don't materialize, stop the subscription. No infrastructure changes required.
- **Our role**: We provide the routing algorithm, monitoring, and optimization. We're a research lab (Harvard), not a vendor -- our goal is to validate this in production and publish the results.

## What OpenRouter Gets

1. **20-40% cost reduction** on the pilot model, scaling to more models over time
2. **A new procurement lever** -- subscription deals give you negotiating power with providers and insulate you from per-token price volatility
3. **Latency improvements** -- our system also includes latency-aware provider selection with smart hedging (31x reduction in SLO violations in our experiments)

## What the Provider Gets

1. **Guaranteed monthly revenue** -- subscription fee is paid regardless of traffic fluctuations
2. **Capacity planning** -- predictable committed volume makes GPU provisioning easier
3. **Customer lock-in** -- subscription creates a sticky relationship

## About Us

We are the Harvard MadSys Lab. Our routing system, RouteWise, combines:

- **Online algorithms** with provable competitive ratios for cost-optimal routing
- **Learning-augmented predictions** for output token estimation
- **LP-based latency optimization** with survival-analysis-based hedging

The system has been validated on 4 production traces (ShareGPT, FreeInference, Enterprise, BurstGPT) totaling 2M+ requests. We are preparing a submission to **NSDI** (top systems conference).

A real deployment on OpenRouter would benefit both sides -- OpenRouter saves money, and we get production validation for our research.

---

**Contact**: Harvard MadSys Lab

**One-liner**: *Buy subscriptions from your providers, and let our algorithm decide which requests deserve them. Save 20-40%. No infrastructure changes.*
