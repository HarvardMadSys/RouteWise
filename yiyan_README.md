# RouteWise Synthetic Simulation Task

Post-refactor path note: this is an older task brief. Use `rwsim/` and
`experiments/` for new work; `legacy/experiment/...` references below describe
the original implementation or compatibility wrappers.

## Goal

Build a synthetic simulation environment with mock providers to verify routing algorithm behavior under controlled conditions. This helps us understand the system without real-world noise.

## Background

RouteWise routes LLM requests across multiple providers that serve the same model. The latency router picks which provider gets each request. We have two routing approaches:

1. **LP Mix** (`legacy/experiment/strategies/online_latency_router.py`): Solves an LP to find provider weights that minimize cost subject to `sum(pi_j * F_j(SLO)) >= 0.99`. Updates weights every 60s based on rolling latency profiles.
2. **V2 P50 Router** (`legacy/experiment/strategies/v2_router.py`): Ranks providers by P50 latency, picks cheapest in near-best band. Simpler, no LP.
3. **Smart Hedging** (`legacy/experiment/strategies/smart_hedging.py`): Sends backup request to faster provider when primary is slow. Economic decision rule: hedge when `P(violation) * P(backup_succeeds) > C_backup / V_penalty`.

**Problem we found**: LP Mix over-diversifies when one provider dominates (both cheapest and fastest). It spreads traffic to worse providers, causing MORE SLO violations than a trivial "always pick cheapest" baseline.

**What we need from synthetic simulation**: Controlled scenarios where we can verify:
- Algorithm correctness (picks the obviously best provider)
- Behavior under trade-offs (cheap-but-slow vs fast-but-expensive)
- Adaptation to latency shifts
- Value of hedging for tail protection

## What You Need to Build

### 1. `SyntheticProvider` class

A mock provider that samples latency from a known distribution.

```python
@dataclass
class SyntheticProvider:
    name: str
    cost_per_token: float  # $/token
    ttft_distribution: Distribution  # e.g., LogNormal(mu, sigma)
    tps_distribution: Distribution   # tokens per second
    
    def sample_ttft(self) -> float:
        """Sample a TTFT in milliseconds."""
        ...
    
    def sample_request(self, output_tokens: int) -> tuple[float, float]:
        """Return (ttft_ms, e2e_ms) for a request with given output tokens."""
        ...
```

Suggested distributions:
- **LogNormal** for TTFT (matches real provider behavior: right-skewed, heavy tail)
- Parameters: `mu` (log-mean), `sigma` (log-std). Higher sigma = heavier tail = worse P99

### 2. `SyntheticWorkload` generator

Generate a stream of requests with timestamps.

```python
def generate_workload(
    n_requests: int,
    duration_seconds: float,
    arrival_process: str = "poisson",  # or "bursty"
    output_token_dist: Distribution = LogNormal(mu=4.0, sigma=1.0),
) -> list[Request]:
    ...
```

Use the `Request` dataclass from `legacy/experiment/data/schema.py`.

### 3. Scenario Configurations

Start with these scenarios (from Juncheng's meeting guidance):

| Scenario | Provider A | Provider B | Provider C | Expected Winner |
|----------|-----------|-----------|-----------|----------------|
| S1: Dominant | Slow+Expensive | Fast+Cheap | - | 100% B |
| S2: Trade-off | Cheap+Slow (P50=500ms) | Fast+Expensive (P50=100ms) | - | Depends on SLO |
| S3: Tail-heavy | Good P50 (100ms), Bad P99 (5s) | OK P50 (300ms), OK P99 (1s) | - | B if SLO tight |
| S4: Shift | A starts fast, slows at t=T/2 | B constant | - | A then B |
| S5: Similar | P50=200ms, $1/M | P50=220ms, $0.8/M | P50=250ms, $0.5/M | C (cheapest in band) |

### 4. Simulation Runner

For each scenario, run all routing strategies and collect:
- SLO violation rate (for SLO thresholds: 1s, 2s, 3s, 5s)
- Average cost per request
- Provider selection distribution over time
- P50 and P99 of actual latencies

### 5. Plots

For each scenario generate:
1. **Bar chart**: SLO violation rate by strategy
2. **Bar chart**: Average cost by strategy  
3. **Time series**: Provider selection fraction over time (especially for S4 shift scenario)
4. **CDF**: Latency CDF by strategy

## Strategies to Test

| Strategy | Code | Description |
|----------|------|-------------|
| `cheapest_fixed` | Pick cheapest provider always | Baseline |
| `fastest_fixed` | Pick lowest-P50 provider always | Baseline |
| `round_robin` | Rotate uniformly | Baseline |
| `lp_mix` | `OnlineLatencyRouter` | LP-based mixing |
| `v2_p50_hedge` | `V2Router` + `SmartHedger` | New P50 ranking + hedging |
| `oracle_per_window` | Best provider per 15-min window (hindsight) | Upper bound |

## Key Files to Read

- `legacy/experiment/strategies/v2_router.py` — **Start here**. The V2 router is simple and clean.
- `legacy/experiment/strategies/smart_hedging.py` — Hedging logic, focus on `SMART_ECONOMIC` strategy.
- `legacy/experiment/strategies/online_latency_router.py` — LP router (the one that over-diversifies).
- `legacy/experiment/scripts/simulate/offline_counterfactual.py` — **Reference implementation** for how we run counterfactual simulations. Your synthetic sim should follow a similar pattern.
- `legacy/experiment/scripts/simulate/policies.py` — How policies are defined and dispatched.
- `legacy/experiment/data/schema.py` — Data structures (`Request`, `ProviderConfig`, `RoutingDecision`).
- `MEMORY/v2-router-design.md` — V2 router design rationale.
- `MEMORY/acf-analysis-probing-effectiveness.md` — Why P50 works but P99 doesn't for prediction.

## Sample Data

- `sample_data/evaluation_log_sample.csv` — 100 rows from a real online evaluation. Shows the schema.
- `sample_data/counterfactual_sample.csv` — Counterfactual simulation output format.
- `sample_data/provider_percentiles_sample.csv` — 15-min window provider percentiles.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
uv pip install numpy scipy matplotlib pandas pyyaml tqdm
```

## Output

Put results in `results/synthetic/` with one subdirectory per scenario:
```
results/synthetic/
  s1_dominant/
    summary.json       # {strategy: {slo_viol, cost, p50, p99}}
    slo_violation.png
    cost_comparison.png
    provider_selection.png   # (for shift scenario)
    latency_cdf.png
  s2_tradeoff/
    ...
```
