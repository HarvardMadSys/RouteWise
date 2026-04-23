# Phase 3: Online Latency-Aware Provider Routing

## 1. Overview

### 1.1 Objective

Implement an **online latency-aware routing** module that selects providers from a multi-provider platform (e.g., OpenRouter) to minimize cost while satisfying a **tail constraint**: `Σ π_j · F_j(L) ≥ 0.99`.

### 1.2 Key Design Principles

1. **Latency as an independent dimension**: Does not interfere with existing cost optimization (Stage 1/2). Applied when using API fallback with multi-provider platforms.

2. **LP-based optimal mixing**: Use Linear Programming to find the optimal provider mix, not heuristic Pareto selection. The LP naturally yields at most 2 non-zero providers (simplex property with one constraint).

3. **Online profiling with moving windows**: Maintain latency distributions via probing requests, using time-based windows to capture non-stationarity.

4. **Reliability-aware**: Failures (timeout, 429, 5xx) are treated as missed deadlines, naturally incorporated into the CDF constraint.

### 1.3 Scope

- **In scope**: Online LP-mix routing, probing-based profiling, BurstGPT replay evaluation
- **Out of scope (Phase 4)**: Smart hedging, advanced cost penalties (kappa tuning)

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Online Latency Router                            │
│                                                                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  Probing Module  │───>│  Profile Store   │───>│   LP Solver      │  │
│  │  (45s/provider)  │    │  (short+long)    │    │  (scipy.linprog) │  │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
│           │                       │                       │             │
│           │                       ▼                       ▼             │
│           │              ┌──────────────────┐    ┌──────────────────┐  │
│           │              │  Pre-filter      │    │  SWRR Sampler    │  │
│           │              │  (error/latency) │    │  (quota-based)   │  │
│           │              └──────────────────┘    └──────────────────┘  │
│           │                                               │             │
│           ▼                                               ▼             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Request Router                                 │  │
│  │                    route(request) -> provider                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Components

### 3.1 Moving Window Profiler

Maintains per-provider latency distributions using **time-based windows**. Uses **Mixed-Window CDF** with adaptive shrinkage to handle small sample variance.

#### 3.1.1 Mixed-Window CDF with Adaptive Shrinkage

```python
def get_cdf_at(self, L_sec: float) -> float:
    """
    Compute mixed-window CDF with adaptive shrinkage:

    F_hat(L) = β · F_short(L) + (1 - β) · F_long(L)

    where β = N_eff / (N_eff + prior_strength)

    This is equivalent to an empirical Bayes / pseudo-count estimator:
    - Long window provides `prior_strength` pseudo-samples as stable prior
    - Short window provides N_eff new observations

    Default prior_strength (λ) = 10:
    - 20 short samples → β ≈ 0.67 (67% short, 33% long)
    - 10 short samples → β = 0.50 (equal weight)
    - 5 short samples  → β ≈ 0.33 (33% short, 67% long)
    """
```

**Interpretation of `prior_strength` (λ)**:
- Represents how many "pseudo-samples" the long window contributes as prior
- Aligns with time scale: short window (15min, ~20 samples) with λ=10 means long window contributes ~30% weight
- **Sensitivity analysis**: λ ∈ {10, 20, 50} in ablation study

#### 3.1.2 Failure Handling Modes

**SLO Semantic**: "User request returns successfully within L seconds"

Under this semantic, timeouts and errors are SLO violations and should be treated as missed deadlines.

```python
class FailureMode(Enum):
    INFINITY = "infinity"   # Mode A: failures = latency ∞ (missed deadline)
    SEPARATE = "separate"   # Mode B: F_eff = (1 - err_rate) * F_success
```

**Mode A (Recommended Main)**: Failures as Infinity
- Timeout/error counts as missed deadline (latency = ∞)
- CDF naturally incorporates reliability: `F(L) = P(success AND TTFT ≤ L)`
- Very clean: tail constraint simultaneously manages latency AND reliability
- Example: 5% timeout → `F(L) ≤ 0.95` for any L

**Mode B (Ablation)**: Separate Reliability
- `F_effective(L) = (1 - error_rate) · F_success(L)`
- `F_success(L)` computed only on successful requests
- Useful when errors are rate-limits (not true slowness)
- Allows separate reporting of failure rate

#### 3.1.3 Data Structure

```python
@dataclass
class ProviderProfile:
    """Real-time latency profile for a provider."""
    provider: str

    # Latency samples: list of (timestamp, ttft_ms)
    short_window_samples: List[Tuple[float, float]]  # 15 min
    long_window_samples: List[Tuple[float, float]]   # 2-3 hours

    # Error tracking: list of (timestamp, error_type)
    # error_type: "timeout" | "rate_limit" | "server_error" | None (success)
    error_samples: List[Tuple[float, Optional[str]]]

    # Configuration
    prior_strength: float = 10.0  # λ for shrinkage
    failure_mode: FailureMode = FailureMode.INFINITY

    def get_cdf_at(self, L_sec: float) -> float:
        """Compute mixed-window CDF with adaptive shrinkage."""

    def get_error_rates(self, window_minutes: float = 15) -> Dict[str, float]:
        """Return {"timeout": rate, "rate_limit": rate, "server_error": rate}."""

    def get_p99(self, use_short: bool = True) -> float:
        """Return P99 latency in seconds (for reporting only)."""
```

**Window Configuration**:
| Window | Duration | Purpose |
|--------|----------|---------|
| Short | 15 minutes | Adaptive to drift, primary for LP |
| Long | 2-3 hours | Stable prior, shrinkage target |

### 3.2 Pre-filter (Hard Constraints)

Before LP, filter out providers that are clearly unusable:

```python
def pre_filter(
    profiles: Dict[str, ProviderProfile],
    L_min: float = 1.0,
    max_error_rate: float = 0.05,
    min_cdf_threshold: float = 0.80,
) -> List[str]:
    """
    Hard filtering rules:
    1. total_error_rate > 5% → offline (clearly broken)
    2. F_hat(L_min) < 0.80 → offline (can't meet even relaxed SLO)

    Returns list of eligible provider names.
    """
    eligible = []
    for name, profile in profiles.items():
        error_rates = profile.get_error_rates()
        total_error = sum(error_rates.values())

        if total_error > max_error_rate:
            continue  # Hard offline: clearly broken

        if profile.get_cdf_at(L_min) < min_cdf_threshold:
            continue  # Can't meet basic SLO

        eligible.append(name)

    return eligible
```

### 3.3 LP Solver

Solve the cost-minimization LP with a **tail constraint** (CDF at L).

#### 3.3.1 Formulation

```
minimize:   Σ π_j · c_j · (1 + κ · e_j)
subject to: Σ π_j · F_j(L) ≥ 0.99
            Σ π_j = 1
            π_j ≥ 0
```

Where:
- `c_j`: Cost per request for provider j (from token pricing)
- `F_j(L)`: Mixed-window CDF at SLO L (failures handled per failure_mode)
- `e_j`: Error rate for provider j
- `κ`: Error penalty coefficient (default: 0.0)

**Why κ=0 as default?**
- Failures are already incorporated into `F_j(L)` via failure_mode (Mode A)
- `κ > 0` would double-count errors
- `κ` is retained as secondary preference for ablation: {0, 5, 10}

#### 3.3.2 Relaxed Fallback for Infeasible LP

If the LP is infeasible (no provider mix can achieve 0.99 at given SLO), use progressive relaxation:

```python
def solve_lp_with_fallback(
    providers: List[str],
    profiles: Dict[str, ProviderProfile],
    costs: Dict[str, float],
    slo_sec: float,
    relaxation_factors: List[float] = [1.2, 1.5, 2.0],
    kappa: float = 0.0,
) -> Tuple[Dict[str, float], str]:
    """
    Solve LP with progressive relaxation fallback.

    Returns:
        (weights_dict, status) where status is one of:
        - "optimal": Original SLO achieved
        - "relaxed_1.2x" / "relaxed_1.5x" / "relaxed_2.0x": Relaxed SLO used
        - "best_effort": All relaxations failed, use max F_j(L) provider
    """
    # Try original SLO
    result = solve_lp(providers, profiles, costs, slo_sec, kappa)
    if result is not None:
        return result, "optimal"

    # Try relaxed SLOs
    for factor in relaxation_factors:
        relaxed_slo = slo_sec * factor
        result = solve_lp(providers, profiles, costs, relaxed_slo, kappa)
        if result is not None:
            return result, f"relaxed_{factor}x"

    # Best-effort: select provider with max F_j(L)
    best_provider = max(providers, key=lambda p: profiles[p].get_cdf_at(slo_sec))
    return {best_provider: 1.0}, "best_effort"
```

#### 3.3.3 Why at most 2 providers?

The LP has:
- n variables (π_1, ..., π_n)
- 1 inequality constraint (SLO)
- 1 equality constraint (sum = 1)

By the fundamental theorem of LP, an optimal basic feasible solution has at most `m` non-zero variables where `m` is the number of constraints. Here m=2, so at most 2 providers.

### 3.4 Smooth Weighted Round-Robin (SWRR) Sampler

Instead of pure random sampling (which causes short-term variance), use **Smooth Weighted Round-Robin** for deterministic interleaving.

#### 3.4.1 Algorithm

```python
class SWRRSampler:
    """
    Smooth Weighted Round-Robin sampler.

    Given weights π = {A: 0.7, B: 0.3}, produces a smooth interleaving:
    A, A, B, A, A, B, A, A, A, B, ...

    Equivalent to probabilistic mixing but with reduced short-term variance.
    """

    def __init__(self, weights: Dict[str, float]):
        self.providers = list(weights.keys())
        self.weights = weights.copy()
        self.current_weights = {p: 0.0 for p in self.providers}

    def next(self) -> str:
        """Select next provider using SWRR algorithm."""
        # Add original weights
        for p in self.providers:
            self.current_weights[p] += self.weights[p]

        # Select provider with highest current weight
        selected = max(self.providers, key=lambda p: self.current_weights[p])

        # Subtract total weight from selected
        self.current_weights[selected] -= sum(self.weights.values())

        return selected
```

#### 3.4.2 Weight Updates with Smoothing

```python
def update_weights(self, new_weights: Dict[str, float], smoothing: float = 0.3):
    """
    Update weights with exponential smoothing to avoid abrupt changes.

    π_new = smoothing · π_LP + (1 - smoothing) · π_old

    Also performs soft reset of current_weights to prevent drift.
    """
    # Handle provider changes
    all_providers = set(self.providers) | set(new_weights.keys())

    for p in all_providers:
        old_w = self.weights.get(p, 0.0)
        new_w = new_weights.get(p, 0.0)
        self.weights[p] = smoothing * new_w + (1 - smoothing) * old_w

    # Remove providers with negligible weight
    self.weights = {p: w for p, w in self.weights.items() if w > 0.001}
    self.providers = list(self.weights.keys())

    # Normalize
    total = sum(self.weights.values())
    if total > 0:
        self.weights = {p: w/total for p, w in self.weights.items()}

    # Soft reset current_weights to prevent accumulation drift
    self.current_weights = {p: self.current_weights.get(p, 0.0) * 0.5
                           for p in self.providers}
```

**Why SWRR over random sampling?**
- Random: π=[0.7, 0.3] might give "AAAA" or "BBBB" in short bursts
- SWRR: Guarantees smooth distribution "AABABAA..."
- Same long-term distribution, lower short-term variance

**Why smoothing on π updates?**
- LP solution can jump when profiles change
- Smoothing prevents oscillation between providers
- Default α=0.3: 30% new + 70% old

---

## 4. Parameters

### 4.1 SLO Values for Evaluation

| SLO (seconds) | Expected Behavior |
|---------------|-------------------|
| 1s | Tight: only fastest providers (Groq) |
| 2s | Moderate: mix of fast providers |
| 3s | **Key inflection point**: start seeing 2-provider mix |
| 5s | Relaxed: cheaper providers become viable |
| 10s | Very relaxed: mostly cheapest provider |

### 4.2 Profiling Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Probe interval | 45 seconds/provider | Matches Phase 1 baseline |
| Probe request | "Say OK" (minimal tokens) | Measures TTFT, not decode |
| Short window | 15 minutes | Captures drift, ~20 samples |
| Long window | 3 hours | Stable estimate, ~240 samples |
| Latency metric | TTFT | Short-output probing = queueing/prefill |
| LP update throttle | 60 seconds | Prevent oscillation |

### 4.3 Shrinkage and Filtering

| Parameter | Default | Ablation |
|-----------|---------|----------|
| prior_strength (λ) | 10 | {10, 20, 50} |
| Hard offline threshold | error_rate > 5% | - |
| Minimum CDF threshold | F_hat(1s) < 80% | - |
| Failure mode | INFINITY | {INFINITY, SEPARATE} |

### 4.4 LP and Sampling

| Parameter | Default | Ablation |
|-----------|---------|----------|
| Error penalty κ | 0.0 | {0, 5, 10} |
| Relaxation factors | [1.2, 1.5, 2.0] | - |
| Weight smoothing α | 0.3 | - |

---

## 5. Evaluation Plan

### 5.1 Simulation Setup

1. **Profile source**: Phase 1 data (latency_llama70b_24h.csv)
   - **Time Causality**: At decision time `t`, only use probe samples with timestamp `< t`
   - **Evaluation**: Use nearest-neighbor interpolation as "posterior ground truth"
   - Evaluation data is NOT fed back into router's profile (no data leakage)
   - Use "base" workload only (TTFT-focused)

2. **Workload**: BurstGPT trace (200-500 requests)
   - Replay with realistic arrival times
   - Sample latency from current profile (simulating real provider response)

3. **Baselines**:
   | Policy | Description |
   |--------|-------------|
   | **Single-Best** | Always use provider with lowest P99 |
   | **Cheapest** | Always use cheapest provider |
   | **Random** | Uniform random among eligible |
   | **LP-Mix (Ours)** | Online LP with SWRR sampling |

### 5.2 Metrics

| Metric | Definition |
|--------|------------|
| **Cost** | Total $ spent / number of requests |
| **P50/P90/P99** | Latency percentiles (success only) |
| **SLO Violation Rate** | Fraction of requests exceeding SLO (including failures) |
| **Error Rate** | Fraction of failed requests (timeout + other errors) |
| **Effective Cost** | Cost / successful requests |

### 5.3 Ablation Studies

| Ablation | Values | Purpose |
|----------|--------|---------|
| prior_strength λ | {10, 20, 50} | Shrinkage sensitivity |
| failure_mode | {INFINITY, SEPARATE} | Reliability handling |
| κ | {0, 5, 10} | Error penalty effect |

### 5.4 Expected Outputs

1. **Main figure**: Cost vs SLO violation rate for different SLOs
2. **Routing mix over time**: Show how π evolves with profile drift
3. **Violation rate comparison**: LP-Mix vs baselines at each SLO
4. **Ablation plots**: Sensitivity to λ, failure_mode, κ

---

## 6. Implementation Plan

### 6.1 File Structure

```
experiment/
├── strategies/
│   └── online_latency_router.py    # Main module
├── scripts/
│   └── run_phase3_simulation.py    # Evaluation script
└── results/
    └── latency_phase3/             # Output figures and CSV
```

### 6.2 Implementation Order

1. **ProviderProfile**: Time-based windows, mixed CDF, error tracking
2. **PreFilter**: Hard constraints (error rate, minimum CDF)
3. **LPSolver**: scipy.linprog wrapper with fallback
4. **SWRRSampler**: Smooth weighted round-robin with weight smoothing
5. **OnlineLatencyRouter**: Integrate all components
6. **Simulation harness**: Replay Phase 1 + BurstGPT

### 6.3 Testing Strategy

1. Unit tests for each component
2. Integration test: deterministic replay with fixed seed
3. Sanity checks:
   - LP gives at most 2 non-zero providers
   - SWRR distribution matches π over 1000 samples
   - Smoothing prevents weight jumps > 50%
   - Time causality: no future data leakage

---

## 7. Open Questions (Defer to Phase 4)

1. **Hedging**: When to send duplicate request to fastest provider?
2. **E2E metric**: How to handle long-output requests?
3. **Real OpenRouter evaluation**: Live traffic vs simulation
4. **Dynamic SLO**: User-specified per-request SLO

---

## Appendix A: Mathematical Formulation

### A.1 Mixed-Window CDF Estimator

```
F_hat(L) = β · F_short(L) + (1 - β) · F_long(L)

where:
    β = N_eff / (N_eff + λ)
    N_eff = number of samples in short window
    λ = prior_strength (pseudo-count from long window)
```

This is equivalent to empirical Bayes shrinkage:
- When N_eff is large: trust short window (β → 1)
- When N_eff is small: fall back to long window (β → 0)

### A.2 LP Formulation

```
minimize:   Σ_{j=1}^{n} π_j · c_j · (1 + κ · e_j)

subject to: Σ_{j=1}^{n} π_j · F_j(L) ≥ 1 - α        (tail constraint)
            Σ_{j=1}^{n} π_j = 1                      (probability)
            π_j ≥ 0                                  (non-negative)

where:
    π_j = routing probability for provider j
    c_j = cost per request for provider j
    e_j = error rate for provider j (used only if κ > 0)
    κ   = error penalty coefficient (default 0)
    F_j(L) = mixed-window CDF at SLO L
    α   = tail probability (0.01 for P99 constraint)
```

### A.3 SWRR Algorithm

```
Initialize: cw[j] = 0 for all j

For each request:
    1. For all j: cw[j] += w[j]
    2. Select j* = argmax_j cw[j]
    3. cw[j*] -= Σ_j w[j]
    4. Return j*
```

### A.4 Exponential Smoothing for Weight Updates

```
π_new = α · π_LP + (1 - α) · π_old

where α ∈ (0, 1) is the smoothing factor (default 0.3)
```
