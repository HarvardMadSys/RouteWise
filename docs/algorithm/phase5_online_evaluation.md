# Phase 5: Online Evaluation against OpenRouter

## Overview

Phase 5 validates our routing strategies (Phase 3 LP-mix, Phase 4 smart hedging) against OpenRouter's default routing using **real API calls** with **real billing**.

**Goal**: Prove that our routing achieves:
- Lower cost at same latency
- Lower latency at same cost
- Fewer SLO violations

**Key Constraint**: Minimize cost while producing statistically valid results.

---

## Design Principles

### 1. N-way Interleaving by Time (Not by Request)

**Problem**: If each trace request is sent to all N policies, cost multiplies by N (or more with hedging).

**Solution**: Each trace request is assigned to exactly ONE policy via round-robin.

```
Request 1 → openrouter_auto
Request 2 → lp_mix
Request 3 → smart_hedge
Request 4 → openrouter_auto
...
```

**Benefits**:
- Cost ≈ ×1 (not ×N)
- No "same request sent multiple times" artifacts
- Statistical comparison still valid with sufficient samples

**Sanity Check**: Verify balanced allocation:
```python
assert abs(count_A - count_B) / total < 0.05  # <5% imbalance
```

### 2. Trace Replay: Two Layers

| Layer | What | Cost | Realism |
|-------|------|------|---------|
| **P0: Arrival Pattern Only** | BurstGPT arrival times + fixed short prompt | Low | Medium |
| **P1: Full Trace** | Real prompts + variable lengths | High | High |

**P0 is sufficient for ICML**: Proves "under realistic arrival patterns, our routing reduces tail latency and cost."

### 3. Budget Guardrails

Before running, estimate cost bounds:

| Bound | Assumption |
|-------|------------|
| **Lower** | All requests → cheapest provider, no hedging |
| **Upper** | All requests → most expensive provider, always hedge |

**Hard constraints**:
- `--cost-cap`: Abort if exceeded
- `--max-requests`: Limit total requests
- `max_tokens=16`: Short responses only

### 4. Validity Criteria

A run is **valid** only if:
```python
for policy in policies:
    assert stats[policy]["estimated_cost_count"] == 0  # All costs are real
    assert stats[policy]["error_rate"] < 0.10  # <10% errors
```

---

## Policies to Compare

| Policy | Description | Provider Selection |
|--------|-------------|-------------------|
| `openrouter_auto` | OpenRouter default (baseline) | `provider=None` |
| `lp_mix` | Phase 3 LP-based routing | `OnlineLatencyRouter.route()` |
| `smart_hedge` | Phase 4 with hedging | LP primary + survival-based hedge |

**Note**: `cheapest_fixed` and `fastest_fixed` are optional (can add later for ablation).

---

## Implementation Plan

### Step 0: Budget Guardrails (P0)

**File**: Modify `phase5_online_evaluation.py`

```python
@dataclass
class BudgetConfig:
    max_requests: int = 500
    cost_cap_usd: float = 5.0
    max_tokens: int = 16
    policies: list[str] = ("openrouter_auto", "lp_mix", "smart_hedge")
```

**Cost Estimator**:
```python
def estimate_cost_bounds(n_requests: int, pricing: dict) -> tuple[float, float]:
    """Return (lower_bound, upper_bound) in USD."""
    cheapest = min(pricing.values())
    most_expensive = max(pricing.values())

    lower = n_requests * cheapest
    upper = n_requests * most_expensive * 2  # ×2 for potential hedging

    return lower, upper
```

### Step 1: Trace Loader (P0)

**File**: `experiment/scripts/trace_loader.py`

```python
@dataclass
class TraceRequest:
    arrival_time_sec: float  # Relative to trace start
    prompt: str = "Say hello in exactly 5 words."
    max_tokens: int = 16

def load_burstgpt_trace(
    trace_file: str,
    max_requests: int | None = None,
    time_limit_sec: float | None = None,
) -> list[TraceRequest]:
    """Load BurstGPT trace, extracting arrival times.

    P0: Use fixed prompt, only replay arrival pattern.
    P1: Extract actual prompts and lengths.
    """
    pass

def estimate_duration(trace: list[TraceRequest], speedup: float = 1.0) -> float:
    """Estimate wall-clock duration in seconds."""
    if not trace:
        return 0.0
    return (trace[-1].arrival_time_sec - trace[0].arrival_time_sec) / speedup
```

### Step 2: Replay Evaluator (P0)

**File**: Modify `phase5_online_evaluation.py`

```python
class TraceReplayEvaluator(Phase5OnlineEvaluator):
    """Evaluator that replays trace with N-way interleaving."""

    def run_trace_replay(
        self,
        trace: list[TraceRequest],
        policies: list[str],
        speedup: float = 1.0,
    ) -> dict[str, PolicyState]:
        """Replay trace with round-robin policy assignment.

        Args:
            trace: List of trace requests with arrival times.
            policies: Policies to interleave.
            speedup: Time compression factor (1.0 = real-time).

        Returns:
            Policy states with results.
        """
        start_time = time.time()
        trace_start = trace[0].arrival_time_sec

        for i, req in enumerate(trace):
            # Round-robin policy assignment
            policy = policies[i % len(policies)]

            # Wait until scheduled arrival time
            target_time = (req.arrival_time_sec - trace_start) / speedup
            elapsed = time.time() - start_time
            if target_time > elapsed:
                time.sleep(target_time - elapsed)

            # Check budget
            if self.total_cost > self.config.cost_cap_usd:
                print(f"Cost cap exceeded. Stopping at request {i}.")
                break

            # Execute request
            result = self._execute_policy_request(policy, time.time())
            # ... record result ...

        return self.policies
```

### Step 3: Smoke Test (P0)

**Duration**: 2-5 minutes
**Requests**: 30-50
**Purpose**: Verify before full run

**Checklist**:
- [ ] All `cost_is_estimated == False`
- [ ] `actual_provider` is not "unknown" for `openrouter_auto`
- [ ] Hedging triggers as expected (check `hedge_triggered` rate)
- [ ] No budget overrun

```bash
python phase5_online_evaluation.py \
  --trace experiment/data/burstgpt_sample.json \
  --max-requests 50 \
  --cost-cap 1.0 \
  --policies openrouter_auto lp_mix smart_hedge
```

### Step 4: Full Run (P1)

**Duration**: 30-60 minutes
**Requests**: 300-500
**Budget**: $3-5

```bash
python phase5_online_evaluation.py \
  --trace experiment/data/burstgpt_trace.json \
  --max-requests 500 \
  --cost-cap 5.0 \
  --warmup 900 \
  --policies openrouter_auto lp_mix smart_hedge \
  --output experiment/results/phase5_online
```

---

## Output & Deliverables

### CSV Log Schema

```
timestamp, request_id, policy, selected_provider, actual_provider,
ttft_ms, e2e_ms, status, slo_violated, cost_usd, cost_is_estimated,
hedge_triggered, hedge_provider, hedge_winner, prompt_tokens, completion_tokens
```

### Summary Statistics

Per policy:
- `total_requests`, `successful_requests`, `error_rate`
- `slo_violation_rate`
- `total_cost`, `avg_cost`
- `p50_ms`, `p90_ms`, `p99_ms`
- `provider_distribution`
- `hedge_trigger_rate` (for smart_hedge)

### Figures (3 total)

**Figure 1: Cost vs P99 Scatter (Main Result)**
- X: Average cost per request ($/1000 requests)
- Y: P99 TTFT (ms)
- Points: Each policy
- Annotation: Improvement margin

**Figure 2: Margin Table**

| Policy | Cost | P99 | SLO Viol% | vs Baseline |
|--------|------|-----|-----------|-------------|
| openrouter_auto | $X | Yms | Z% | — |
| lp_mix | $X' | Y'ms | Z'% | -A% cost, -B% P99 |
| smart_hedge | $X'' | Y''ms | Z''% | -C% cost, -D% viol |

**Figure 3: Provider Distribution / Hedge Rate (Mechanism Sanity)**
- Bar chart: Provider selection % by policy
- For smart_hedge: Hedge trigger rate, winner distribution

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Cost overrun | `--cost-cap`, `--max-requests`, cost estimation |
| OpenRouter doesn't return provider | Log as "unknown", exclude from provider analysis |
| OpenRouter doesn't return cost | `cost_is_estimated=True`, invalidate run if >0 |
| High error rate | Backoff + skip policy, require <10% error |
| Unbalanced interleaving | Assert <5% imbalance in post-analysis |

---

## Timeline

| Step | Duration | Output |
|------|----------|--------|
| Step 0: Budget guardrails | 0.5 day | Code changes |
| Step 1: Trace loader | 0.5 day | `trace_loader.py` |
| Step 2: Replay evaluator | 1 day | Modified `phase5_online_evaluation.py` |
| Step 3: Smoke test | 0.5 day | Validation |
| Step 4: Full run | 1-2 hours | Results + figures |
| Step 5: Analysis | 0.5 day | Paper figures |

**Total**: ~3 days implementation + 1 day experiments

---

## Appendix: BurstGPT Trace Format

Expected input format (JSON lines):
```json
{"timestamp": 0.0, "prompt_tokens": 50, "completion_tokens": 100}
{"timestamp": 0.5, "prompt_tokens": 30, "completion_tokens": 50}
...
```

For P0, we only use `timestamp`. For P1, we use token counts to generate synthetic prompts of appropriate length.
