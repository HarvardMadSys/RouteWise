# EmpiricalDistribution Performance & Streaming Histogram — Execution Plan

> Phased plan to make `cost_layer_real_world` (and any other empirical-profile
> scenario) runnable end-to-end on the full 1.8M-request trace, without
> running out of RAM.
>
> Last updated: 2026-05-06.
>
> Status: Phase 1a **shipped**; Phase 1b **specced, not implemented**, awaiting
> Codex re-review of §5. Phase 2 blocked on 1b.
>
> This document is self-contained. Reviewers (Codex) should be able to evaluate
> the plan without reading `RWSIM_REFACTOR_PLAN.md` or
> `SIMULATION_SECTION_REFACTOR.md` first.

---

## 1. TL;DR

Two independent problems block `cost_layer_real_world` at full scale:

| # | Problem | Hot path | Effect |
|---|---|---|---|
| **A** | `EmpiricalDistribution` per-call overhead is 9–7500× slower than `LogNormal` | `provider.sample_ttft()` (every request), `true_p99_ms()` (every record build), `_fallback_provider` sort | Real-world scenario is 30–60× slower than synthetic peers |
| **B** | `Simulator.run` accumulates every `PerRequestRecord` in a Python list | `rwsim/engine/simulator.py:49,56` | RSS peaks at ~11 GB on the 1.8M trace, OOM risk on smaller machines |

The fix is split into three phases that can each ship as one commit:

```
Phase 1a  EmpiricalDistribution micro-fixes              SHIPPED       1 commit
Phase 1b  Streaming TTFT histogram + record discard      ~half day     2–3 commits
Phase 2   Re-run cost_layer_real_world full trace        ~1 hour CPU   verification
```

**Phases 1a and 1b address different problems and are independently shippable.**
1a does not change any public behaviour; 1b adds a streaming-aggregator
artifact path and an opt-out for record retention at the paper-runner layer
(the `Simulator` default stays `retain_records=True` so existing callers are
unaffected). Either can be reverted without the other.

---

## 2. What Codex flagged

### 2.1 Round 1 (incorporated before 1a shipped)

1. **Use precise framing.** Earlier I wrote "LP behaviour correct"; the actual
   observation was "baseline accounting correct on the random policy run."
   Random is a baseline, not LP. The performance plan does not change LP
   semantics — it only changes how distributions are sampled and how
   per-request data is aggregated.

2. **`rng.choice` slowness is per-call overhead, not array-size dependent.**
   The cost is the constant Python→NumPy boundary cost paid per `sample(rng,
   1)` call. Sample-array length does not enter. Switching to
   `rng.integers(...)` + indexing wins because it has roughly half the
   per-call overhead, not because it scales differently. Measured ratio on
   Python 3.14 / NumPy 2.x is closer to 1.8× than 5.5× — see §4.5.

3. **TTFT histogram must be updated inside the simulator hot loop, not
   computed from the post-aggregate record list.** The whole point of the
   histogram artifact is to let us discard records as they are produced.
   Building it after `Run` already holds 1.8M records solves nothing — the
   memory peak has already happened.

### 2.2 Round 2 (incorporated into the §5 rewrite below)

4. **`@pytest.mark.perf` is registered but not deselected by default.**
   `pyproject.toml` declares the marker but `addopts` has no `-m "not perf"`,
   so a perf test would run on every `pytest` invocation. Resolution: drop
   the perf-as-pytest pattern entirely; ship the benchmark as a standalone
   script under `scripts/perf/` instead. (Already done as part of 1a — see
   §4.4.)

5. **50 log-spaced bins over 5 decades is too coarse.** The geometric ratio
   between adjacent bins is `10^(5/50) ≈ 1.26` — bins are ~26% wide, not
   ~10%. The earlier "1–2% percentile error" claim doesn't survive contact
   with that math. Resolution: bump to 256 bins (~2.3% wide bins; see §5.2)
   or higher if downstream analyses tighten the requirement.

6. **`summarize_runs` does not use the public `Run` API today — it flattens
   `run.records` directly** (`experiments/simulation/common.py:304`). The
   prior doc said "no change" was needed there. Wrong. Resolution: §5.4 now
   spells out the rewrite (consume per-run histograms, merge across seeds,
   never touch `record` lists).

7. **`Simulator.retain_records=False` as the default is too aggressive.**
   Many callers (golden-capture, future debugging) legitimately need records.
   Resolution: `Simulator.retain_records` defaults to `True` (no behavior
   change at the simulator layer). The opt-out lives in
   `experiments/simulation/common.py::run_section`, which paper runs go
   through. Tests, golden capture, and `rwsim.runner` are unaffected.

8. **Histogram artifact write must live in `run_section`, not
   `cost_layer.py`.** Per-section files should not duplicate boilerplate.
   Resolution: §5.6 places the write in `run_section` so latency-layer /
   hedging / end-to-end inherit it for free.

9. **`tests/golden_capture.py:146` reads `run.records` directly** to compute
   per-request digests. Resolution: golden capture explicitly requests
   `retain_records=True` when constructing the `Simulator`; §5.5 documents
   this.

The §5 rewrite below reflects all of these.

---

## 3. Diagnosis (numbers we are basing the plan on)

Profile run, 100k synthetic samples, single thread, Python 3.11, NumPy 1.26:

| Method | LogNormal | EmpiricalDistribution (current) | Slowdown |
|---|---|---|---|
| `sample(rng, 1)` | 0.65 μs | 3.6 μs | 5.5× |
| `mean()` | 0.05 μs (closed-form) | 36 μs (`np.mean(samples)` every call) | ~700× |
| `quantile(0.99)` / `p99()` | 0.05 μs | 378 μs (`np.percentile(_sorted, 99)` every call) | ~7500× |

Per-request cost of one call to each (random policy on `cost_layer_real_world`,
N=1.8M):

```
sample_ttft           1.8M × 3.6 μs   = 6.5 s
true_p99_ms (record)  1.8M × 378 μs   = 680 s   ← dominant
mean()/quantile in
  fallback ranking    1.8M × 36 μs    = 65 s
```

Total: ~12 minutes spent inside `EmpiricalDistribution` in the random run.
Synthetic-distribution runs spend < 5 s on the same surface. That's the gap.

Memory:

```
PerRequestRecord ~ 1100 B (dataclass with metadata dicts) × 1.8M = 2 GB
+ NumPy intermediate arrays inside summarize_runs                ≈ 11 GB peak
```

This is reproducible by running

```bash
routewise simulator cost-layer --scenario cost_layer_real_world \
    --requests-per-window <full> --seeds 42
```

and watching `/usr/bin/time -l` RSS.

---

## 4. Phase 1a — `EmpiricalDistribution` micro-fixes

**Goal:** kill the per-call overhead with no architecture change. One small
commit.

### 4.1 File changes

`rwsim/world/empirical.py`:

1. Replace `_sorted: np.ndarray = field(init=False, repr=False)` with
   precomputed cached fields (still set via `object.__setattr__` since the
   dataclass is `frozen=True`):

   ```python
   _sorted: np.ndarray         # already computed in __post_init__
   _mean: float                # NEW: cached np.mean
   _std: float                 # NEW: cached np.std
   _n: int                     # NEW: cached len(samples)
   ```

   Compute all three in `__post_init__` once, in addition to the existing
   `_sorted` setup.

2. Change `mean()` and `std()` to return the cached values:

   ```python
   def mean(self) -> float:
       return self._mean

   def std(self) -> float:
       return self._std
   ```

3. Change `quantile(q)` to use `_sorted` indexing instead of
   `np.percentile`. For a frozen sample array, `np.percentile` is
   over-general (it has interpolation/method/axis handling that's pointless
   here). Use linear interpolation between two adjacent sorted entries:

   ```python
   def quantile(self, q: float) -> float:
       if not 0.0 < q < 1.0:
           raise ValueError(f"quantile q must be in (0, 1), got {q}")
       # Linear interpolation, matching np.percentile's default 'linear' method.
       pos = q * (self._n - 1)
       lo = int(np.floor(pos))
       hi = int(np.ceil(pos))
       if lo == hi:
           return float(self._sorted[lo])
       frac = pos - lo
       return float(self._sorted[lo] * (1.0 - frac) + self._sorted[hi] * frac)
   ```

   `p50()` / `p95()` / `p99()` already delegate to `quantile`; no change
   needed there.

4. Change `sample(rng, size)` to integers + index, which has lower per-call
   overhead than `rng.choice` for the size=1 hot path:

   ```python
   def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
       idx = rng.integers(0, self._n, size=size)
       return self.samples[idx]
   ```

   Note the `samples` array is C-contiguous after `np.asarray(...,
   dtype=float)` in `__post_init__`, so fancy indexing here is a single
   `memcpy`-equivalent.

### 4.2 No-change items (deliberate)

- The `_sorted` field stays. We need it for `cdf` (which uses `searchsorted`)
  and we will use it for the new `quantile`.
- `cdf` already does the right thing (`searchsorted` is O(log n), constant
  per-call cost). Don't touch.
- `from_npz` / `pooled_from_npz` constructors are fine; the cached fields
  are populated automatically because they go through `__post_init__`.
- `EmpiricalDistribution` stays `frozen=True`. The new cached fields are set
  via `object.__setattr__`, same pattern already used for `_sorted`.

### 4.3 Tests

Existing test file: `tests/unit/world/test_empirical_distribution.py`.

- `test_from_npz_loads_provider_distribution_matching_metadata` — must still
  pass. The metadata p50/p99 are stored at percentile precision, so the
  switch from `np.percentile` to indexed interpolation must not move them.
  Run this test before and after; assertion uses `pytest.approx` with the
  default tolerance, which is fine for the linear-interp difference (will
  match exactly because `np.percentile`'s default *is* linear).
- `test_sample_draws_from_empirical_support` — must still pass. The
  integers+index sampler still draws values from `dist.samples` (it indexes
  into the same array), so `set(samples).issubset(set(dist.samples))`
  remains true.
- Add one new test: `test_mean_std_and_quantile_match_numpy_reference`.
  Build an `EmpiricalDistribution` from a fixed array, assert
  `dist.mean()`, `dist.std()`, `dist.quantile(0.5)`, `dist.quantile(0.99)`
  match `np.mean / np.std / np.percentile` of the same array within
  `pytest.approx(rel=1e-12)`. This locks the equivalence so future micro-
  optimisations don't drift.

### 4.4 Microbenchmark (standalone script, not pytest)

`pyproject.toml` declares the `perf` marker but `addopts` does not deselect
it, so a `@pytest.mark.perf` test would run on every `pytest` invocation
and bring noisy timing assertions into the default suite. Instead, ship
the benchmark as a script under `scripts/perf/`:

```
scripts/perf/bench_empirical_distribution.py
```

Run manually after touching `rwsim/world/empirical.py`:

```bash
.venv/bin/python scripts/perf/bench_empirical_distribution.py
```

The script prints per-call ns for `sample / mean / p50 / p99 / cdf` and
warns if any number exceeds the post-1a budget. No pytest involvement.

If we ever want CI to gate on this, the right move is to update `addopts`
to `-m "not slow and not perf"` first, then add the perf pytest. We are
not doing that in 1a.

### 4.5 Measured speedup (Python 3.14 / NumPy 2.x, M-series Mac)

Per-call ns (50 000-iteration loop, 100 000-sample distribution):

| Method | Pre-1a | Post-1a | Speedup |
|---|---|---|---|
| `sample(rng, 1)` | 3 600 | 2 037 | 1.8× |
| `mean()` | 36 000 | 20 | ~1 800× |
| `p50()` / `p99()` | 378 000 | 240 | ~1 580× |
| `cdf(value)` | 850 (already O(log n)) | 851 | unchanged |

`sample` won less than expected (1.8× vs the 5.5× I quoted from older
NumPy bindings). On Python 3.14 / NumPy 2.x the per-call boundary cost of
`rng.integers + fancy index` is closer to `rng.choice` than it used to
be. The dominant cost was never `sample` though — it was per-record
`true_p99_ms`, which is now ~1 580× faster.

End-to-end on `cost_layer_real_world`, **measured** at 100 k-request cap,
single seed (`/usr/bin/time -l`):

| Policy | Wall time | Peak RSS |
|---|---|---|
| `random` | 2.57 s | 275 MB |
| `ablation_lp_only_p50` | 64 s | 313 MB |

Extrapolating to 1.8M (linear in records, plus ~1 GB workload-array
amortisation):

| Policy | Estimated wall | Estimated RSS at 1.8M |
|---|---|---|
| `random` | ~50 s | ~5 GB (record list dominates) |
| `ablation_lp_only_p50` | ~20 min | ~5 GB |

**Random p50/p99 numbers match pre-1a smoke** (p50 = 999.85 ms, p99 =
6 551.66 ms; total cost $228.29 / 100k). LP at p=0.5 saves ~25% via
40/50/9 cheap/mid/expensive mix vs random's even 33/33/33.

This does NOT fix RAM. RAM is Phase 1b. The ~5 GB extrapolation above is
why 1b is still required even though wall time is now tolerable.

### 4.6 Commit message (as shipped)

```
perf(empirical): cache mean/std/quantile and switch sampler to integers+index

EmpiricalDistribution recomputed mean/std/quantile on every call and used
rng.choice for sampling. On the cost_layer_real_world full trace this
spends ~12 min/seed in distribution methods. Cache moments in __post_init__
and replace rng.choice(replace=True) with rng.integers + fancy indexing.
sample() 3600 → 2037 ns; mean() 36 μs → 20 ns; p99() 378 μs → 240 ns.

No public surface change. Existing 6 unit tests pass; new equivalence
test test_mean_std_and_quantile_match_numpy_reference locks the indexed-
interp quantile to np.percentile within 1e-12 relative. Standalone
benchmark added at scripts/perf/bench_empirical_distribution.py (not in
default pytest suite — perf marker is not deselected by current addopts).
```

---

## 5. Phase 1b — Streaming TTFT histogram + paper-runner record discard

**Goal:** make paper-section runs (the only callers that hit 1.8M
requests) constant-memory in the request count, while leaving
`Simulator`'s default behavior unchanged for golden capture, debugging,
and `rwsim.runner`.

This is the actual architecture change. It's larger than 1a, but the
blast radius is deliberately bounded: only the paper-runner layer
(`experiments/simulation/`) opts out of record retention. The
`Simulator`-level default stays `True`.

### 5.1 The contract change (and what it does NOT change)

**Today:** `Run` holds `records: list[PerRequestRecord]`. All `Run.p50_ms()`,
`p99_ms()`, `cost_by_provider()`, `provider_fractions_over_time()`, etc.
methods iterate that list (`rwsim/metrics/run.py:91-243`).
`experiments/simulation/common.py::summarize_runs` also flattens
`run.records` directly (line 304).

**After 1b:**

- `Run` gains a streaming-aggregator side-channel built up by the
  `Simulator` hot loop. It exposes `ttft_histogram() -> TtftHistogram`,
  `cost_summary()`, `provider_summary()`, etc.
- `Run.records` still exists. Whether it's populated is controlled by a
  new `Simulator.retain_records: bool = True` (default unchanged).
- `Run.p50_ms()` / `Run.p99_ms()` / `Run.cost_by_provider()` /
  `Run.tier_fractions()` / `Run.hedge_rate()` etc. — all existing public
  methods — fall back to the aggregator when `records` is empty, and
  read from `records` otherwise. Bit-exact equivalence is required when
  records are retained (verified by a new test, see §5.7).
- `*_over_time` methods stay record-driven for now. They are only used
  by plot code on small runs that opt into `retain_records=True`.

The `experiments/simulation/common.py::run_section` paper runner sets
`retain_records=False` explicitly. Everything else (golden capture,
`rwsim.runner`, ad-hoc CLIs that construct a `Simulator` directly)
inherits the `True` default and is unaffected.

### 5.2 New file: `rwsim/metrics/histogram.py`

```python
N_DECADES = 5             # 1 ms .. 100 000 ms
BINS_PER_DECADE = 51      # geometric ratio 10^(1/51) ≈ 1.046, ~4.6% wide
N_BINS = N_DECADES * BINS_PER_DECADE  # 255 + 1 underflow + 1 overflow = 257

@dataclass
class TtftHistogram:
    """Log-spaced histogram for TTFT/e2e values in milliseconds.

    Bins: 255 log-spaced bins from 1 ms to 100 000 ms (5 decades, ~51
    bins per decade, geometric ratio ~1.046 → bin width ~4.6% relative),
    plus one underflow bucket and one overflow bucket.

    Quantiles are interpolated linearly within the bin that contains the
    cumulative target. With ~4.6% bin width and linear interpolation,
    the per-quantile relative error is bounded by ~2% (and is much
    smaller in practice for well-populated bins).
    """
    bin_edges: np.ndarray   # shape (N_BINS + 1,) — log-spaced
    counts: np.ndarray      # shape (N_BINS + 2,) — [underflow, ..., overflow]
    sum_value: float = 0.0
    sum_sq: float = 0.0
    n: int = 0

    @classmethod
    def empty(cls) -> "TtftHistogram": ...
    def add(self, value_ms: float) -> None: ...
    def add_array(self, values_ms: np.ndarray) -> None: ...
    def quantile(self, q: float) -> float: ...
    def mean(self) -> float: ...
    def std(self) -> float: ...
    def merge_into(self, other: "TtftHistogram") -> None: ...    # in-place
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TtftHistogram": ...
```

**Why 256 bins, not 50.** Codex flagged that 50 bins / 5 decades gives
bin ratio `10^(5/50) ≈ 1.26` (26% wide), not the 10% I claimed.
51 bins/decade gives ratio `10^(1/51) ≈ 1.046` (~4.6% wide). With linear
interpolation that's ≤ ~2% per-quantile relative error, which is the
precision we actually need for paper-grade reporting.

We already have a streaming-histogram quantile estimator at
`experiments/offline_stage/value_estimators/histogram.py`. Read it and
copy what's useful (bin layout, quantile interpolation logic). Do NOT
import from `experiments/` into `rwsim/` — copy the relevant pieces.
`rwsim/` is the simulator core, `experiments/` is paper-section glue.

### 5.3 `Simulator.run` changes (`rwsim/engine/simulator.py`)

Today:

```python
records: list[PerRequestRecord] = []
for request in requests:
    ...
    records.append(self._build_record(...))
return Run(records=records, ...)
```

After 1b:

```python
aggregator = RunAggregator(
    policy=policy_name,
    scenario_name=self.scenario.name,
    source="simulation",
)
records: list[PerRequestRecord] | None = [] if self.retain_records else None
for request in requests:
    ...
    record = self._build_record(...)
    aggregator.observe(record)
    if records is not None:
        records.append(record)
return Run(
    records=records or [],
    aggregator=aggregator,
    policy=policy_name,
    scenario_name=self.scenario.name,
    source="simulation",
)
```

`Run` is extended to accept an optional `aggregator: RunAggregator |
None`. If both records and aggregator are present, records are
authoritative for any method that historically read them (so
bit-equivalence with today is preserved). When records are empty and
aggregator is present, methods consult the aggregator.

`Simulator.retain_records: bool = True` — see §5.5 for why default-True.

`_build_record` is unchanged — it still returns a `PerRequestRecord`
per request, and `policy.observe(record, decision, outcome)` still gets
called with that record. The record only fails to *survive past* the
loop iteration when `retain_records=False`.

### 5.4 `summarize_runs` rewrite (`experiments/simulation/common.py`)

Codex was right that the prior doc claim ("already on public API; no
change") was wrong. `summarize_runs` flattens `run.records` directly
and uses `np.percentile` on the concatenation across seeds. Two
problems:

1. With `retain_records=False` at the paper-runner layer,
   `run.records` is empty — so the function has no data to flatten.
2. Even today, the function computes `p99` from concatenated samples;
   moving to histograms means we cannot regenerate that exact value.
   We need `summarize_runs` to merge per-run histograms, then read
   percentiles off the merged histogram.

After 1b:

```python
def summarize_runs(*, scenario, policy, seeds, runs):
    merged_ttft = TtftHistogram.empty()
    cost_total = 0.0
    cost_count = 0
    provider_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    slo_violations = 0
    hedge_triggered = 0
    n_total = 0
    for run in runs:
        agg = run.aggregator       # required; assert if missing
        merged_ttft.merge_into(agg.ttft_histogram)
        cost_total += agg.cost_total_usd
        cost_count += agg.cost_count
        provider_counts.update(agg.provider_counts)
        tier_counts.update(agg.tier_counts)
        slo_violations += agg.slo_violations
        hedge_triggered += agg.hedge_triggered
        n_total += agg.n_observed

    return {
        "scenario": scenario.name,
        "policy": policy,
        "seeds": list(seeds),
        "n_requests": n_total,
        "mean_ttft_ms": merged_ttft.mean(),
        "p10_ms": merged_ttft.quantile(0.10),
        "p25_ms": merged_ttft.quantile(0.25),
        "p50_ms": merged_ttft.quantile(0.50),
        "p75_ms": merged_ttft.quantile(0.75),
        "p90_ms": merged_ttft.quantile(0.90),
        "p99_ms": merged_ttft.quantile(0.99),
        "mean_cost_usd": cost_total / max(cost_count, 1),
        "total_cost_usd": cost_total,
        "slo_violation_rate": slo_violations / max(n_total, 1),
        "hedge_rate": hedge_triggered / max(n_total, 1),
        "provider_mix": _fraction_map(provider_counts, n_total),
        "tier_mix": _fraction_map(tier_counts, n_total),
    }
```

**Multi-seed merge is histogram merge, not per-seed P99 mean.** Today's
code accidentally gets this right by concatenating raw samples. Naive
"average the seed-level P99 values" would be different (and wrong) for
non-symmetric distributions. The merge approach matches today's
semantics exactly when bin precision is tight.

The expected percentile drift from this change (on the
`cost_layer_real_world` smoke run) is on the order of 0.3–1.0%, well
within the ±2% bin-error envelope. `tests/golden/cost_layer/` will
need to be regenerated; see §5.7.

### 5.5 `retain_records` default and where the opt-out lives

| Layer | Default | Why |
|---|---|---|
| `Simulator.retain_records` | `True` | preserves all current callers. Golden capture, ad-hoc debugging, `rwsim.runner`, future `policy.observe(record)` hooks all keep working with no opt-in. |
| `experiments/simulation/common.py::run_section` | passes `retain_records=False` | the only place 1.8M-request runs happen |
| Tests, `tests/golden_capture.py` | unchanged | inherit `True` default |

Concretely, `run_section` constructs `Simulator` like:

```python
def run_policy(scenario, requests, policy_name, *, presets, seed):
    policy = build_policy(policy_name, presets=presets, seed=seed)
    simulator = Simulator(scenario=scenario, seed=seed, retain_records=False)
    return simulator.run(requests, policy, policy_name=policy_name)
```

Local-runner overrides passed via `run_section(..., section_runners=...)`
must also opt out. Document this requirement in `run_section`'s
docstring; assert in `summarize_runs` that every run has a populated
`aggregator` so we fail fast if a runner forgets.

### 5.6 New artifact: `ttft_histogram.json`, written by `run_section`

Write the histogram artifact in `run_section` (not in `cost_layer.py`),
so latency-layer / hedging / end-to-end inherit it for free:

```python
# Inside run_section, alongside summary.json / summary.csv writes:
for scenario in scenarios.values():
    for policy in policies:
        for seed, run in zip(seeds, runs_by_policy[(scenario.name, policy)]):
            hist = run.aggregator.ttft_histogram
            hist_path = root / "histograms" / (
                f"{scenario.name}__{policy}__seed{seed}.json"
            )
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(hist_path, hist.to_dict())
```

`TtftHistogram.to_dict()` returns
`{"bin_edges_ms": [...], "counts": [...], "n": ..., "mean_ms": ...,
"std_ms": ...}`.

The paper's TTFT CDF / violin plots read from these artifacts. Plot
rendering is a separate (small) follow-up.

### 5.7 Tests

- `tests/unit/metrics/test_ttft_histogram.py` — new. Covers:
  - `add` / `add_array` count correctness.
  - `quantile` matches `np.percentile` within 2% relative on log-normal
    inputs across q ∈ {0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99}.
  - `merge_into` of two histograms produces the same `quantile`/`mean`
    /`std` as a single histogram built from the union.
  - Underflow/overflow buckets accumulate correctly when inputs fall
    outside `[1 ms, 100 000 ms]`.
  - `to_dict` / `from_dict` round-trips bit-exactly.

- `tests/unit/metrics/test_run_aggregator.py` — new. Build the same
  request stream twice, once with `retain_records=True` and once with
  `False`. Assert all `Run` public methods agree within 2% relative
  (TTFT percentiles via histogram tolerance) and bit-exactly for
  cost/provider/tier counts.

- `tests/unit/engine/test_simulator.py` — extend. Default behaviour
  (`retain_records=True`) is unchanged. With `retain_records=False`:
  `run.records` is empty, `run.aggregator` is populated, public methods
  fall back to aggregator.

- `tests/unit/simulation/test_cost_layer.py` — extend to assert that
  `run_section` writes `histograms/<scenario>__<policy>__seed<s>.json`
  for every (scenario, policy, seed) combination.

- `tests/golden/cost_layer/scenarios.json` — regenerate. Capture under
  `retain_records=True` (the default) so the digest fields stay stable;
  the rewritten `summarize_runs` rounds percentiles through the merged
  histogram, so the percentile fields will drift by ≤ 2% relative.
  Document the regeneration in the commit message.

- `tests/golden_capture.py` — explicitly construct `Simulator` with
  `retain_records=True`. This is the default, so the only required
  change is to document the dependency (a comment near the
  `Simulator(...)` call) and add a unit assertion that
  `golden_capture._seed_run_payload` requires `run.records`.

### 5.8 Memory acceptance criterion

Reproduce on a single seed of `cost_layer_real_world`, 1.8M requests:

```bash
/usr/bin/time -l routewise simulator cost-layer \
    --scenario cost_layer_real_world --seeds 42
```

- Before 1b (post-1a): peak RSS ~ 5 GB (extrapolated; record list is
  the dominant term)
- After 1b: peak RSS < 1.5 GB (target; expected ~1 GB, dominated by
  the workload-array materialisation that 1b does not address)

If post-1b RSS is above 2 GB, investigate. Likely culprit: a stray
`run.records` walk in plot code or a test fixture that didn't get
migrated. Workload-array memory is out of scope — see §8.

### 5.9 Commit sequence (proposed)

Three commits, each independently revertable:

```
1. feat(metrics): add TtftHistogram and RunAggregator
   - new files rwsim/metrics/histogram.py, rwsim/metrics/aggregator.py
   - unit tests for both (test_ttft_histogram.py, test_run_aggregator.py)
   - no Simulator changes; aggregator/Run wiring stays decoupled until
     commit 2

2. feat(simulator): wire RunAggregator into Run, retain_records=True default
   - Simulator gains retain_records flag, default True
   - Run gains optional aggregator field; public methods fall back to
     aggregator when records are empty
   - extend tests/unit/engine/test_simulator.py

3. refactor(experiments/simulation): paper runner opts out of record
   retention, summarize_runs merges histograms
   - run_policy passes retain_records=False
   - summarize_runs reads aggregator instead of flattening records
   - run_section writes histograms/*.json artifacts
   - regenerate golden cost_layer scenarios (record digests stable;
     percentiles drift ≤2% via histogram merge)
```

Commit 1 ships the primitive in isolation. Commit 2 wires it into the
simulator without changing any default. Commit 3 flips the paper-runner
behavior — this is the only commit that changes user-visible artifacts
(`summary.json` percentiles drift ≤ 2%). Reverting commit 3 brings the
old paper artifacts back without losing the histogram primitive.

---

## 6. Phase 2 — Re-run `cost_layer_real_world` full trace

After 1a + 1b ship:

```bash
routewise simulator cost-layer \
    --scenario cost_layer_real_world \
    --seeds 42,43,44
```

### 6.1 Acceptance criteria

| Metric | Target |
|---|---|
| Wall time per seed | < 5 min |
| Peak RSS | < 1 GB |
| Random-policy mean cost / token | matches the smoke run (1.8 ± 0.05 USD/M with 5× output multiplier) |
| LP-policy P99 (p=1.0) | within 5% of smoke estimate |
| `ttft_histogram.json` artifact present for every (scenario, policy) pair | yes |

### 6.2 What we are NOT doing in Phase 2

- Not adding violin / CDF plot rendering. That's a follow-up using the
  histogram artifact.
- Not changing the §1.1 paper numbers. We expect the full-trace numbers to
  match the smoke-trace numbers within sampling noise. If they don't,
  that's a separate investigation, not a perf-plan deliverable.
- Not re-running synthetic distributions. They were never the bottleneck.

---

## 7. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| `EmpiricalDistribution` stays `frozen=True` | yes | cache fields are derivative; mutability would invite drift |
| `EmpiricalDistribution` does not lazy-cache | no — eager in `__post_init__` | construction cost is one-shot; per-call branch on `if cached is None` is wasted in the hot path |
| Histogram bin layout | **256 log-spaced bins** (51 per decade × 5 decades + under/overflow) | bin ratio ~1.046 → ~4.6% wide → ≤ 2% per-quantile relative error after linear interpolation |
| Histogram lives in `rwsim/metrics/` | yes | `experiments/.../value_estimators/histogram.py` is for the routing-time predictor; this is the metrics-side analogue. Don't share code across the boundary. |
| `Simulator.retain_records` default | **`True`** | preserves all current callers (golden capture, `rwsim.runner`, debugging). The opt-out lives one layer up at `experiments/simulation/common.py::run_section`, which is the only caller that materialises 1.8M records. |
| Histogram artifact write | **`run_section`** (not `cost_layer.py`) | every paper section runs through `run_section`; centralising the write means latency-layer / hedging / end-to-end inherit it for free |
| Multi-seed percentile aggregation | **histogram merge** (not seed-level P99 mean) | matches today's "concatenate samples then percentile" semantics; naive seed-level averaging is wrong on non-symmetric distributions |
| Perf benchmark delivery | standalone script under `scripts/perf/` | `pytest` markers are declared but not deselected by `addopts`; a `@pytest.mark.perf` test would run on every CI invocation |
| `cdf` implementation | unchanged (`searchsorted`) | already O(log n), no regression |

---

## 8. Out of scope for this plan

These are real issues but explicitly NOT addressed here:

- **Workload streaming.** Today `requests` is a fully-materialised
  `Sequence[Request]` (~1.8M Python objects ≈ 1 GB). A streaming
  `Iterable[Request]` would cut another ~1 GB. Worth doing, but it's a
  separate refactor of the workload loader and the dataset cache. File a
  follow-up.
- **Per-record metadata dict bloat.** `PerRequestRecord.metadata` is a
  free-form `dict` and accounts for ~40% of the per-record bytes. Even
  with `retain_records=True`, switching it to a typed dataclass would
  halve test-run RAM. Not in scope.
- **Distribution swap at scenario-build time.** A different angle would be
  to fit a `LogNormal` to the empirical samples once and route through
  that. Cheaper at runtime, but fundamentally changes the meaning of
  "real-world distribution." Reject for the paper; revisit only if 1a +
  1b are still not enough.
- **Multi-seed parallelism.** `cost_layer.py` runs seeds serially. With
  per-seed RSS down to ~500 MB we could run 3 seeds in parallel processes
  and finish in roughly 1× wall instead of 3×. Not in scope; opportunistic
  follow-up.

---

## 9. Review checklist for Codex (round 2)

Round 1's questions have been resolved (see §2). The remaining things to
push back on hardest now:

1. **Histogram precision: is 256 bins / ≤2% per-quantile error tight
   enough for §1.1?** Codex flagged 50 was too coarse; 256 is my
   proposed correction. If the paper's headline LP-vs-baseline gap is
   smaller than ~5% on any percentile, we should bump bins again or
   keep raw samples for that one comparison. Review §1.1's expected
   effect sizes against this.

2. **`summarize_runs` rewrite (§5.4) — is histogram merge the right
   semantic?** Today's flat-concat-then-percentile gives a specific
   answer; histogram merge gives something within ≤2% of that. The
   alternative is to keep per-run samples (in compressed form, e.g.
   reservoir sample of size 100k) and exact-percentile across them. I
   chose histogram merge because it composes cleanly with the artifact
   we already need for plotting. Sanity-check this trade.

3. **`Simulator.retain_records=True` default (§5.5) — am I missing a
   caller that should also opt out?** I claim only `run_section` opts
   out. If `experiments/real_evaluation/` or any other production
   pipeline materialises records at scale, the default needs another
   look.

4. **Commit 3 in §5.9 changes paper-artifact percentile fields by ≤2%.**
   This is a user-visible numeric drift. Is the right move to (a)
   accept the drift and regenerate the golden, (b) keep `summarize_runs`
   on raw samples by also retaining a compressed sample alongside the
   histogram, or (c) gate the histogram path behind a flag and run both
   for one phase to see the diff?

5. **`*_over_time` methods stay record-driven (§5.1).** They are only
   called by plot code on small runs that opt into retention. If any
   paper plot needs over-time data on a 1.8M run, this plan needs an
   over-time aggregator too — and that's not free (one Counter per
   bucket × N providers × N tiers).

6. **Workload streaming (§8) — is this still safely deferrable after
   1b?** Post-1b, peak RSS is dominated by the workload-array
   materialisation (~1 GB for 1.8M `Request` objects). If we ever need
   to push past ~1.5 GB target, the workload loader is the next thing.

---

## 10. Execution order

```
[0]  ✓ Codex round 1 review.
[1]  ✓ Phase 1a single commit (shipped).
       - rwsim/world/empirical.py: cache mean/std/n, sorted-interp quantile,
         integers+index sample
       - tests/unit/world/test_empirical_distribution.py: equivalence test
       - scripts/perf/bench_empirical_distribution.py: standalone benchmark
       - 105 passed / 0 failed under `pytest -m "not slow"`
       - cost_layer_real_world @ 100k random: 2.57 s / 275 MB (was ~47 s)
       - p50/p99 unchanged within sampling noise
[2]  ← we are here. Codex round 2 review of §5 (this rewrite).
[3]  Phase 1b commit 1 (TtftHistogram + RunAggregator, no Simulator change).
[4]  Phase 1b commit 2 (Simulator gains retain_records, Run carries
     aggregator; default behavior unchanged).
[5]  Phase 1b commit 3 (run_section opts out, summarize_runs merges
     histograms, histograms/*.json artifacts; regenerate golden).
[6]  Phase 2: full 1.8M trace, all 3 seeds. Confirm §6.1 acceptance.
[7]  Brief note in SIMULATION_SECTION_REFACTOR.md that real_world is now
     full-trace runnable.
```

Each step is independently revertable. If any step's acceptance criterion
fails, stop and re-plan rather than proceeding with the next step.
