# Synthetic Simulation — Results

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Requests per scenario | 2 000 |
| Simulation duration | 3 600 s (1 hour) |
| Arrival process | Poisson |
| Input tokens | 100 (fixed) |
| Output tokens | LogNormal(mu=4.0, σ=1.0), clipped [1, 4096] — P50 ≈ 55, E[·] ≈ 90 |
| Token-per-second distribution | LogNormal(mu=5.5, σ=0.3) — P50 ≈ 245 tps |
| Workload seed | 0 (shared across all strategies) |
| Strategy seeds | [42, 43, 44] — 3 trials, results averaged |
| LP update interval | 60 s (all four adaptive strategies) |
| Router SLO | 2 000 ms |
| Reporting SLO thresholds | 1 000 / 2 000 / 3 000 / 5 000 ms |
| Profile window | 15 min (900 s) |
| Warm-up samples | 50 per provider, spread over 14 min before t=0 |
| Probe rate | 5 % per non-primary provider per request |
| Hedge cost_ratio | 0.1 |
| Hedge dispatch overhead | 50 ms |
| V2 near-best P50 band | 10 % |

---

## Provider Configurations

All costs are in USD per token. P50/P99 are TTFT in milliseconds. LogNormal parameters computed from target P50/P99: `mu = ln(P50)`, `sigma = (ln(P99) − ln(P50)) / 2.326`.

### S1 — Dominant Provider

Provider B is simultaneously cheapest and fastest. This is the core LP over-diversification scenario.

| Provider | Cost | Target P50 | Target P99 | mu | sigma |
|----------|------|-----------|-----------|-----|-------|
| A | $2/M tokens (2.0 × 10⁻⁶ /token) | 1 100 ms | 7 000 ms | 7.003 | 0.796 |
| B | $0.5/M tokens (0.5 × 10⁻⁶ /token) | 100 ms | 320 ms | 4.605 | 0.500 |

### S2 — Cost/Latency Trade-off

A is cheap but slow; B is fast but expensive. The optimal choice depends on which SLO matters.

| Provider | Cost | Target P50 | Target P99 | mu | sigma |
|----------|------|-----------|-----------|-----|-------|
| A | $0.5/M tokens | 500 ms | 1 600 ms | 6.215 | 0.500 |
| B | $2/M tokens | 100 ms | 320 ms | 4.605 | 0.500 |

### S3 — Tail-Heavy vs Stable

A has an excellent P50 but a catastrophic P99 (log-normal with high sigma). B is slower on P50 but tightly bounded.

| Provider | Cost | Target P50 | Target P99 | mu | sigma |
|----------|------|-----------|-----------|-----|-------|
| A | $1/M tokens | 100 ms | 5 000 ms | 4.605 | 1.682 |
| B | $1/M tokens | 300 ms | 1 000 ms | 5.704 | 0.518 |

### S4 — Distribution Shift at t = 30 min

Provider A degrades at t = 1 800 s (30 min). The 15-min rolling window means the router's profile for A fully reflects the slow distribution only after t = 2 700 s (45 min), i.e., 15 min after the shift. Provider B is constant throughout.

| Provider | Phase | Cost | Target P50 | mu | sigma |
|----------|-------|------|-----------|-----|-------|
| A | t < 1 800 s | $1/M | 100 ms | 4.605 | 0.500 |
| A | t ≥ 1 800 s | $1/M | 1 000 ms | 6.908 | 0.500 |
| B | all | $1/M | 300 ms | 5.704 | 0.500 |

### S5 — Three Similar Providers

All three providers have P50 values within V2's 10 % near-best band. C is cheapest. Note: the README's original P50=250 ms for C would place it 25 % above A (outside the 10 % band); adjusted to 215 ms (7.5 % above A).

| Provider | Cost | Target P50 | mu | sigma |
|----------|------|-----------|-----|-------|
| A | $1/M tokens | 200 ms | 5.298 | 0.500 |
| B | $0.8/M tokens | 207 ms | 5.332 | 0.500 |
| C | $0.5/M tokens | 215 ms | 5.371 | 0.500 |

---

## Results

Numbers are means across 3 seeds. Violation rates at SLO=2 000 ms are the primary metric. All costs in USD per request.

### S1 — Dominant Provider

**Ground truth:** B should receive 100 % of traffic.

| Strategy | SLO@1s | SLO@2s | SLO@3s | SLO@5s | Mean cost | P50 | P99 | Provider mix | Hedge rate |
|----------|--------|--------|--------|--------|-----------|-----|-----|--------------|-----------|
| cheapest_fixed | 0.0 % | **0.0 %** | 0.0 % | 0.0 % | $9.50e-5 | 99.5 ms | 324 ms | B=100 % | — |
| fastest_fixed | 0.0 % | **0.0 %** | 0.0 % | 0.0 % | $9.50e-5 | 99.5 ms | 324 ms | B=100 % | — |
| round_robin | 27.1 % | **11.1 %** | 5.3 % | 1.5 % | $2.40e-4 | 249 ms | 5 534 ms | A=50 %, B=50 % | — |
| lp_mix | 0.7 % | **0.3 %** | 0.2 % | 0.03 % | $9.88e-5 | 101 ms | 591 ms | A=1.4 %, B=98.6 % | — |
| lp_hedge | 0.0 % | **0.0 %** | 0.0 % | 0.0 % | $1.01e-4 | 101 ms | 326 ms | A=1.4 %, B=98.6 % | 1.5 % |
| v2_only | 0.7 % | **0.3 %** | 0.2 % | 0.03 % | $9.88e-5 | 101 ms | 591 ms | A=1.4 %, B=98.6 % | — |
| v2_p50_hedge | 0.0 % | **0.0 %** | 0.0 % | 0.0 % | $1.01e-4 | 101 ms | 326 ms | A=1.4 %, B=98.6 % | 1.5 % |
| oracle_per_window | 0.0 % | **0.0 %** | 0.0 % | 0.0 % | $9.50e-5 | 99.5 ms | 324 ms | B=100 % | — |

**Findings:**
- `cheapest_fixed` and `fastest_fixed` both pick B (it dominates on both dimensions) — 0 % violations at lowest cost. This is the ideal outcome.
- `lp_mix` and `v2_only` produce **identical results** in S1: both route ~98.6 % to B with 1.4 % probe traffic to A, giving the same 0.3 % violations and cost. When B overwhelmingly dominates, both LP and V2 converge to the same routing distribution — the routing algorithm choice is irrelevant.
- `lp_hedge` and `v2_p50_hedge` likewise produce identical results: same routing distribution, same 1.5 % hedge rate (triggered on the rare slow A probes), 0 % violations. The hedge rate is low because A receives only probe traffic.
- `round_robin` forces 50 % to slow/expensive A → 11 % violations and 2.5× higher cost. Confirms the damage of ignoring provider quality.

---

### S2 — Cost/Latency Trade-off

**Ground truth:** Optimal choice depends on SLO. At SLO=2 s, A is viable (only 0.4 % violations) and 4× cheaper. At SLO=1 s, A fails (8.3 % violations) and B is required.

| Strategy | SLO@1s | SLO@2s | SLO@3s | SLO@5s | Mean cost | P50 | P99 | Provider mix | Hedge rate |
|----------|--------|--------|--------|--------|-----------|-----|-----|--------------|-----------|
| cheapest_fixed | 8.3 % | **0.4 %** | 0.0 % | 0.0 % | $9.50e-5 | 498 ms | 1 622 ms | A=100 % | — |
| fastest_fixed | 0.0 % | **0.0 %** | 0.0 % | 0.0 % | $3.80e-4 | 99.5 ms | 324 ms | B=100 % | — |
| round_robin | 4.2 % | **0.2 %** | 0.0 % | 0.0 % | $2.34e-4 | 220 ms | 1 380 ms | A=50 %, B=50 % | — |
| lp_mix | 8.1 % | **0.3 %** | 0.0 % | 0.0 % | $9.60e-5 | 499 ms | 1 628 ms | A=99.6 %, B=1.2 % | — |
| lp_hedge | 8.4 % | **0.02 %** | 0.0 % | 0.0 % | $1.06e-4 | 499 ms | 1 431 ms | A=100 % | 3.0 % |
| v2_only | 0.13 % | **0.02 %** | 0.0 % | 0.0 % | $3.76e-4 | 101 ms | 421 ms | A=1.4 %, B=98.6 % | — |
| v2_p50_hedge | 0.1 % | **0.0 %** | 0.0 % | 0.0 % | $3.76e-4 | 101 ms | 443 ms | A=1.4 %, B=98.6 % | 0.12 % |
| oracle_per_window | 0.0 % | **0.0 %** | 0.0 % | 0.0 % | $3.80e-4 | 99.5 ms | 324 ms | B=100 % | — |

**Findings:**
- `lp_mix` and `lp_hedge` both route almost entirely to A (cheap, slow). `lp_mix` produces 0.3 % SLO@2s violations; `lp_hedge` hedges 3.0 % of A requests (the P99 tail that breaches 2 s), reducing violations to 0.02 %. Adding hedging to LP preserves LP's cost advantage while nearly eliminating the SLO tail.
- `v2_only` and `v2_p50_hedge` both route to B (better P50). `v2_only` produces 0.02 % SLO@2s violations (B has a very low tail); `v2_p50_hedge` triggers almost no hedges (0.12 %) and reaches 0 %. The hedge is largely superfluous here because B is already SLO-safe.
- The routing algorithm is the critical axis: LP (→ A) achieves 4× lower cost than V2 (→ B) regardless of whether hedging is applied. The SLO@1s dimension exposes LP's fundamental limit — A's P50 is 500 ms, so ~8 % of requests breach 1 s — which hedging cannot fix cheaply.
- **Takeaway:** When cost/latency trade-off governs, LP's cost objective dominates. Adding hedging to LP (`lp_hedge`) achieves near-LP-cost at near-V2 SLO@2s, at only a 10 % cost premium over `lp_mix`.

---

### S3 — Tail-Heavy vs Stable

**Ground truth:** A has better P50 (100 ms vs 300 ms) but B has far lower P99 (1 000 ms vs 5 000 ms). At SLO=2 s, A produces ~3.9 % violations; B produces ~0 %. With both providers costing the same, SLO-optimal routing should prefer B.

| Strategy | SLO@1s | SLO@2s | SLO@3s | SLO@5s | Mean cost | P50 | P99 | Provider mix | Hedge rate |
|----------|--------|--------|--------|--------|-----------|-----|-----|--------------|-----------|
| cheapest_fixed | 8.6 % | **3.9 %** | 2.2 % | 1.1 % | $1.90e-4 | 98.4 ms | 5 245 ms | A=100 % | — |
| fastest_fixed | 8.6 % | **3.9 %** | 2.2 % | 1.1 % | $1.90e-4 | 98.4 ms | 5 245 ms | A=100 % | — |
| round_robin | 4.9 % | **1.9 %** | 1.1 % | 0.5 % | $1.90e-4 | 228 ms | 3 049 ms | A=50 %, B=50 % | — |
| lp_mix | 3.4 % | **1.2 %** | 0.7 % | 0.3 % | $1.90e-4 | 267 ms | 2 241 ms | A=30 %, B=70 % | — |
| lp_hedge | 0.8 % | **0.0 %** | 0.0 % | 0.0 % | $2.07e-4 | 265 ms | 961 ms | A=28.5 %, B=71.5 % | 9.2 % |
| v2_only | 8.5 % | **3.7 %** | 2.2 % | 1.1 % | $1.90e-4 | 100 ms | 5 316 ms | A=100 % | — |
| v2_p50_hedge | 0.6 % | **0.0 %** | 0.0 % | 0.0 % | $2.53e-4 | 99.3 ms | 916 ms | A=100 % | 33.2 % |
| oracle_per_window | 8.6 % | **3.9 %** | 2.2 % | 1.1 % | $1.90e-4 | 98.4 ms | 5 245 ms | A=100 % | — |

**Findings:**
- `v2_only` routes 100 % to A (best P50) with no hedging, producing 3.7 % SLO@2s violations — matching the stateless baselines. **V2 routing alone is insufficient** here because P50 ranking selects the heavy-tailed provider.
- `lp_mix` diversifies 70 % to B (better CDF at 2 s), reducing violations to 1.2 %. LP's diversification is **beneficial** — it routes away from the heavy tail even without hedging.
- `v2_p50_hedge` routes 100 % to A but hedges 33.2 % of requests, eliminating violations at a 33 % cost premium. Hedging compensates for the routing algorithm's failure to avoid the tail.
- `lp_hedge` routes 28.5 % to A / 71.5 % to B (same LP diversification as `lp_mix`), then hedges only 9.2 % of requests. This achieves 0 % violations at $2.07e-4 — **18 % cheaper than `v2_p50_hedge`** ($2.53e-4). The routing diversification reduces hedge frequency dramatically (9.2 % vs 33.2 %).
- `cheapest_fixed`, `fastest_fixed`, and `oracle_per_window` all pick A, getting 3.9 % violations. **The P50-based oracle is suboptimal** when the P50-optimal provider has a heavy tail.
- **Takeaway:** S3 is the key scenario distinguishing routing from hedging. LP's diversification reduces the hedge burden; combining LP routing with hedging (`lp_hedge`) achieves the best cost-SLO trade-off.

---

### S4 — Distribution Shift

**Ground truth:** Route to A (P50=100 ms) for t < 30 min; switch to B (P50=300 ms) for t ≥ 30 min. Latency window dynamics: fast A samples expire from the 15-min window at t = 45 min (15 min after the shift), so full adaptation takes 15 min.

| Strategy | SLO@1s | SLO@2s | SLO@3s | SLO@5s | Mean cost | P50 | P99 | Provider mix | Hedge rate |
|----------|--------|--------|--------|--------|-----------|-----|-----|--------------|-----------|
| cheapest_fixed | 25.4 % | **4.2 %** | 0.8 % | 0.02 % | $1.90e-4 | 351 ms | 2 853 ms | A=100 % | — |
| fastest_fixed | 0.9 % | **0.0 %** | 0.0 % | 0.0 % | $1.90e-4 | 299 ms | 973 ms | B=100 % | — |
| round_robin | 13.0 % | **2.1 %** | 0.4 % | 0.0 % | $1.90e-4 | 305 ms | 2 433 ms | A=50 %, B=50 % | — |
| lp_mix | 6.6 % | **1.0 %** | 0.2 % | 0.02 % | $1.90e-4 | 183 ms | 2 022 ms | A=61.1 %, B=38.9 % | — |
| lp_hedge | 3.3 % | **0.03 %** | 0.0 % | 0.0 % | $2.09e-4 | 182 ms | 1 187 ms | A=61.3 %, B=38.7 % | 9.7 % |
| v2_only | 6.0 % | **0.98 %** | 0.12 % | 0.02 % | $1.90e-4 | 183 ms | 1 999 ms | A=60.3 %, B=39.7 % | — |
| v2_p50_hedge | 2.7 % | **0.03 %** | 0.0 % | 0.0 % | $2.09e-4 | 182 ms | 1 143 ms | A=60.9 %, B=39.1 % | 9.8 % |
| oracle_per_window | 0.5 % | **0.0 %** | 0.0 % | 0.0 % | $1.90e-4 | 173 ms | 856 ms | A=49 %, B=51 % | — |

**Findings:**
- `lp_mix` and `v2_only` converge to nearly identical routing distributions (~61 % A / 39 % B) and similar SLO@2s violations (1.0 % vs 0.98 %). Both adapt through the same 15-min rolling window mechanism and are equally slow to react.
- `lp_hedge` and `v2_p50_hedge` both achieve 0.03 % SLO@2s at identical cost ($2.09e-4) and similar hedge rates (9.7 % vs 9.8 %). The hedge absorbs violations during the 15-min transition window (t = 30–45 min) that the routing algorithm cannot prevent through weight adjustment alone.
- The routing algorithm choice (`lp_mix` vs `v2_only`, or `lp_hedge` vs `v2_p50_hedge`) makes **no material difference** in S4. Both converge to the same distribution. The hedging layer is what separates the two performance tiers.
- `oracle_per_window` switches exactly at t = 30 min, achieving near-zero violations. The remaining 0.5 % SLO@1s violations come from B's natural P99 (≈ 973 ms breaching 1 000 ms).
- **Takeaway:** In distribution-shift scenarios, hedging is the critical mechanism. The routing algorithm adapts at the same rate regardless of LP vs V2; the hedge catches violations during the adaptation gap.

---

### S5 — Three Similar Providers

**Ground truth:** All three providers are within V2's 10 % near-best P50 band. C (cheapest) should be selected. Adjusted P50 values: A=200 ms, B=207 ms (3.5 % above A), C=215 ms (7.5 % above A). Note: the original README values (P50=250 ms for C, 25 % above A) would place C outside V2's 10 % band; they were adjusted to validate the intended behavior.

| Strategy | SLO@1s | SLO@2s | SLO@3s | SLO@5s | Mean cost | P50 | P99 | Provider mix | Hedge rate |
|----------|--------|--------|--------|--------|-----------|-----|-----|--------------|-----------|
| cheapest_fixed | 0.2 % | **0.0 %** | 0.0 % | 0.0 % | $9.50e-5 | 214 ms | 697 ms | C=100 % | — |
| fastest_fixed | 0.05 % | **0.0 %** | 0.0 % | 0.0 % | $1.90e-4 | 199 ms | 649 ms | A=100 % | — |
| round_robin | 0.1 % | **0.0 %** | 0.0 % | 0.0 % | $1.46e-4 | 206 ms | 679 ms | A=33 %, B=33 %, C=33 % | — |
| lp_mix | 0.2 % | **0.0 %** | 0.0 % | 0.0 % | $9.62e-5 | 214 ms | 705 ms | A=1.4 %, C=98.6 % | — |
| lp_hedge | 0.13 % | **0.0 %** | 0.0 % | 0.0 % | $9.65e-5 | 214 ms | 698 ms | A=1.4 %, C=98.6 % | 0.19 % |
| v2_only | 0.17 % | **0.0 %** | 0.0 % | 0.0 % | $1.13e-4 | 211 ms | 693 ms | A=9.1 %, B=17.6 %, C=73.3 % | — |
| v2_p50_hedge | 0.1 % | **0.0 %** | 0.0 % | 0.0 % | $1.29e-4 | 209 ms | 678 ms | A=5.6 %, B=49.1 %, C=45.3 % | 0.5 % |
| oracle_per_window | 0.05 % | **0.0 %** | 0.0 % | 0.0 % | $1.90e-4 | 199 ms | 649 ms | A=100 % | — |

**Findings:**
- All strategies meet SLO=2 s. The differentiating dimension is **cost**.
- `cheapest_fixed`, `lp_mix`, and `lp_hedge` all route primarily to C (cheapest). LP's cost objective stably selects C regardless of minor P50 noise; `lp_hedge` fires a negligible 0.19 % hedge rate (the band is comfortably within SLO) and matches LP's cost almost exactly.
- `v2_only` shows the P50-measurement-noise problem without hedging: C has true P50=215 ms but a standard error of ±14 ms with 150 samples. When C's measured P50 exceeds the 220 ms band threshold, V2 falls back to B (cheaper than A within the band). This explains the A=9.1 %, B=17.6 %, C=73.3 % split and the higher-than-optimal cost ($1.13e-4 vs $9.50e-5).
- `v2_p50_hedge` has the same noise-driven routing instability (A=5.6 %, B=49.1 %, C=45.3 %) at even higher cost ($1.29e-4) due to redundant hedges. Adding hedging to a noisy V2 worsens the cost outcome without improving SLO (which was already 0 %).
- **Takeaway:** In the similar-provider scenario, LP robustly picks the cheapest option while V2's P50 measurement noise causes it to fluctuate across providers. Adding hedging to V2 compounds this cost without benefit, while adding hedging to LP (`lp_hedge`) costs essentially nothing extra.

---

## Summary Table (SLO@2 s, averaged across scenarios)

| Strategy | S1 | S2 | S3 | S4 | S5 | Avg |
|----------|----|----|----|----|----|----|
| cheapest_fixed | 0.0 % | 0.4 % | 3.9 % | 4.2 % | 0.0 % | 1.7 % |
| fastest_fixed | 0.0 % | 0.0 % | 3.9 % | 0.0 % | 0.0 % | 0.8 % |
| round_robin | 11.1 % | 0.2 % | 1.9 % | 2.1 % | 0.0 % | 3.1 % |
| lp_mix | 0.3 % | 0.3 % | 1.2 % | 1.0 % | 0.0 % | 0.6 % |
| lp_hedge | **0.0 %** | **0.02 %** | **0.0 %** | **0.03 %** | **0.0 %** | **0.01 %** |
| v2_only | 0.3 % | 0.02 % | 3.7 % | 0.98 % | 0.0 % | 1.0 % |
| **v2_p50_hedge** | **0.0 %** | **0.0 %** | **0.0 %** | **0.03 %** | **0.0 %** | **0.01 %** |
| oracle_per_window | 0.0 % | 0.0 % | 3.9 % | 0.0 % | 0.0 % | 0.8 % |

`lp_hedge` and `v2_p50_hedge` are tied on SLO@2s (both 0.01 % average). `v2_only` is worse than `lp_mix` mainly due to S3 (3.7 % vs 1.2 %), where LP's diversification helps.

### Cost comparison (mean cost per request)

| Strategy | S1 | S2 | S3 | S4 | S5 |
|----------|----|----|----|----|-----|
| cheapest_fixed | ★ $9.50e-5 | ★ $9.50e-5 | $1.90e-4 | $1.90e-4 | ★ $9.50e-5 |
| lp_mix | $9.88e-5 (+4 %) | $9.60e-5 (+1 %) | $1.90e-4 | $1.90e-4 | $9.62e-5 (+1 %) |
| **lp_hedge** | **$1.01e-4 (+6 %)** | **$1.06e-4 (+12 %)** | **$2.07e-4 (+9 %)** | **$2.09e-4 (+10 %)** | **$9.65e-5 (+2 %)** |
| v2_only | $9.88e-5 (+4 %) | $3.76e-4 (+3.9×) | $1.90e-4 | $1.90e-4 | $1.13e-4 (+19 %) |
| **v2_p50_hedge** | **$1.01e-4 (+6 %)** | **$3.76e-4 (+3.9×)** | **$2.53e-4 (+33 %)** | **$2.09e-4 (+10 %)** | **$1.29e-4 (+36 %)** |
| fastest_fixed | ★ $9.50e-5 | $3.80e-4 (+4.0×) | $1.90e-4 | $1.90e-4 | $1.90e-4 (2×) |

Percentages are relative to cheapest_fixed. In S3 and S4 all same-cost providers share equal baseline. `lp_hedge` achieves the best SLO performance at a consistently moderate cost premium (2–12 %), while `v2_p50_hedge` incurs a large premium in S2 (+3.9×) and S5 (+36 %) due to V2's P50-first routing.

---

## Key Cross-Scenario Insights

1. **`lp_hedge` and `v2_p50_hedge` are tied on SLO@2s** (both 0.01 % average), but `lp_hedge` achieves this at consistently lower cost: +2–12 % over the cheapest baseline, versus +6–390 % for `v2_p50_hedge`. The routing algorithm choice (LP vs V2) matters more for cost than for SLO when hedging is applied.

2. **`v2_only` is worse than `lp_mix` overall** (1.0 % vs 0.6 % average SLO@2s), primarily because V2's P50-first selection routes to the heavy-tailed provider A in S3 (3.7 % violations) while LP's diversification routes to stable B (1.2 % violations). Without hedging, LP's routing is more SLO-aware.

3. **Hedging contribution is scenario-dependent**:
   - S3 (tail-heavy): hedging is the key mechanism for V2 (`v2_p50_hedge` 0 % vs `v2_only` 3.7 %), but `lp_hedge` hedges 5× less (9.2 % vs 33.2 %) by combining routing diversification with hedging.
   - S4 (shift): hedging bridges the 15-min adaptation gap for both `lp_hedge` and `v2_p50_hedge`. Routing algorithm choice makes no difference.
   - S1, S2, S5: hedging has marginal value — the routing decision already determines the outcome.

4. **LP over-diversification is confirmed in S1** (1.4 % to slow+expensive A) but is *helpful* in S3 (routing 70 % to stable B reduces violations from 3.9 % to 1.2 %). LP's diversification is harmful when one provider dominates and neutral-to-helpful when one provider has a heavy tail.

5. **The P50-based oracle is not SLO-optimal**: in S3, the oracle picks A (best P50) and gets 3.9 % violations — worse than LP (1.2 %) and far worse than either hedged strategy (0 %). Both `v2_only` and the oracle suffer the same fate, showing that P50 ranking alone fails when tail distribution is highly asymmetric.

6. **LP is more cost-stable than V2 in the similar-provider scenario (S5)**: V2's P50 measurement noise causes both `v2_only` (+19 % cost) and `v2_p50_hedge` (+36 % cost) to route significant traffic to non-cheapest providers. LP stably routes to C in both `lp_mix` and `lp_hedge` (+1–2 % cost). Adding hedging to V2 in S5 worsens cost without any SLO benefit.
