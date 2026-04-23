# Smart Hedging: From Heuristic to Economic Model

## 1. Background: The Mismatch

During the implementation of our Smart Hedging feature (Phase 4 & 5), we discovered a discrepancy between the theoretical concept and the actual system execution. 

Originally, the idea discussed with Juncheng was framed around a **Serial (Cancel-and-Resend)** model:
> "if the E[latency | elapsed time] + E[latency of the fastest provider] > P99, then we **cancel** the current request and send a duplicate request"

However, modern LLM gateways (and our actual codebase) use a **Parallel (Racing)** model to minimize tail latency: we keep the primary request running, send a backup concurrently, and use whichever returns first. 

Using a formula derived for serial execution (which adds expected times together) to govern a parallel system (which takes the `min` of completion times) leads to extreme over-hedging. The system becomes overly pessimistic and triggers hedge requests too often (sometimes up to 100% hedge rate).

## 2. The Solution: A Principled Economic Model

To fix this, we upgraded the heuristic into a rigorous joint-probability model for parallel racing, framed as a cost-benefit decision.

We now hedge only if the **expected benefit of avoiding a violation exceeds the cost of the backup request**.

### The New Formula (`SMART_ECONOMIC`):

$$ \underbrace{\frac{S_{primary}(L)}{S_{primary}(t)}}_{\text{Primary Risk}} \times \underbrace{F_{backup}(Remaining)}_{\text{Backup Viability}} > \underbrace{\frac{C_b}{V}}_{\text{Cost Ratio}} $$

Where:
- $S_{primary}(L) / S_{primary}(t)$ = the probability the primary violates the SLO, given it has already been running for $t$ seconds.
- $F_{backup}(Remaining)$ = the probability the backup finishes within the remaining SLO budget.
- $C_b$ = Cost of the backup request.
- $V$ = Penalty cost for violating the SLO.

### Why this is a Massive Upgrade for the Paper:

1. **Mathematically Rigorous for Parallel Systems:** Instead of adding expectations, it calculates the joint probability that *both* the primary and the backup fail to meet the SLO.
2. **Economically Interpretable:** The threshold is no longer a magic number (`theta = 0.9`), but a principled ratio ($C_b / V$).
3. **Natural Adaptivity:** If the backup provider is cheap (e.g., Parasail), the threshold is low, and the system hedges aggressively. If the backup is expensive (e.g., Groq), the system hedges conservatively. 
4. **Subsumes Previous Baselines:** The original survival heuristic is just a special case of this model where $V \to \infty$ (violations are infinitely costly).

## 3. Action Items Completed / In Progress

1. **Codebase updated:** We replaced the flawed `smart_residual` and `PERCENTILE_BASED` strategies with the new `SMART_ECONOMIC` implementation in `experiment/strategies/smart_hedging.py`.
2. **Re-running Phase 4 (Simulation):** We are sweeping the `cost_ratio` parameter to generate a smooth, stable Pareto frontier of Hedge Rate vs. Violation Rate.
3. **Re-running Phase 5 (Production):** We will use the optimal `cost_ratio` to run a new 24-hour live evaluation against OpenRouter to get our final production numbers.
4. **Updating Paper/Slides:** We will rewrite Section 4b to highlight this joint-probability economic model, turning a potential implementation flaw into a major theoretical contribution.
