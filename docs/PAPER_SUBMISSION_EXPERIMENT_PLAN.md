# Paper Submission Experiment Plan

This document is the source material for a short slide deck to discuss with
Juncheng. Its purpose is to answer one question:

```text
What experiments remain before paper submission?
```

Algorithm history and code-reading guidance live in
`docs/ALGORITHM_EVOLUTION_ROADMAP.md`.

## Final Method To Claim

The final method we are trying to support is:

```text
cost router
  -> LP-TTFT-budget
  -> Hedge-ProbTarget
  -> Explorer feedback
```

Historical variants such as `LP-CDF`, `Hedge-Economic`, and `LP-V2/P50-band`
are ablations or background. They are not the final method unless we decide to
change the paper story.

## Paper-Facing Workloads

The main paper tables should be organized around three datasets:

| Paper name | Code/data alias | Size | Role |
| --- | --- | ---: | --- |
| ShareGPT | `sharegpt` | 201K requests / 7 days / 1 model | Standard chatbot workload |
| FreeInference | `freeinference` | 371K requests / 90 days / 12 models | Multi-model API gateway workload |
| Enterprise | `rednote` currently in code | 55K requests / 84 days / 8 models | Long-context, high-variance production workload |

Naming action item:

- Decide whether the public paper name is `Enterprise` or `RedNote`, then use
  the same name in code aliases, artifact directories, plots, and paper text.

BurstGPT handling:

- BurstGPT is not one of the three main datasets in this plan.
- If used, treat it as an arrival/timing trace for online replay only — the
  ShareGPT+BurstGPT composite dataset already covers the natural-rate replay
  use case (see `WORKLOAD_DATASET_IDS` in `eval_grid.py`).
- **Do not use BurstGPT for "scale stress" experiments.** Per the
  2026-04-29 directive (DESIGN_PRINCIPLES.md §4), the simulator does not
  fabricate capacity stress by compressing arrival timestamps; trace replay
  is unconditionally natural-rate. Capacity stress is a production-layer
  concern handled by real-experiment data, not by the simulator grid.
- Do not mix BurstGPT into the three-dataset result table unless the paper
  explicitly calls it a fourth trace.

## P0. Implementation Gate

Before running paper-grade experiments, the latest method must be runnable and
auditable.

Required:

- `LP-TTFT-budget` is available as a named strategy, not only as a sidecar helper.
- `Hedge-ProbTarget` is available as a named hedger, not silently mapped to
  `smart_economic`.
- Explorer can be toggled independently from the hedge trigger.
- Runs log strategy name, LP status, hedge trigger type, hedge rate, Explorer
  feedback count, provider mix, and artifact path.

Pass condition:

- One command can run the final method and the ablations below on a selected
  dataset.

## P1. Main Progression Experiment

Goal: show the progression from old method to final method.

Run each paper workload with:

1. `LP-CDF + no hedge`
2. `LP-CDF + Hedge-Economic`
3. `LP-TTFT-budget + no hedge`
4. `LP-TTFT-budget + Hedge-ProbTarget`
5. `LP-TTFT-budget + Hedge-ProbTarget + Explorer`

Metrics:

- mean cost/request
- P50/P95/P99 TTFT
- SLO violation rate
- hedge rate
- Explorer feedback count
- provider/tier distribution

Pass condition:

- The final method improves tail metrics over `LP-TTFT-budget + no hedge`.
- The final method has interpretable cost overhead.
- Explorer does not materially regress cost/tail, or the paper explicitly frames
  it as a profile-freshness tradeoff.

## P2. Hedge Formula Ablation

Goal: justify moving from economic hedge to probability-target hedge.

Compare under the same primary selector:

- no hedge
- residual/additive hedge, if still runnable or reconstructable
- `Hedge-Economic`
- `Hedge-ProbTarget`
- always hedge

Metrics:

- P99 TTFT
- SLO violation rate
- hedge rate
- cost overhead
- backup winner rate

Pass condition:

- `Hedge-ProbTarget` gets close to always-hedge tail behavior with much lower
  hedge rate/cost.
- Additive/residual hedge is removed from the paper or shown only as a retired
  design that over-hedges.

## P3. Explorer Ablation

Goal: isolate whether hedge-as-probe improves future routing.

Compare:

- `LP-TTFT-budget + Hedge-ProbTarget`, no backup feedback
- same, with Explorer feedback
- same, with Explorer feedback and no dedicated probing
- same, with dedicated probing only

Metrics:

- stale-profile rate
- provider profile sample counts
- provider switch latency after drift
- P99 and SLO violation after drift
- hedge rate and cost

Pass condition:

- Explorer reduces stale-profile failures or speeds up adaptation after drift.
- If random backup exploration is enabled, report its probability and separate
  it from plain hedge-as-probe.

## P4. End-To-End FreeInference First

Goal: support the paper's full RouteWise claim with a real end-to-end table.

First target:

- FreeInference must be the first E2E workload we wire and run because it is the
  strongest multi-model / high-variance API-gateway trace and is already central
  to the paper.

Required E2E policies:

- `API-only`: every request goes to on-demand API. This is the clean baseline
  for "what happens without subscriptions".
- `Greedy`: fills cheap subscription/capacity first, then spills to API.
- `Cost-only`: RouteWise cost router over `S_Q`, `S_C`, and `S_A`, with no
  latency-aware provider selection or hedging.
- `Latency-only`: latest latency stack over API providers only:
  `LP-TTFT-budget + Hedge-ProbTarget + Explorer`, with no subscription cost
  router.
- `Joint`: full RouteWise E2E:
  cost router over `S_Q`, `S_C`, `S_A` + `LP-TTFT-budget` +
  `Hedge-ProbTarget` + Explorer.
- `OpenRouter auto / sort modes`: only for production/OpenRouter validation,
  not required for offline trace-only FreeInference.

Minimum first-batch E2E:

- FreeInference with `API-only`, `Greedy`, `Cost-only`, and `Joint`.
- Add `Latency-only` once the API-provider latency layer is wired.
- Then repeat the finalized E2E set on ShareGPT and Enterprise.

FreeInference integration requirements:

- Load FreeInference as a real trace-driven workload, not synthetic token draws.
- Preserve request timestamps, model names, input tokens, output tokens, and
  total tokens.
- Map every model to pricing and provider eligibility.
- Define `S_A`, `S_Q`, and `S_C` providers used in the E2E run.
- Log per-request selected tier, selected provider, realized TTFT, SLO
  violation, hedge metadata, and realized cost.
- Store artifacts under a dataset-specific output directory, for example
  `outputs/e2e/freeinference/<run_id>/`.

Metrics:

- total cost
- relative cost vs API-only and, if available, offline optimal
- mean cost/request
- P50/P95/P99 TTFT
- SLO violation rate
- hedge rate
- tier/provider distribution

Pass condition:

- The paper has a real FreeInference E2E table with generated artifacts.
- `Joint` beats `API-only` on cost and does not materially violate the target
  TTFT SLO.
- `Joint` improves over `Greedy` on either cost, tail latency, or both, with the
  tradeoff stated clearly.
- The current paper `End-to-End Evaluation` TODO is removed only after this
  exists.

## P5. Production/OpenRouter Reproducibility

Goal: make production latency numbers auditable.

Required artifacts:

- raw evaluation log
- provider percentile/profile log
- counterfactual or replay command
- generated summary CSV
- generated plots/tables

Pass condition:

- Every headline OpenRouter number in the paper can be traced to a CSV/JSON
  artifact.
- If headline numbers change from older drafts, update abstract, introduction,
  experiments, conclusion, and appendix consistently.

## Artifact Rule

Every paper figure/table must have:

```text
raw data path
command to generate
output CSV/JSON path
paper figure/table path
commit hash or run timestamp
```

Do not manually type final paper numbers unless the generated artifact is also
checked in or documented.

## Suggested Slide Deck

Keep this deck short. The goal is not to explain all algorithm history; it is to
get agreement on the remaining experiments.

1. **Goal**
   - What experiments remain before paper submission?

2. **Final Method To Claim**
   - `cost router -> LP-TTFT-budget -> Hedge-ProbTarget -> Explorer`
   - Historical variants are ablations.

3. **Datasets**
   - ShareGPT, FreeInference, Enterprise/RedNote.
   - FreeInference is first E2E target.

4. **Implementation Gate**
   - Named strategy, named hedger, Explorer toggle, logging/artifacts.

5. **Main Progression Experiment**
   - Five variants from `LP-CDF` to final method.

6. **Hedge + Explorer Ablations**
   - Hedge formula ablation.
   - Explorer feedback ablation.

7. **E2E FreeInference**
   - `API-only`, `Greedy`, `Cost-only`, `Joint`, then `Latency-only`.

8. **Submission Gate**
   - Every claim has data, command, artifact, and paper number alignment.
