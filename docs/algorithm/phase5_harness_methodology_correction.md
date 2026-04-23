# Phase 5 Harness Methodology Correction

## Purpose

This document explains:

1. How the **original** Phase 5 live evaluation harness worked.
2. Why that design was **not** a faithful trace replay.
3. How the **corrected** harness works now.
4. Which old results should be treated as pilot evidence versus paper-grade evidence.

This is a methodology note for the OpenRouter live evaluation only. It is
intended to make the experimental semantics explicit before we use the results
in the paper.

---

## Executive Summary

The original harness was **closed-loop**:

- For each trace entry, it dispatched the same request to all policies.
- It then waited for all policies to finish before moving to the next trace
  entry.

This design preserved per-entry cross-policy fairness, but it did **not**
preserve the original trace arrival process. In practice, the replay speed was
limited by the slowest provider in each round, so a "24-hour replay" meant
"however many trace entries the harness could process in 24 hours of wall
clock."

The corrected harness is **open-loop**:

- Requests are dispatched according to trace timestamps.
- New arrivals do not wait for earlier requests to finish.
- Each policy maintains its own mutable online router state.
- Router learning uses observation time rather than dispatch time.

The corrected harness is the version that should be treated as
publication-quality evidence.

---

## Original Harness

### High-Level Flow

The original `run_trace_replay()` logic followed this pattern:

1. Read the next trace entry.
2. Dispatch the same request to all evaluation policies in parallel.
3. Wait for every policy request in that round to complete.
4. Advance to the next trace entry.

Conceptually, the harness executed:

```text
for trace_entry in trace:
    dispatch request to all policies
    wait until the slowest policy finishes
    move to next trace entry
```

This design looked appealing because all policies saw the same request in the
same round.

### What This Design Actually Measured

It measured a **harness-limited paired shadow experiment**, not a faithful live
trace replay.

The key issue is that progress through the trace was gated by completion of the
slowest request in each round, rather than by the next trace arrival time.

If the trace had dense arrivals but one provider occasionally took 10-20
seconds, the harness effectively stretched the workload in wall-clock time and
smoothed out burst structure.

### Additional Problems in Some Earlier Runs

Beyond the replay loop itself, some earlier runs also had two other evaluation
issues:

1. `lp_mix` and `smart_hedge` shared the same mutable online router state
   during evaluation, making the ablation unclean.
2. Some earlier Qwen runs used short probe-style prompts as the **main**
   workload instead of real trace prompts in P1 mode.

These two issues are separate from the replay-loop problem, but they affected
the validity of earlier live results.

---

## Why the Original Harness Failed

### 1. It Was Closed-Loop Instead of Trace-Driven

The trace arrival process should determine **when** requests enter the system.
The original harness instead let the **slowest completion in the previous
round** determine when the next request could be issued.

That changes the workload semantics.

### 2. It Smoothed Out Bursts

BurstGPT traces can contain dense arrival bursts. The original harness could
not reproduce these bursts if the previous round had not yet drained.

As a result:

- queueing pressure on providers was reduced or distorted,
- burst-induced tail events were underrepresented or shifted,
- and the replay no longer matched the intended live arrival pattern.

### 3. "24 Hours" Referred to Wall Clock, Not Trace Time

Under the old harness, a "24-hour replay" really meant:

> run this loop for 24 wall-clock hours and process as many trace entries as the
> harness can get through.

That is not the same as replaying a 24-hour trace window faithfully.

### 4. The Ablation Was Not Clean

`smart_hedge` is supposed to use the same **primary routing algorithm** as
`lp_mix`, but it should not share the same mutable runtime state during
evaluation.

When both policies updated the same router instance, the comparison was no
longer:

- `lp_mix`
- versus `lp_mix + hedging`

Instead, it became two policies sharing a single online state, which is not a
clean ablation.

### 5. Some Older Qwen Runs Had Workload-Mode Mismatch

Short probe prompts are correct for online profiling. They are **not** the same
as real user prompts.

The problem in some earlier Qwen runs was not that probing used a short prompt.
The problem was that the **main evaluation traffic** was effectively running in
probe-style short-prompt mode rather than true P1 trace-prompt mode.

---

## Corrected Harness

### High-Level Flow

The corrected harness is open-loop.

For each trace entry:

1. Compute the scheduled dispatch time from the trace timestamp.
2. Dispatch the request group when that trace time arrives.
3. Do **not** wait for previous requests to finish before dispatching the next
   arrival.
4. Use bounded concurrency and backpressure to avoid turning the local harness
   queue into an unbounded hidden bottleneck.

Conceptually:

```text
for trace_entry in trace:
    wait until trace arrival time
    dispatch requests for this trace entry
    immediately continue scheduling future arrivals
```

### Key Corrections

The corrected harness includes the following methodology fixes:

1. **Open-loop dispatch**
   Trace timestamps drive arrivals directly.

2. **Per-policy router state**
   `lp_mix` and `smart_hedge` keep separate router instances.

3. **Policy-local learning**
   Each policy learns from:
   - shared probing observations, and
   - its own completed requests
   only.

4. **Observation-time updates**
   Router learning uses observation time rather than request dispatch time.

5. **Per-request hedging cost**
   Smart hedging uses backup cost under the actual request token budget rather
   than a single global scalar approximation.

6. **Graceful stop + drain**
   Wall-clock stop or cost-cap stop halts new dispatches and drains in-flight
   requests cleanly.

### What the Corrected Harness Measures

The corrected harness measures a **live open-loop replay under real API calls**
that is much closer to the intended experiment:

- real provider behavior,
- real arrival bursts,
- real online adaptation,
- and real interaction between provider drift and routing decisions.

---

## What Changed Semantically

### Old Semantics

The old experiment answered:

> Under a harness-limited paired shadow setup, does our method look better than
> the baselines?

### New Semantics

The corrected experiment answers:

> Under an arrival-faithful live replay with real online adaptation, how do the
> policies behave under realistic provider dynamics?

These are related questions, but they are not the same question.

---

## Status of Old Results

### Old Results That Should Be Treated as Pilot Evidence

The earlier live runs collected under the closed-loop harness should be treated
as:

- pilot evidence,
- debugging evidence, or
- qualitative support that the idea was promising.

They should **not** be treated as the final paper-grade evaluation.

### Results That Should Be Treated as Paper-Quality

Only runs collected under the corrected open-loop harness with:

- real trace timestamps,
- correct P1 prompt mode when claimed,
- per-policy router isolation,
- and the corrected hedging accounting

should be treated as publication-quality results.

---

## Practical Implications for the Paper

### What We Can No Longer Claim

We should not describe the old closed-loop runs as:

- faithful 24-hour trace replay,
- true arrival-driven live replay,
- or clean online ablations.

### What We Can Claim for the Corrected Runs

We can now claim:

- open-loop replay against OpenRouter using real API calls,
- trace-timestamp-driven arrivals,
- real BurstGPT timing with real ShareGPT prompts in P1 mode,
- and policy-isolated online adaptation.

### What This Means for the Narrative

The corrected results may be less dramatic than the earlier pilot numbers, but
they are substantially easier to defend. For paper quality, **faithful
semantics matter more than larger headline improvements**.

---

## Recommended Use

Use this document when:

- writing the methodology section for the live evaluation,
- explaining why earlier pilot numbers were replaced,
- justifying why the current runs are the paper-quality evidence,
- or answering reviewer questions about replay semantics and online state
  isolation.

