# Real-Eval Profile Maintenance

## Goal

The real online evaluator needs fresh latency profiles before RouteWise can make
probability-driven routing and hedging decisions. This profile maintenance layer
is experiment infrastructure. It is separate from Explorer.

## Terms

- **Explorer feedback**: when RouteWise dispatches a hedged backup, the observed
  backup TTFT/error is fed back into the provider latency profile.
- **Profile maintenance probing**: lightweight requests sent by the harness to
  keep every provider's rolling profile populated.

Explorer feedback is part of the algorithmic observation path. Profile
maintenance probing is a shared measurement path used to bootstrap and refresh
the empirical profiles.

## Current Design

The live evaluator maintains one rolling TTFT profile per provider per policy.
The default profile window is 15 minutes.

Before replay, the launcher can prebuild a shared warmup profile and seed
`shared_profile_events.jsonl`. Each policy process loads that same warmup
profile before replay. Warmup is round-based by default: one round probes all
providers once, then the runner waits before the next round. With the current
true-24h launcher defaults, warmup uses 24 rounds at a 5-second cadence.
For smoke tests, set `--warmup-probe-interval-sec 0`.

During replay, profile sharing happens through `shared_profile_events.jsonl`:
natural request feedback and shared-prober samples are appended to the log, and
every policy process tails the log into its own local profile state.

Per-policy periodic probing during replay has been removed. Multi-policy
experiments should not run one maintenance probe loop per policy process; that
duplicates measurements and makes provider/key pressure hard to reason about.
Runtime maintenance should be performed by the shared prober
(`scripts/shared_profile_probe.py`) when explicit shared probing is enabled.

## Bootstrap Guard

RouteWise policies that depend on empirical latency profiles should not enter
replay with empty profiles. The runner therefore supports a bootstrap guard:

```text
--min-profile-success-samples K
```

If any provider has fewer than `K` successful warmup samples in any
profile-dependent policy state, the run fails before replay. This makes missing
profile data explicit instead of silently routing with unprofiled penalties.

For a latency-focused pilot, use:

```text
--warmup-probes 24
--warmup-probe-interval-sec 5
--min-profile-success-samples 5
--profile-window-sec 900
```

If replay-time profile maintenance is needed, launch the shared prober instead
of enabling per-policy probing.

## Cost Accounting

Profile probes are real API requests. Their physical cost contributes to the
runner-level total cost cap, but they are not counted as per-policy workload
requests in the main request CSV/summary. This keeps policy metrics focused on
trace replay while still preventing runaway probe spend.

## Non-Goals

This layer does not implement random backup selection, bandit exploration, or a
50/50 Explorer variant. Backup dispatch remains probability-driven.
