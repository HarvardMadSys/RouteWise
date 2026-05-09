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

Before replay, the runner sends warmup probes to every provider and broadcasts
the resulting TTFT/error samples to all policy-local profiles. Warmup is
round-based by default: one round probes all providers once, then the runner
waits before the next round. With the defaults, the runner sends five warmup
rounds spaced 180 seconds apart. The first round starts immediately, so replay
starts after roughly 12 minutes plus probe execution time; this keeps all five
samples safely inside the 15-minute profile window instead of placing the first
sample on the pruning boundary. For smoke tests, set
`--warmup-probe-interval-sec 0`.

After replay starts, a background maintenance loop can periodically probe all
providers once per interval and broadcast those observations in the same way.

The default periodic interval is 180 seconds. With the default 15-minute
profile window, that maintains five probe-only samples per provider when there
is no real traffic or hedge feedback for that provider. The older Phase 5/6
online harness used a 300-second periodic interval; that is still available as
a lower-probe-rate override.

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
--warmup-probes 5
--warmup-probe-interval-sec 180
--min-profile-success-samples 5
--periodic-probe-interval-sec 180
--profile-window-sec 900
```

If quota pressure is high, use:

```text
--periodic-probe-interval-sec 300
```

## Cost Accounting

Profile probes are real API requests. Their physical cost contributes to the
runner-level total cost cap, but they are not counted as per-policy workload
requests in the main request CSV/summary. This keeps policy metrics focused on
trace replay while still preventing runaway probe spend.

## Non-Goals

This layer does not implement random backup selection, bandit exploration, or a
50/50 Explorer variant. Backup dispatch remains probability-driven.
