"""Physical and protocol constants shared across RouteWise harnesses.

Constants in this module are deliberately defined once and re-imported by
every consumer (policies, engine, ablations) so that there is exactly one
place to change a value. If you edit a constant here, every caller picks it
up automatically — but you should also re-capture golden baselines, because
the simulator's behaviour will change.

Do not add module-local copies of these constants in other files. The
historical cost of having two `DISPATCH_OVERHEAD_MS = 50.0` definitions
(one in the hedging policy helper for the algorithm's estimate, one in
`rwsim/engine/simulator.py` for the simulator's physical model) was exactly
the kind of silent drift this module exists to prevent.
"""

from __future__ import annotations

# Canonical primary SLO threshold used by the paper-facing RouteWise runs.
# Section-specific stress tests may override this explicitly, but default
# simulator scenarios, RouteWise policy presets, and real-eval launch scripts
# should converge on this value.
DEFAULT_PRIMARY_SLO_MS: float = 3000.0


# Estimated local hedge-dispatch overhead (milliseconds): the time between a
# RouteWise checkpoint deciding to launch a backup and the backup entering the
# transport timing window used for provider TTFT samples.
#
# In real-eval, this interval is real wall-clock time. The live runner compares
# absolute primary/backup first-token timestamps, so it must not add this
# constant to measured outcomes. The constant exists for the simulator, which
# has no real thread/HTTP handoff to account for.
#
# This is NOT network latency or provider TTFT. Provider latency profiles are
# measured from transport start to first visible token and already include HTTP
# request/response time and server round trip. Treating this constant as
# network time would double-count latency and make hedging too conservative.
#
# The simulator-side consumers must use the same value:
#
#   1. `rwsim/core/hedging.py:combined_success_probability` subtracts it
#      from `remaining_ms` when estimating how much time a simulated backup has
#      after a hedge decision.
#   2. `rwsim/engine/simulator.py:_execute_request` adds it to simulated
#      backup-wins TTFT to emulate the local dispatch handoff that real-eval
#      observes naturally.
#
# If the policy estimate and the simulator's physical emulation diverge,
# simulator hedging results stop being self-consistent.
#
# Calibrated 2026-05-21 from ~10,200 hedge dispatches across MiniMax
# real-eval runs (2026-05-13 -- 2026-05-20), measured via
# `backup_dispatch_overhead_ms` in `experiments/real_evaluation/recorder.py`.
# That metric is `backup_start_ts - backup_dispatch_ts`: the time from just
# before the backup thread is started to the transport entry timestamp, before
# the transport begins its own TTFT timer. Empirical distribution:
#   P50 = 0.31 ms, P90 = 5.41 ms, P95 = 6.61 ms, P99 = 17.96 ms,
#   max = 87.44 ms (single outlier).
# The value tracks roughly the empirical P90: enough headroom for typical
# local handoff delays without being skewed by tail outliers.
#
# Caveats:
#   1. Calibration data is all MiniMax live-eval traffic on the current runner.
#      Different hosts, executor implementations, or load regimes may warrant
#      re-measurement.
#   2. Overhead is load-dependent: P90 ~0.4 ms at low load (sanity runs),
#      ~5 ms at BurstGPT cap10s load. The 5 ms value matches the high-load
#      operating point, which is the regime where hedging decisions matter.
#   3. Historical value was 50 ms (pre-data heuristic). See the commit that
#      introduced this module for the rationale change.
#
# To update: re-run the quantile analysis on fresh real-eval CSVs, edit the
# value below, and re-capture golden baselines.
DISPATCH_OVERHEAD_MS: float = 5.0


# Minimum acceptable end-to-end success probability for a hedging decision
# to fire. RouteWise's probability-target hedger searches for the latest
# dispatch time `t` such that `P(primary or backup meets SLO | wait t) >=
# HEDGE_SUCCESS_TARGET`. Setting the target higher fires hedges earlier
# (more cost, more SLO safety); lowering it fires later (less cost, more
# SLO risk).
#
# This is a paper-level RouteWise protocol constant. Every harness that
# implements probability-target hedging must read the same value:
#
#   1. `rwsim/core/hedging.py` — shared hedging math.
#   2. `experiments/real_evaluation/policies.py` — live-API hedging adapter.
#   3. (future) hybridInference production router, once it adopts the
#      probability-target hedger in place of the legacy SMART_ECONOMIC logic.
#
# The 0.99 value follows the paper convention (Section 4 hedging design).
# Do not change this without a paper-level decision: every figure and
# headline number that mentions hedging SLO success rate assumes 0.99.
HEDGE_SUCCESS_TARGET: float = 0.99


# Probability-target hedge checkpoint schedule. RouteWise re-evaluates each
# in-flight request at a fixed set of elapsed times relative to its SLO, and
# may dispatch a backup at any of them if the conditional success probability
# falls below `HEDGE_SUCCESS_TARGET`. The three fractions below define that
# schedule:
#
#   - `HEDGE_CHECKPOINT_START_FRACTION` — earliest elapsed fraction of SLO at
#     which a hedge may be considered. Earlier elapsed times are skipped so
#     that the primary has enough opportunity to finish without paying for an
#     unnecessary backup.
#   - `HEDGE_CHECKPOINT_END_FRACTION` — latest elapsed fraction. Past this
#     point, dispatching a backup is unlikely to beat the SLO even if it runs
#     at the provider's median TTFT.
#   - `HEDGE_CHECKPOINT_INTERVAL_FRACTION` — gap between adjacent checkpoints
#     expressed as a fraction of SLO. Finer granularity catches more
#     opportunities to hedge at the cost of more frequent re-evaluation.
#
# These are paper-level RouteWise protocol constants. Every harness that
# implements probability-target hedging reads the same values:
#
#   1. `rwsim/core/hedging.py:hedge_checkpoints_for_slo` — canonical
#      simulator + real-eval implementation.
#   2. `experiments/real_evaluation/runner.py` — calls the canonical helper
#      with the real-eval SLO converted to milliseconds.
#   3. (future) hybridInference production router, once it adopts the
#      checkpoint-based hedger in place of the legacy SMART_ECONOMIC logic.
#
# Defaults (0.25 / 0.90 / 0.025) match the paper Section 4 hedging schedule.
# Editing them shifts every hedge decision in every harness; re-capture
# golden baselines after any change.
HEDGE_CHECKPOINT_START_FRACTION: float = 0.25
HEDGE_CHECKPOINT_END_FRACTION: float = 0.90
HEDGE_CHECKPOINT_INTERVAL_FRACTION: float = 0.025


__all__ = [
    "DEFAULT_PRIMARY_SLO_MS",
    "DISPATCH_OVERHEAD_MS",
    "HEDGE_CHECKPOINT_END_FRACTION",
    "HEDGE_CHECKPOINT_INTERVAL_FRACTION",
    "HEDGE_CHECKPOINT_START_FRACTION",
    "HEDGE_SUCCESS_TARGET",
]
