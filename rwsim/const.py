"""Physical and protocol constants shared across RouteWise harnesses.

Constants in this module are deliberately defined once and re-imported by
every consumer (policies, engine, ablations) so that there is exactly one
place to change a value. If you edit a constant here, every caller picks it
up automatically — but you should also re-capture golden baselines, because
the simulator's behaviour will change.

Do not add module-local copies of these constants in other files. The
historical cost of having two `DISPATCH_OVERHEAD_MS = 50.0` definitions
(one in `rwsim/policies/hedging.py` for the algorithm's estimate, one in
`rwsim/engine/simulator.py` for the simulator's physical model) was exactly
the kind of silent drift this module exists to prevent.
"""

from __future__ import annotations

# Estimated client-side dispatch overhead (milliseconds): the wall-clock time
# between a hedging policy deciding to send a backup and the HTTP request
# actually leaving the client. Models Python async event-loop ticks, lock
# acquisition, httpx connection setup, and payload serialization — work that
# happens *before* `start_perf = time.perf_counter()` is taken inside
# `experiments/real_evaluation/transports.py`, and is therefore not captured
# in any provider's latency profile.
#
# NOT network latency. The per-provider latency profile already includes the
# server round-trip; subtracting network latency here would double-count it
# and make hedging artificially aggressive.
#
# This constant has two consumers, which is why it lives in this module
# instead of in either of them:
#
#   1. `rwsim/policies/hedging.py:combined_success_probability` subtracts it
#      from `remaining_ms` when estimating how much time the backup actually
#      has, so the algorithm's world model matches physical wall-clock time.
#   2. `rwsim/engine/simulator.py:_execute_request` adds it to the
#      simulator's physical hedged TTFT to emulate the real dispatch delay.
#
# The two consumers must use the same value. If the algorithm's estimate and
# the simulator's physical emulation diverge, simulator results stop being
# self-consistent.
#
# Calibrated 2026-05-21 from ~10,200 hedge dispatches across MiniMax
# real-eval runs (2026-05-13 -- 2026-05-20), measured via
# `backup_dispatch_overhead_ms` recorded by
# `experiments/real_evaluation/recorder.py`. Empirical distribution:
#   P50 = 0.31 ms, P90 = 5.41 ms, P95 = 6.61 ms, P99 = 17.96 ms,
#   max = 87.44 ms (single outlier, likely a cold connection).
# The value tracks roughly the empirical P90 — enough headroom for typical
# dispatches without being skewed by tail outliers.
#
# Caveats:
#   1. Calibration data is all MiniMax. Other providers with different
#      connection-reuse behaviour may warrant re-measurement.
#   2. Overhead is load-dependent: P90 ~0.4 ms at low load (sanity runs),
#      ~5 ms at BurstGPT cap10s load. The 5 ms value matches the high-load
#      operating point, which is the regime where hedging decisions matter.
#   3. Historical value was 50 ms (pre-data heuristic). See the commit that
#      introduced this module for the rationale change.
#
# To update: re-run the quantile analysis on fresh real-eval CSVs, edit the
# value below, and re-capture golden baselines.
DISPATCH_OVERHEAD_MS: float = 5.0


__all__ = ["DISPATCH_OVERHEAD_MS"]
