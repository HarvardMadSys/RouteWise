"""Microbenchmark for EmpiricalDistribution hot-path methods.

Run manually after touching ``rwsim/world/empirical.py``:

    python scripts/perf/bench_empirical_distribution.py

This is not a pytest test — perf numbers vary across CI runners and we
don't want a noisy assertion in the default suite. Compare the printed
numbers against the baseline and target values printed at the end.
"""

from __future__ import annotations

import time

import numpy as np

from rwsim.world.empirical import EmpiricalDistribution


def _bench_call(label: str, fn, n: int) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed_ns = (time.perf_counter() - t0) * 1e9 / n
    print(f"  {label:<24s} {elapsed_ns:8.0f} ns/call")
    return elapsed_ns


def main() -> None:
    rng = np.random.default_rng(0)
    samples = rng.exponential(scale=300.0, size=100_000)
    dist = EmpiricalDistribution(samples)

    n_iter = 50_000
    print(f"EmpiricalDistribution benchmark (N={dist._n}, iter={n_iter}):")

    sample_ns = _bench_call("sample(rng, 1)", lambda: dist.sample(rng, 1), n_iter)
    mean_ns = _bench_call("mean()", dist.mean, n_iter)
    p50_ns = _bench_call("p50()", dist.p50, n_iter)
    p99_ns = _bench_call("p99()", dist.p99, n_iter)
    cdf_ns = _bench_call("cdf(p50)", lambda: dist.cdf(dist.p50()), n_iter)

    print()
    print("Pre-1a baseline (measured 2026-05-04):")
    print("  sample(rng, 1)            ~3600 ns/call")
    print("  mean()                   ~36000 ns/call")
    print("  p99()                   ~378000 ns/call")
    print()
    print("Post-1a targets:")
    print("  sample < 1500, mean < 200, p99 < 500 ns/call")

    if sample_ns >= 1500 or mean_ns >= 200 or p99_ns >= 500 or cdf_ns >= 5000:
        print()
        print("WARNING: at least one method exceeded the post-1a budget.")


if __name__ == "__main__":
    main()
