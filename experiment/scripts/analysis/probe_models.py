#!/usr/bin/env python3
"""Lightweight probe-then-simulate script for multi-model evaluation.

Workflow:
  1. Discover providers for each model via OpenRouter auto-routing
  2. Probe each discovered provider to build latency profiles
  3. Solve LP for optimal provider mix
  4. Simulate OpenRouter Auto's inverse-square pricing baseline
  5. Compare and report which models show "faster AND cheaper" potential

Usage:
  python probe_models.py [--models MODEL1 MODEL2 ...] [--probes-per-provider 80]
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

# Add project root to path.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiment.strategies.online_latency_router import (
    OnlineLatencyRouter,
    ProviderProfile,
    solve_lp_with_fallback,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Default candidate models with many providers.
DEFAULT_MODELS = [
    "openai/gpt-oss-120b",
    "deepseek/deepseek-v3.2",
    "qwen/qwen3-235b-a22b-2507",
]

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
SHORT_PROBE_PROMPT = "Say hello in one word."
PROBE_MAX_TOKENS = 16


@dataclass
class ProviderStats:
    """Collected stats for one provider."""

    name: str
    ttft_samples: list[float] = field(default_factory=list)  # ms
    costs: list[float] = field(default_factory=list)  # USD
    errors: int = 0
    total: int = 0


@dataclass
class ModelResult:
    """Probe + simulation results for one model."""

    model_id: str
    providers: dict[str, ProviderStats] = field(default_factory=dict)
    lp_mix: dict[str, float] = field(default_factory=dict)
    lp_cost: float = 0.0
    lp_p50: float = 0.0
    lp_p99: float = 0.0
    auto_cost: float = 0.0
    auto_p50: float = 0.0
    auto_p99: float = 0.0
    lp_status: str = ""


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    return key


def send_probe(
    api_key: str,
    model: str,
    provider: str | None = None,
    timeout: float = 30.0,
) -> tuple[float, float, str, str]:
    """Send a single probe request. Returns (ttft_ms, cost_usd, provider, status)."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/HarvardSys/hybridInference",
        "X-Title": "hybridInference model probe",
    }
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": SHORT_PROBE_PROMPT}],
        "max_tokens": PROBE_MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if provider:
        payload["provider"] = {
            "order": [provider],
            "allow_fallbacks": False,
        }

    start = time.perf_counter()
    ttft_ms = -1.0
    cost = 0.0
    actual_provider = provider or "auto"

    try:
        resp = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=timeout,
        )
        resp.raise_for_status()

        first_token_seen = False
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8", errors="replace")
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Extract actual provider from first chunk.
            if not first_token_seen:
                actual_provider = (
                    chunk.get("provider", actual_provider)
                    or actual_provider
                )

            # Detect first content token for TTFT.
            choices = chunk.get("choices", [])
            if (
                not first_token_seen
                and choices
                and choices[0].get("delta", {}).get("content")
            ):
                ttft_ms = (time.perf_counter() - start) * 1000
                first_token_seen = True

            # Extract cost from usage in final chunk.
            usage = chunk.get("usage")
            if usage:
                cost = float(usage.get("cost", 0.0) or 0.0)

        resp.close()
        status = "success" if first_token_seen else "no_content"
    except requests.exceptions.Timeout:
        status = "timeout"
    except requests.exceptions.HTTPError as e:
        status = f"HTTP {e.response.status_code}"
    except Exception as e:
        status = f"error: {type(e).__name__}"

    return ttft_ms, cost, actual_provider, status


def discover_providers(
    api_key: str,
    model: str,
    n_discovery: int = 30,
    delay: float = 0.5,
) -> dict[str, ProviderStats]:
    """Discover providers by sending auto-routed requests."""
    log.info(f"[{model}] Discovering providers with {n_discovery} auto requests...")
    providers: dict[str, ProviderStats] = {}

    for i in range(n_discovery):
        ttft_ms, cost, actual_provider, status = send_probe(api_key, model)

        if actual_provider not in providers:
            providers[actual_provider] = ProviderStats(name=actual_provider)

        ps = providers[actual_provider]
        ps.total += 1
        if status == "success" and ttft_ms > 0:
            ps.ttft_samples.append(ttft_ms)
            ps.costs.append(cost)
        else:
            ps.errors += 1

        if (i + 1) % 10 == 0:
            log.info(
                f"  [{model}] Discovery {i + 1}/{n_discovery}, "
                f"found {len(providers)} providers: "
                f"{list(providers.keys())}"
            )
        time.sleep(delay)

    log.info(
        f"[{model}] Discovered {len(providers)} providers: "
        f"{list(providers.keys())}"
    )
    return providers


def probe_providers(
    api_key: str,
    model: str,
    providers: dict[str, ProviderStats],
    probes_per_provider: int = 50,
    delay: float = 0.5,
) -> None:
    """Send targeted probes to each discovered provider."""
    for prov_name, ps in providers.items():
        existing = len(ps.ttft_samples)
        needed = max(0, probes_per_provider - existing)
        if needed == 0:
            log.info(
                f"  [{model}/{prov_name}] Already have {existing} samples, skipping"
            )
            continue

        log.info(f"  [{model}/{prov_name}] Sending {needed} probes...")
        for i in range(needed):
            ttft_ms, cost, actual_provider, status = send_probe(
                api_key, model, provider=prov_name
            )

            ps.total += 1
            if status == "success" and ttft_ms > 0:
                ps.ttft_samples.append(ttft_ms)
                ps.costs.append(cost)
            else:
                ps.errors += 1

            if (i + 1) % 20 == 0:
                log.info(
                    f"    [{model}/{prov_name}] {i + 1}/{needed} done, "
                    f"error_rate={ps.errors / ps.total:.1%}"
                )

            # Early termination: skip provider if error rate > 80% after 10 probes.
            if ps.total >= 10 and ps.errors / ps.total > 0.8:
                log.warning(
                    f"    [{model}/{prov_name}] Error rate too high "
                    f"({ps.errors}/{ps.total}), skipping remaining probes"
                )
                break

            time.sleep(delay)

        error_rate = ps.errors / ps.total if ps.total > 0 else 1.0
        p50 = np.median(ps.ttft_samples) if ps.ttft_samples else float("inf")
        log.info(
            f"  [{model}/{prov_name}] Done: {len(ps.ttft_samples)} samples, "
            f"P50={p50:.0f}ms, error_rate={error_rate:.1%}"
        )


def build_profiles(
    providers: dict[str, ProviderStats],
) -> dict[str, ProviderProfile]:
    """Build ProviderProfile objects from collected samples."""
    profiles: dict[str, ProviderProfile] = {}
    now = time.time()
    for prov_name, ps in providers.items():
        profile = ProviderProfile(
            provider=prov_name,
            samples=[],
            error_samples=[],
            window_sec=3600,  # Use all samples (1h window).
        )
        for j, ttft_ms in enumerate(ps.ttft_samples):
            ts = now - len(ps.ttft_samples) + j
            profile.add_sample(ts, ttft_ms, error_type=None)
        for j in range(ps.errors):
            ts = now - ps.errors + j
            profile.add_sample(ts, -1.0, error_type="server_error")
        profiles[prov_name] = profile
    return profiles


def compute_costs(
    providers: dict[str, ProviderStats],
) -> dict[str, float]:
    """Compute average cost per request for each provider."""
    costs: dict[str, float] = {}
    for prov_name, ps in providers.items():
        if ps.costs:
            costs[prov_name] = float(np.mean(ps.costs))
        else:
            costs[prov_name] = 1e-6  # Fallback: near-zero.
    return costs


def simulate_inverse_square(
    providers: dict[str, ProviderStats],
    costs: dict[str, float],
) -> dict[str, float]:
    """Simulate OpenRouter Auto's inverse-square pricing distribution."""
    weights: dict[str, float] = {}
    for prov_name, cost in costs.items():
        ps = providers[prov_name]
        error_rate = ps.errors / ps.total if ps.total > 0 else 1.0
        if error_rate > 0.5:
            continue  # Skip broken providers.
        if cost > 0:
            weights[prov_name] = 1.0 / (cost**2)
        else:
            weights[prov_name] = 1e12  # Free provider gets massive weight.

    total = sum(weights.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in weights.items()}


def simulate_latency(
    mix: dict[str, float],
    providers: dict[str, ProviderStats],
    n_sim: int = 10000,
) -> tuple[float, float, float]:
    """Simulate P50, P99, cost from a routing mix."""
    if not mix:
        return float("inf"), float("inf"), float("inf")

    prov_names = list(mix.keys())
    prov_weights = [mix[p] for p in prov_names]

    latencies = []
    costs_sim = []

    for _ in range(n_sim):
        # Sample provider according to mix.
        idx = np.random.choice(len(prov_names), p=prov_weights)
        prov = prov_names[idx]
        ps = providers[prov]

        if not ps.ttft_samples:
            latencies.append(float("inf"))
            costs_sim.append(0.0)
            continue

        # Sample a latency from this provider's empirical distribution.
        # Include error probability.
        error_rate = ps.errors / ps.total if ps.total > 0 else 0.0
        if np.random.random() < error_rate:
            latencies.append(float("inf"))  # Error = SLO violation.
            costs_sim.append(np.mean(ps.costs) if ps.costs else 0.0)
        else:
            lat = np.random.choice(ps.ttft_samples)
            latencies.append(lat)
            costs_sim.append(np.mean(ps.costs) if ps.costs else 0.0)

    latencies_arr = np.array(latencies)
    finite = latencies_arr[np.isfinite(latencies_arr)]
    p50 = float(np.percentile(finite, 50)) if len(finite) > 0 else float("inf")
    p99 = float(np.percentile(finite, 99)) if len(finite) > 0 else float("inf")
    avg_cost = float(np.mean(costs_sim))

    return p50, p99, avg_cost


def evaluate_model(
    api_key: str,
    model: str,
    probes_per_provider: int,
    slo_sec: float,
    n_discovery: int,
) -> ModelResult:
    """Full probe + simulate pipeline for one model."""
    result = ModelResult(model_id=model)

    # Step 1: Discover providers.
    providers = discover_providers(
        api_key, model, n_discovery=n_discovery
    )
    if not providers:
        log.warning(f"[{model}] No providers found, skipping")
        return result

    # Step 2: Probe each provider.
    probe_providers(api_key, model, providers, probes_per_provider)
    result.providers = providers

    # Step 3: Build profiles and costs.
    profiles = build_profiles(providers)
    costs = compute_costs(providers)

    # Step 4: Solve LP for optimal mix.
    valid_providers = [
        p for p, ps in providers.items() if len(ps.ttft_samples) >= 5
    ]
    if not valid_providers:
        log.warning(f"[{model}] No providers with enough samples")
        return result

    lp_mix, lp_status = solve_lp_with_fallback(
        providers=valid_providers,
        profiles=profiles,
        costs=costs,
        slo_sec=slo_sec,
        current_time=time.time(),
    )
    result.lp_mix = lp_mix
    result.lp_status = lp_status

    # Step 5: Simulate LP mix performance.
    result.lp_p50, result.lp_p99, result.lp_cost = simulate_latency(
        lp_mix, providers
    )

    # Step 6: Simulate OpenRouter Auto (inverse-square).
    auto_mix = simulate_inverse_square(providers, costs)
    result.auto_p50, result.auto_p99, result.auto_cost = simulate_latency(
        auto_mix, providers
    )

    return result


def print_summary(results: list[ModelResult], slo_sec: float) -> None:
    """Print comparison table."""
    slo_ms = slo_sec * 1000

    print("\n" + "=" * 100)
    print("PROBE-THEN-SIMULATE RESULTS")
    print("=" * 100)

    for r in results:
        if not r.providers:
            print(f"\n{r.model_id}: NO PROVIDERS FOUND")
            continue

        # Per-provider summary.
        print(f"\n{'=' * 80}")
        print(f"Model: {r.model_id}")
        print(f"{'=' * 80}")
        print(f"  Providers discovered: {len(r.providers)}")
        print(f"  LP status: {r.lp_status}")
        print(f"  LP mix: {r.lp_mix}")

        print(f"\n  {'Provider':<15} {'Samples':>8} {'Err%':>6} {'P50ms':>8} "
              f"{'P99ms':>8} {'AvgCost':>10} {'SLO%':>6}")
        print(f"  {'-' * 70}")
        for prov_name, ps in sorted(
            r.providers.items(),
            key=lambda x: np.median(x[1].ttft_samples) if x[1].ttft_samples else 1e9,
        ):
            n = len(ps.ttft_samples)
            err = ps.errors / ps.total * 100 if ps.total > 0 else 100
            p50 = np.median(ps.ttft_samples) if ps.ttft_samples else float("inf")
            p99 = np.percentile(ps.ttft_samples, 99) if n >= 5 else float("inf")
            avg_cost = np.mean(ps.costs) * 1e6 if ps.costs else 0  # In micro-dollars.
            slo_ok = (
                sum(1 for t in ps.ttft_samples if t <= slo_ms) / n * 100
                if n > 0
                else 0
            )
            print(
                f"  {prov_name:<15} {n:>8} {err:>5.1f}% {p50:>7.0f} "
                f"{p99:>7.0f} ${avg_cost:>8.2f}u {slo_ok:>5.1f}%"
            )

        # Comparison.
        print(f"\n  {'Policy':<20} {'P50ms':>8} {'P99ms':>8} {'AvgCost':>12}")
        print(f"  {'-' * 52}")
        print(
            f"  {'LP-Mix (ours)':<20} {r.lp_p50:>7.0f} {r.lp_p99:>7.0f} "
            f"${r.lp_cost * 1e6:>10.2f}u"
        )
        print(
            f"  {'InvSq (OR Auto)':<20} {r.auto_p50:>7.0f} {r.auto_p99:>7.0f} "
            f"${r.auto_cost * 1e6:>10.2f}u"
        )

        # Verdict.
        faster = r.lp_p99 < r.auto_p99
        cheaper = r.lp_cost < r.auto_cost
        if faster and cheaper:
            verdict = "FASTER AND CHEAPER"
        elif faster:
            p99_ratio = r.auto_p99 / r.lp_p99 if r.lp_p99 > 0 else float("inf")
            cost_ratio = r.lp_cost / r.auto_cost if r.auto_cost > 0 else float("inf")
            verdict = f"FASTER ({p99_ratio:.1f}x P99), costs {cost_ratio:.1f}x"
        elif cheaper:
            verdict = "CHEAPER but slower"
        else:
            verdict = "WORSE on both dimensions"

        print(f"\n  >>> VERDICT: {verdict}")

    # Final ranking.
    print(f"\n{'=' * 100}")
    print("RANKING (best candidates for full experiment)")
    print("=" * 100)

    scored = []
    for r in results:
        if not r.lp_mix:
            continue
        # Score: lower is better. Combine P99 improvement and cost improvement.
        p99_ratio = (
            r.auto_p99 / r.lp_p99 if r.lp_p99 > 0 else 1.0
        )
        cost_ratio = (
            r.lp_cost / r.auto_cost if r.auto_cost > 0 else 1.0
        )
        # Positive score = good. Higher P99 improvement, lower cost ratio.
        score = p99_ratio / cost_ratio
        scored.append((r.model_id, score, p99_ratio, cost_ratio))

    scored.sort(key=lambda x: -x[1])
    for i, (mid, score, p99r, costr) in enumerate(scored, 1):
        flag = " <<<" if costr <= 1.0 else ""
        print(
            f"  {i}. {mid:<40} "
            f"P99 {p99r:.1f}x better, cost {costr:.2f}x{flag}"
        )


def save_results(
    results: list[ModelResult],
    output_dir: Path,
) -> None:
    """Save results to JSON for later analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for r in results:
        providers_data = {}
        for prov_name, ps in r.providers.items():
            providers_data[prov_name] = {
                "n_samples": len(ps.ttft_samples),
                "n_errors": ps.errors,
                "n_total": ps.total,
                "p50_ms": float(np.median(ps.ttft_samples))
                if ps.ttft_samples
                else None,
                "p99_ms": float(np.percentile(ps.ttft_samples, 99))
                if len(ps.ttft_samples) >= 5
                else None,
                "avg_cost": float(np.mean(ps.costs)) if ps.costs else None,
                "error_rate": ps.errors / ps.total if ps.total > 0 else None,
            }
        out.append(
            {
                "model_id": r.model_id,
                "providers": providers_data,
                "lp_mix": r.lp_mix,
                "lp_status": r.lp_status,
                "lp_p50": r.lp_p50,
                "lp_p99": r.lp_p99,
                "lp_cost": r.lp_cost,
                "auto_p50": r.auto_p50,
                "auto_p99": r.auto_p99,
                "auto_cost": r.auto_cost,
            }
        )

    outfile = output_dir / "probe_results.json"
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"Results saved to {outfile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe models on OpenRouter")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model IDs to probe",
    )
    parser.add_argument(
        "--probes-per-provider",
        type=int,
        default=80,
        help="Number of probes per provider (default: 80)",
    )
    parser.add_argument(
        "--n-discovery",
        type=int,
        default=30,
        help="Number of auto-routed requests for provider discovery (default: 30)",
    )
    parser.add_argument(
        "--slo",
        type=float,
        default=2.0,
        help="SLO threshold in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "experiment" / "results" / "probe_models"),
        help="Output directory for results",
    )
    args = parser.parse_args()

    api_key = get_api_key()
    results: list[ModelResult] = []

    for model in args.models:
        log.info(f"\n{'#' * 60}")
        log.info(f"Probing model: {model}")
        log.info(f"{'#' * 60}")

        result = evaluate_model(
            api_key=api_key,
            model=model,
            probes_per_provider=args.probes_per_provider,
            slo_sec=args.slo,
            n_discovery=args.n_discovery,
        )
        results.append(result)

    print_summary(results, args.slo)
    save_results(results, Path(args.output_dir))


if __name__ == "__main__":
    main()
