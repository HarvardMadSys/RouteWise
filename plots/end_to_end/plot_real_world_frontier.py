"""Plot real-world end-to-end frontier results for the paper.

Input is a real-eval output directory with one subdirectory per policy and a
``requests.csv`` file inside each policy directory. The script aggregates the
raw per-request logs into paper metrics and emits:

* one single-panel cost vs. mean TTFT figure;
* one single-panel cost vs. SLO-violation figure; and
* one single-panel TTFT CDF figure; and
* a LaTeX table-row fragment consumed by ``paper/sections/05-evaluation.tex``.

Example:

    .venv/bin/python -m plots.end_to_end.plot_real_world_frontier \
      --input-dir outputs/real_eval/real_eval_minimax_m25_burstgpt_cap10s_rwp_sweep_20260514_001724_minimax_m25_burstgpt_cap10s_rwp_sweep_slo3s \
      --mean-ttft-out ../paper/figures/real_world_partial_mean_ttft_placeholder.pdf \
      --slo-out ../paper/figures/real_world_partial_slo_placeholder.pdf \
      --table-out ../paper/tables/real_world_frontier_placeholder_rows.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from plots.end_to_end.frontier_plotting import (
    CdfSeries,
    DEFAULT_BASELINE_ORDER,
    FrontierPoint,
    MixRow,
    MixSegment,
    PROVIDER_COLOR_CYCLE,
    PROVIDER_MIX_COLORS,
    plot_mean_ttft_frontier,
    plot_slo_frontier,
    plot_stacked_mix,
    plot_ttft_cdf as plot_common_ttft_cdf,
    policy_plot_label as common_policy_plot_label,
)
from plots.style import apply_style


ROUTEWISE_PREFIX = "budget_range_p"
ROUTEWISE_HEDGE_SUFFIX = "_hedge"
ROUTEWISE_FRONTIER_PLOT_ALPHA = 0.25
DEFAULT_FIGURE_POLICIES = (
    "budget_range_p25_hedge",
    "greedy_cost",
    "greedy_latency",
    "or_auto",
    "or_sort_cost",
    "or_sort_latency",
    "random",
)
BASELINE_ORDER = DEFAULT_BASELINE_ORDER
POLICY_LABELS = {
    "greedy_cost": "Greedy-cost",
    "greedy_latency": "Greedy-latency",
    "random": "Random",
    "or_auto": "OpenRouter auto",
    "or_sort_cost": "OpenRouter sort=price",
    "or_sort_latency": "OpenRouter sort=latency",
}
DEFAULT_PROVIDER_MIX_POLICIES = DEFAULT_FIGURE_POLICIES
DEFAULT_CDF_POLICIES = DEFAULT_FIGURE_POLICIES


@dataclass(frozen=True)
class PolicySummary:
    policy: str
    label: str
    alpha: float | None
    n: int
    span_hours: float
    total_cost_usd: float
    billed_cost_usd: float
    fixed_cost_usd: float
    physical_cost_usd: float
    mean_cost_usd: float
    ttft_mean_ms: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    e2e_mean_ms: float
    e2e_p99_ms: float
    slo_violation_rate: float
    success_rate: float
    rate_limited: int
    hedge_rate: float
    hedge_backup_win_rate: float
    tier_mix: dict[str, float]


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    label: str
    tier: str
    input_price_per_m: float | None
    cached_input_price_per_m: float | None
    output_price_per_m: float | None
    fixed_cost_usd_override: float | None
    quota_requests: int | None
    quota_window_sec: float | None
    concurrency_limit: int | None


@dataclass(frozen=True)
class ProviderDiagnostics:
    provider: str
    label: str
    tier: str
    n: int
    share: float
    mean_ttft_ms: float
    input_price_per_m: float | None
    cached_input_price_per_m: float | None
    output_price_per_m: float | None
    fixed_cost_usd_override: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Real-eval run directory containing policy subdirectories.",
    )
    parser.add_argument(
        "--mean-ttft-out",
        type=Path,
        default=Path("../paper/figures/real_world_partial_mean_ttft_placeholder.pdf"),
        help="Output path for the cost-vs-mean-TTFT PDF figure.",
    )
    parser.add_argument(
        "--slo-out",
        type=Path,
        default=Path("../paper/figures/real_world_partial_slo_placeholder.pdf"),
        help="Output path for the cost-vs-SLO-violation PDF figure.",
    )
    parser.add_argument(
        "--table-out",
        type=Path,
        default=Path("../paper/tables/real_world_frontier_placeholder_rows.tex"),
        help="Output path for the LaTeX table-row fragment.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional output path for machine-readable aggregated metrics.",
    )
    parser.add_argument(
        "--provider-mix-out",
        type=Path,
        default=None,
        help="Optional output path for a provider-mix stacked-bar PDF.",
    )
    parser.add_argument(
        "--provider-latency-out",
        type=Path,
        default=None,
        help="Optional output path for a provider mean-TTFT bar PDF.",
    )
    parser.add_argument(
        "--cdf-out",
        type=Path,
        default=None,
        help="Optional output path for a TTFT CDF PDF.",
    )
    parser.add_argument(
        "--cdf-policies",
        nargs="+",
        default=None,
        help=(
            "Optional exact policy directory names to include in the TTFT CDF. "
            "Defaults to representative RouteWise and OpenRouter policies."
        ),
    )
    parser.add_argument(
        "--provider-diagnostics-table-out",
        type=Path,
        default=None,
        help="Optional LaTeX rows for provider price/latency diagnostics.",
    )
    parser.add_argument(
        "--provider-summary-out",
        type=Path,
        default=None,
        help="Optional JSON provider diagnostics output.",
    )
    parser.add_argument(
        "--provider-mix-policies",
        nargs="+",
        default=None,
        help=(
            "Optional exact policy directory names to include in the "
            "provider-mix figure. Other outputs still use --policies."
        ),
    )
    parser.add_argument("--slo-ms", type=float, default=3000.0)
    parser.add_argument(
        "--billing-duration-sec",
        type=float,
        default=28800.0,
        help="Billing horizon used to prorate subscription fixed costs.",
    )
    parser.add_argument(
        "--minimax-fixed-cost-usd",
        type=float,
        default=0.2381,
        help="MiniMax Plus prorated fixed cost for one policy run.",
    )
    parser.add_argument(
        "--featherless-monthly-cost-usd",
        type=float,
        default=25.0,
        help="Featherless monthly fixed cost used for prorating.",
    )
    parser.add_argument(
        "--fixed-cost-non-or",
        type=float,
        default=None,
        help=(
            "Fixed cost added to non-OpenRouter policies. By default the script "
            "adds the MiniMax fixed-cost override plus a Featherless monthly fee "
            "prorated to --billing-duration-sec. Pass 0 for billed replay charges only."
        ),
    )
    parser.add_argument(
        "--include-random",
        action="store_true",
        help="Include the random baseline in the paper table and figure.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=None,
        help=(
            "Optional exact policy directory names to include. When set, the "
            "output preserves this order."
        ),
    )
    parser.add_argument(
        "--x-label",
        default="Total cost ($)",
        help="X-axis label for generated frontier figures.",
    )
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    pos = (len(values) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[int(pos)]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def truthy(value: str | None) -> bool:
    return value in {"1", "true", "True", "yes", "YES"}


def parse_alpha(policy: str) -> float | None:
    if not policy.startswith(ROUTEWISE_PREFIX):
        return None
    raw = policy[len(ROUTEWISE_PREFIX) :]
    if raw.endswith(ROUTEWISE_HEDGE_SUFFIX):
        raw = raw[: -len(ROUTEWISE_HEDGE_SUFFIX)]
    if raw == "100":
        return 1.0
    try:
        return float(raw) / 100.0
    except ValueError:
        return None


def table_label(policy: str, alpha: float | None) -> str:
    if alpha is not None:
        hedge_suffix = " + hedge" if policy.endswith(ROUTEWISE_HEDGE_SUFFIX) else ""
        return rf"\sysname{{}} ($\alpha={alpha:g}${hedge_suffix})"
    return POLICY_LABELS.get(policy, policy.replace("_", r"\_"))


def fixed_cost_for_policy(policy: str, args: argparse.Namespace) -> float:
    if policy.startswith("or_"):
        return 0.0
    if args.fixed_cost_non_or is not None:
        return args.fixed_cost_non_or
    featherless = (
        args.featherless_monthly_cost_usd
        * (args.billing_duration_sec / 86400.0)
        / 30.0
    )
    return args.minimax_fixed_cost_usd + featherless


def read_summary(policy_dir: Path, args: argparse.Namespace) -> PolicySummary:
    policy = policy_dir.name
    alpha = parse_alpha(policy)
    rows = list(csv.DictReader((policy_dir / "requests.csv").open(newline="")))
    if not rows:
        raise ValueError(f"{policy_dir}: requests.csv has no rows")

    timestamps = [float(row["ts"]) for row in rows if row.get("ts")]
    successes = [
        row
        for row in rows
        if row.get("status") == "success"
        and row.get("ttft_ms") not in {"", None}
        and float(row["ttft_ms"]) >= 0.0
    ]
    ttft = [float(row["ttft_ms"]) for row in successes]
    e2e = [
        float(row["e2e_ms"])
        for row in successes
        if row.get("e2e_ms") not in {"", None}
    ]

    billed_cost = sum(float(row.get("billed_cost_usd") or 0.0) for row in rows)
    physical_cost = sum(float(row.get("physical_cost_usd") or 0.0) for row in rows)
    fixed_cost = fixed_cost_for_policy(policy, args)
    total_cost = billed_cost + fixed_cost
    slo_violations = sum(
        1
        for row in rows
        if row.get("status") != "success"
        or (
            row.get("ttft_ms") not in {"", None}
            and float(row["ttft_ms"]) > args.slo_ms
        )
    )
    hedge_count = sum(1 for row in rows if truthy(row.get("hedge_triggered")))
    backup_wins = sum(1 for row in rows if row.get("hedge_winner") == "backup")
    tier_counts: dict[str, int] = {}
    for row in rows:
        tier = row.get("tier") or "openrouter"
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    return PolicySummary(
        policy=policy,
        label=table_label(policy, alpha),
        alpha=alpha,
        n=len(rows),
        span_hours=(max(timestamps) - min(timestamps)) / 3600.0 if timestamps else 0.0,
        total_cost_usd=total_cost,
        billed_cost_usd=billed_cost,
        fixed_cost_usd=fixed_cost,
        physical_cost_usd=physical_cost,
        mean_cost_usd=total_cost / len(rows),
        ttft_mean_ms=mean(ttft) if ttft else float("nan"),
        ttft_p50_ms=percentile(ttft, 50.0),
        ttft_p90_ms=percentile(ttft, 90.0),
        ttft_p95_ms=percentile(ttft, 95.0),
        ttft_p99_ms=percentile(ttft, 99.0),
        e2e_mean_ms=mean(e2e) if e2e else float("nan"),
        e2e_p99_ms=percentile(e2e, 99.0),
        slo_violation_rate=slo_violations / len(rows),
        success_rate=len(successes) / len(rows),
        rate_limited=sum(1 for row in rows if truthy(row.get("rate_limited"))),
        hedge_rate=hedge_count / len(rows),
        hedge_backup_win_rate=backup_wins / hedge_count if hedge_count else 0.0,
        tier_mix={tier: count / len(rows) for tier, count in sorted(tier_counts.items())},
    )


def load_inventory(path: Path) -> tuple[dict[str, ProviderInfo], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    providers: dict[str, ProviderInfo] = {}
    hint_to_name: dict[str, str] = {}
    for raw in payload["providers"]:
        name = raw["name"]
        hint = raw.get("provider_hint")
        if hint:
            hint_to_name[hint] = name
        providers[name] = ProviderInfo(
            name=name,
            label=provider_label(name),
            tier=raw.get("tier") or "api",
            input_price_per_m=raw.get("input_price_per_m"),
            cached_input_price_per_m=raw.get("cached_input_price_per_m"),
            output_price_per_m=raw.get("output_price_per_m"),
            fixed_cost_usd_override=raw.get("fixed_cost_usd_override"),
            quota_requests=raw.get("quota_requests"),
            quota_window_sec=raw.get("quota_window_sec"),
            concurrency_limit=raw.get("concurrency_limit"),
        )
    return providers, hint_to_name


def inventory_path_for_run(input_dir: Path) -> Path:
    for policy_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        args_path = policy_dir / "args.json"
        if not args_path.exists():
            continue
        payload = json.loads(args_path.read_text(encoding="utf-8"))
        inventory = payload.get("inventory")
        if inventory:
            return Path(inventory)
    raise FileNotFoundError(f"{input_dir}: could not find inventory in args.json")


def provider_label(provider: str) -> str:
    labels = {
        "MiniMax_Plus_SQ": r"$\mathcal{P}_Q$ MiniMax",
        "Featherless_SC": r"$\mathcal{P}_C$ Featherless",
    }
    if provider in labels:
        return labels[provider]
    if provider.startswith("OR_"):
        return provider.removeprefix("OR_")
    return provider.replace("_", " ")


def provider_mix_legend_label(provider: str) -> str:
    labels = {
        "MiniMax_Plus_SQ": r"$\mathcal{P}_Q$",
        "Featherless_SC": r"$\mathcal{P}_C$",
        "OR_DeepInfra": "DeepInfra",
        "OR_AtlasCloud": "Atlas",
        "OR_AkashML": "Akash",
        "OR_WandB": "W&B",
        "OR_Chutes": "Chutes",
        "OR_Novita": "Novita",
        "OR_Phala": "Phala",
        "OR_SiliconFlow": "SFlow",
    }
    return labels.get(provider, provider_label(provider))


def normalize_provider(
    raw_provider: str,
    *,
    hint_to_name: dict[str, str],
) -> str:
    if not raw_provider:
        return "unknown"
    if "@" in raw_provider:
        left, right = raw_provider.split("@", 1)
        if left.startswith("__"):
            return hint_to_name.get(right, f"OR_{right}")
        return left
    return raw_provider


def policy_plot_label(policy: str) -> str:
    alpha = parse_alpha(policy)
    return common_policy_plot_label(
        policy,
        alpha=alpha,
        hedging=policy.endswith(ROUTEWISE_HEDGE_SUFFIX),
    )


def read_policy_rows(policy_dir: Path) -> list[dict[str, str]]:
    return list(csv.DictReader((policy_dir / "requests.csv").open(newline="")))


def selected_policy_dirs(args: argparse.Namespace) -> list[Path]:
    if args.policies:
        return [args.input_dir / policy for policy in args.policies]
    dirs = [
        policy_dir
        for policy_dir in sorted(args.input_dir.iterdir())
        if policy_dir.is_dir() and (policy_dir / "requests.csv").exists()
    ]
    if not args.include_random:
        dirs = [policy_dir for policy_dir in dirs if policy_dir.name != "random"]
    return dirs


def selected_provider_mix_dirs(args: argparse.Namespace) -> list[Path]:
    if args.provider_mix_policies:
        return [args.input_dir / policy for policy in args.provider_mix_policies]
    policies = list(DEFAULT_PROVIDER_MIX_POLICIES)
    if not args.include_random:
        policies = [policy for policy in policies if policy != "random"]
    return [args.input_dir / policy for policy in policies]


def provider_mix_by_policy(
    args: argparse.Namespace,
    hint_to_name: dict[str, str],
) -> tuple[dict[str, Counter[str]], Counter[str]]:
    per_policy: dict[str, Counter[str]] = {}
    total_counts: Counter[str] = Counter()
    for policy_dir in selected_provider_mix_dirs(args):
        if not (policy_dir / "requests.csv").exists():
            raise FileNotFoundError(policy_dir / "requests.csv")
        counts: Counter[str] = Counter()
        for row in read_policy_rows(policy_dir):
            provider = normalize_provider(
                row.get("actual_provider") or row.get("primary_provider") or "",
                hint_to_name=hint_to_name,
            )
            counts[provider] += 1
            total_counts[provider] += 1
        per_policy[policy_dir.name] = counts
    return per_policy, total_counts


def plot_provider_mix(
    args: argparse.Namespace,
    _providers: dict[str, ProviderInfo],
    hint_to_name: dict[str, str],
    output_path: Path,
) -> None:
    per_policy, total_counts = provider_mix_by_policy(args, hint_to_name)
    providers_to_plot = [provider for provider, _ in total_counts.most_common()]
    segments = [
        MixSegment(
            key=provider,
            label=provider_mix_legend_label(provider),
            color=PROVIDER_MIX_COLORS.get(
                provider,
                PROVIDER_COLOR_CYCLE[idx % len(PROVIDER_COLOR_CYCLE)],
            ),
        )
        for idx, provider in enumerate(providers_to_plot)
    ]
    mix_rows: list[MixRow] = []
    for policy, counts in per_policy.items():
        total = sum(counts.values())
        shares = {
            provider: (counts.get(provider, 0) / total) if total else 0.0
            for provider in providers_to_plot
        }
        mix_rows.append(MixRow(label=policy_plot_label(policy), shares=shares))
    plot_stacked_mix(
        mix_rows,
        segments,
        output_path,
        legend_ncols=5,
        legend_fontsize=5.2,
    )


def provider_diagnostics(
    args: argparse.Namespace,
    providers: dict[str, ProviderInfo],
    hint_to_name: dict[str, str],
) -> list[ProviderDiagnostics]:
    ttft_by_provider: dict[str, list[float]] = defaultdict(list)
    counts: Counter[str] = Counter()
    total = 0
    for policy_dir in selected_policy_dirs(args):
        for row in read_policy_rows(policy_dir):
            provider = normalize_provider(
                row.get("actual_provider") or row.get("primary_provider") or "",
                hint_to_name=hint_to_name,
            )
            counts[provider] += 1
            total += 1
            if row.get("status") == "success" and row.get("ttft_ms"):
                ttft_by_provider[provider].append(float(row["ttft_ms"]))
    seen = set(providers) | set(counts)
    diagnostics: list[ProviderDiagnostics] = []
    for provider in sorted(seen, key=lambda name: provider_sort_key(name, counts, providers)):
        info = providers.get(provider)
        ttft = ttft_by_provider.get(provider, [])
        diagnostics.append(
            ProviderDiagnostics(
                provider=provider,
                label=info.label if info else provider_label(provider),
                tier=info.tier if info else "api",
                n=counts.get(provider, 0),
                share=(counts.get(provider, 0) / total) if total else 0.0,
                mean_ttft_ms=mean(ttft) if ttft else float("nan"),
                input_price_per_m=info.input_price_per_m if info else None,
                cached_input_price_per_m=(
                    info.cached_input_price_per_m if info else None
                ),
                output_price_per_m=info.output_price_per_m if info else None,
                fixed_cost_usd_override=info.fixed_cost_usd_override if info else None,
            )
        )
    return diagnostics


def provider_sort_key(
    provider: str,
    counts: Counter[str],
    providers: dict[str, ProviderInfo],
) -> tuple[int, int, str]:
    tier_order = {"quota": 0, "concurrency": 1, "api": 2}
    tier = providers.get(provider).tier if provider in providers else "api"
    return (tier_order.get(tier, 9), -counts.get(provider, 0), provider)


def format_price(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.3g}"


def format_fixed(diagnostic: ProviderDiagnostics) -> str:
    if diagnostic.fixed_cost_usd_override is not None:
        return f"\\${diagnostic.fixed_cost_usd_override:.3f}/run"
    if diagnostic.tier == "concurrency":
        return "\\$25/mo"
    return "--"


def format_latency_ms(value: float) -> str:
    if math.isnan(value):
        return "--"
    return f"{value / 1000.0:.2f}"


def write_provider_diagnostics_table(
    diagnostics: list[ProviderDiagnostics],
    path: Path,
) -> None:
    lines = []
    for item in diagnostics:
        lines.append(
            f"{item.label} & "
            f"{item.tier} & "
            f"{format_price(item.input_price_per_m)} & "
            f"{format_price(item.cached_input_price_per_m)} & "
            f"{format_price(item.output_price_per_m)} & "
            f"{format_fixed(item)} & "
            f"{item.share * 100.0:.1f}\\% & "
            f"{format_latency_ms(item.mean_ttft_ms)} \\\\"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_provider_latency(
    diagnostics: list[ProviderDiagnostics],
    output_path: Path,
) -> None:
    apply_style("paper")
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "figure.figsize": (3.35, 2.2),
            "savefig.pad_inches": 0.01,
        }
    )
    items = [
        item
        for item in diagnostics
        if item.n > 0 and not math.isnan(item.mean_ttft_ms)
    ]
    items = sorted(items, key=lambda item: item.mean_ttft_ms)
    labels = [provider_mix_legend_label(item.provider) for item in items]
    values = [item.mean_ttft_ms / 1000.0 for item in items]
    colors = [
        PROVIDER_MIX_COLORS.get(
            item.provider,
            PROVIDER_COLOR_CYCLE[idx % len(PROVIDER_COLOR_CYCLE)],
        )
        for idx, item in enumerate(items)
    ]

    fig, ax = plt.subplots(figsize=(3.35, 2.2), constrained_layout=False)
    y = list(range(len(items)))
    ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.35, height=0.72)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean TTFT (s)")
    ax.grid(axis="x", color="#9a9a9a", alpha=0.28, linewidth=0.5)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.29, right=0.99, bottom=0.16, top=0.97)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def cdf_policy_label(policy: str) -> str:
    return policy_plot_label(policy)


def policy_ttft_samples(policy_dir: Path) -> list[float]:
    samples: list[float] = []
    for row in read_policy_rows(policy_dir):
        raw = row.get("ttft_ms")
        if raw in {"", None}:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value) and value >= 0.0:
            samples.append(value)
    return sorted(samples)


def selected_cdf_dirs(args: argparse.Namespace) -> list[Path]:
    policies = list(args.cdf_policies or DEFAULT_CDF_POLICIES)
    if not args.include_random:
        policies = [policy for policy in policies if policy != "random"]
    dirs = [args.input_dir / policy for policy in policies]
    missing = [path.name for path in dirs if not (path / "requests.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"{args.input_dir}: missing CDF policy outputs: {missing}"
        )
    return dirs


def plot_ttft_cdf(args: argparse.Namespace, output_path: Path) -> None:
    series: list[CdfSeries] = []
    p99_values: list[float] = []
    p99_values_for_axis: list[float] = []
    for policy_dir in selected_cdf_dirs(args):
        samples = policy_ttft_samples(policy_dir)
        if not samples:
            continue
        policy = policy_dir.name
        n = len(samples)
        y = [(idx + 1) / n for idx in range(n)]
        alpha = parse_alpha(policy)
        p99 = percentile(samples, 99.0)
        p99_values.append(p99)
        if policy != "random":
            p99_values_for_axis.append(p99)
        series.append(
            CdfSeries(
                policy=policy,
                label=cdf_policy_label(policy),
                x_ms=samples,
                y_values=y,
                p99_ms=p99,
                alpha=alpha,
                include_in_axis=policy != "random",
                drawstyle="steps-post",
            )
        )
    if not series:
        raise ValueError(f"{args.input_dir}: no valid TTFT samples for CDF")

    slo_sec = args.slo_ms / 1000.0
    axis_p99_values = p99_values_for_axis or p99_values
    x_max = max(slo_sec * 4.0, max(axis_p99_values) / 1000.0 * 1.05)
    plot_common_ttft_cdf(series, output_path, slo_sec=slo_sec, x_max_sec=x_max)


def collect_summaries(args: argparse.Namespace) -> list[PolicySummary]:
    if not args.input_dir.exists():
        raise FileNotFoundError(args.input_dir)
    selected = set(args.policies or [])
    summaries = []
    for policy_dir in sorted(path for path in args.input_dir.iterdir() if path.is_dir()):
        if not (policy_dir / "requests.csv").exists():
            continue
        if selected and policy_dir.name not in selected:
            continue
        if policy_dir.name == "random" and not args.include_random:
            continue
        summaries.append(read_summary(policy_dir, args))
    if not summaries:
        raise ValueError(f"{args.input_dir}: found no requests.csv files")
    if args.policies:
        by_policy = {summary.policy: summary for summary in summaries}
        missing = [policy for policy in args.policies if policy not in by_policy]
        if missing:
            raise ValueError(f"{args.input_dir}: missing requested policies: {missing}")
        return [by_policy[policy] for policy in args.policies]
    return sorted(summaries, key=sort_key)


def sort_key(summary: PolicySummary) -> tuple[int, float]:
    if summary.policy == "greedy_cost":
        return (0, 0.0)
    if summary.policy == "greedy_latency":
        return (0, 1.0)
    if summary.alpha is not None:
        return (1, summary.alpha)
    baseline_rank = (
        float(BASELINE_ORDER.index(summary.policy))
        if summary.policy in BASELINE_ORDER
        else 99.0
    )
    return (2, baseline_rank)


def write_table_rows(
    summaries: list[PolicySummary],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for s in summaries:
        hedge_stats = (
            f"{100.0 * s.hedge_rate:.2f} / "
            f"{100.0 * s.hedge_backup_win_rate:.2f}\\%"
            if s.policy.endswith(ROUTEWISE_HEDGE_SUFFIX)
            else "--"
        )
        lines.append(
            f"{s.label} & "
            f"{s.total_cost_usd:.3f} & "
            f"{s.ttft_mean_ms / 1000.0:.2f} & "
            f"{s.ttft_p50_ms / 1000.0:.2f} & "
            f"{s.ttft_p90_ms / 1000.0:.2f} & "
            f"{s.ttft_p95_ms / 1000.0:.2f} & "
            f"{s.ttft_p99_ms / 1000.0:.2f} & "
            f"{100.0 * s.slo_violation_rate:.2f}\\% & "
            f"{hedge_stats} \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def frontier_points(summaries: list[PolicySummary]) -> list[FrontierPoint]:
    return [
        FrontierPoint(
            policy=summary.policy,
            label=policy_plot_label(summary.policy),
            alpha=summary.alpha,
            hedging=summary.policy.endswith(ROUTEWISE_HEDGE_SUFFIX),
            total_cost_usd=summary.total_cost_usd,
            mean_ttft_ms=summary.ttft_mean_ms,
            slo_violation_rate=summary.slo_violation_rate,
            p99_ms=summary.ttft_p99_ms,
            hedge_rate=summary.hedge_rate,
        )
        for summary in summaries
    ]


def plot_metric_frontier(
    summaries: list[PolicySummary],
    path: Path,
    *,
    attr: str,
    ylabel: str,
    xlabel: str,
    title: str | None = None,
) -> None:
    if title:
        raise ValueError("shared frontier plots do not support per-panel titles")
    points = frontier_points(summaries)
    kwargs = {
        "xlabel": xlabel,
        "routewise_alphas": (ROUTEWISE_FRONTIER_PLOT_ALPHA,),
        "baseline_order": BASELINE_ORDER,
    }
    if attr == "ttft_mean_ms":
        plot_mean_ttft_frontier(points, path, **kwargs)
    elif attr == "slo_violation_rate":
        plot_slo_frontier(points, path, **kwargs)
    else:
        raise ValueError(f"unsupported frontier metric: {attr}")


def write_summary(summaries: list[PolicySummary], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(summary) for summary in summaries], indent=2) + "\n",
        encoding="utf-8",
    )


def write_provider_summary(
    diagnostics: list[ProviderDiagnostics],
    path: Path | None,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(item) for item in diagnostics], indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    summaries = collect_summaries(args)
    inventory_path = inventory_path_for_run(args.input_dir)
    providers, hint_to_name = load_inventory(inventory_path)
    plot_metric_frontier(
        summaries,
        args.mean_ttft_out,
        attr="ttft_mean_ms",
        ylabel="Mean TTFT (s)",
        xlabel=args.x_label,
    )
    plot_metric_frontier(
        summaries,
        args.slo_out,
        attr="slo_violation_rate",
        ylabel="SLO violations (%)",
        xlabel=args.x_label,
    )
    write_table_rows(summaries, args.table_out)
    write_summary(summaries, args.summary_out)
    diagnostics: list[ProviderDiagnostics] | None = None
    if args.cdf_out is not None:
        plot_ttft_cdf(args, args.cdf_out)
        print(f"wrote {args.cdf_out}")
    if args.provider_mix_out is not None:
        plot_provider_mix(args, providers, hint_to_name, args.provider_mix_out)
        print(f"wrote {args.provider_mix_out}")
    if (
        args.provider_diagnostics_table_out is not None
        or args.provider_summary_out is not None
        or args.provider_latency_out is not None
    ):
        diagnostics = provider_diagnostics(args, providers, hint_to_name)
    if args.provider_latency_out is not None and diagnostics is not None:
        plot_provider_latency(diagnostics, args.provider_latency_out)
        print(f"wrote {args.provider_latency_out}")
    if args.provider_diagnostics_table_out is not None and diagnostics is not None:
        write_provider_diagnostics_table(
            diagnostics,
            args.provider_diagnostics_table_out,
        )
        print(f"wrote {args.provider_diagnostics_table_out}")
    if args.provider_summary_out is not None and diagnostics is not None:
        write_provider_summary(diagnostics, args.provider_summary_out)
        print(f"wrote {args.provider_summary_out}")
    print(f"wrote {args.mean_ttft_out}")
    print(f"wrote {args.slo_out}")
    print(f"wrote {args.table_out}")
    if args.summary_out is not None:
        print(f"wrote {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
