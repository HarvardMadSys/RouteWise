"""Pareto frontier comparison: original_lp (Phase 3) vs budget_range_p*.

Merges two existing budget-eval runs that share scenario/dataset/seed config
on unified_pool / synthetic:
  - lp_budget_range_legacy_compare_apr27: original_lp* + budget_range_p25/50/75
  - lp_budget_range_sanity_apr27:        budget_range_p0/100 (endpoint fill-in)

Plots three subplots (cost vs P50, cost vs P99, cost vs SLO violation), each
showing four curves:
  - original_lp family (3 points: none/hedge/explorer)
  - budget_range_p* (no hedge):      5 points, p in {0,25,50,75,100}
  - budget_range_p*_hedge:           5 points
  - budget_range_p*_explorer:        5 points

Usage:
    python -m experiments.paper_plots.plot_pareto_frontier
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# experiments/paper_plots/plot_pareto_frontier.py is two levels below repo root.
ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "outputs" / "lp_budget_range_legacy_compare_apr27" / "all_results.csv"
SANITY = ROOT / "outputs" / "lp_budget_range_sanity_apr27" / "all_results.csv"
OUT_DIR = ROOT / "outputs" / "pareto_frontier_apr27"


def load_merged() -> pd.DataFrame:
    legacy = pd.read_csv(LEGACY)
    sanity = pd.read_csv(SANITY)
    merged = pd.concat([legacy, sanity], ignore_index=True)
    # Filter to unified_pool, synthetic only.
    merged = merged[(merged["dataset"] == "synthetic") & (merged["scenario"] == "unified_pool")]
    # Drop budget_vhat family (not on the paper main path).
    merged = merged[~merged["variant"].str.startswith("budget_vhat")]
    # Deduplicate on variant (legacy and sanity may overlap).
    merged = merged.drop_duplicates(subset=["variant"], keep="first").reset_index(drop=True)
    return merged


# Hedge mode classification.
_RE_RANGE = re.compile(r"^budget_range_p(\d+)(?:_(hedge|explorer))?$")


def classify(variant: str) -> tuple[str, str, int | None]:
    """Return (family, hedge_mode, budget_p).

    family: "original_lp" or "budget_range"
    hedge_mode: "none" | "hedge" | "explorer"
    budget_p: int 0..100 for budget_range, else None
    """
    m = _RE_RANGE.match(variant)
    if m:
        p = int(m.group(1))
        mode = m.group(2) or "none"
        return ("budget_range", mode, p)
    if variant == "original_lp":
        return ("original_lp", "none", None)
    if variant == "original_lp_hedge":
        return ("original_lp", "hedge", None)
    if variant == "original_lp_explorer":
        return ("original_lp", "explorer", None)
    return ("other", "none", None)


def main() -> None:
    df = load_merged()
    parts = df["variant"].apply(classify)
    df["family"] = [p[0] for p in parts]
    df["hedge_mode"] = [p[1] for p in parts]
    df["budget_p"] = [p[2] for p in parts]
    df = df[df["family"] != "other"].reset_index(drop=True)

    # cost in $ / 1k req for readability.
    df["cost_per_1k"] = df["avg_cost_usd"] * 1000.0

    metrics = [
        ("p50_ms", "P50 TTFT (ms)"),
        ("p99_ms", "P99 TTFT (ms)"),
        ("slo_violation_rate", "SLO Violation Rate"),
    ]

    hedge_styles = {
        "none": {"linestyle": "-", "marker": "o"},
        "hedge": {"linestyle": "--", "marker": "s"},
        "explorer": {"linestyle": ":", "marker": "^"},
    }
    family_color = {"original_lp": "#d62728", "budget_range": "#1f77b4"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (metric, ylabel) in zip(axes, metrics):
        # Budget_range curves: connect by budget_p within each hedge mode.
        for mode in ("none", "hedge", "explorer"):
            sub = df[(df["family"] == "budget_range") & (df["hedge_mode"] == mode)].sort_values(
                "budget_p"
            )
            if sub.empty:
                continue
            style = hedge_styles[mode]
            ax.plot(
                sub["cost_per_1k"],
                sub[metric],
                color=family_color["budget_range"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=7,
                linewidth=1.6,
                label=f"budget_range_p* ({mode})",
                alpha=0.85,
            )
            # Annotate p values on the no-hedge curve only (avoid clutter).
            if mode == "none":
                for _, row in sub.iterrows():
                    ax.annotate(
                        f"p{row['budget_p']}",
                        (row["cost_per_1k"], row[metric]),
                        textcoords="offset points",
                        xytext=(6, 6),
                        fontsize=8,
                        color=family_color["budget_range"],
                    )

        # Original_lp: 3 standalone points.
        sub = df[df["family"] == "original_lp"]
        for _, row in sub.iterrows():
            mode = row["hedge_mode"]
            ax.scatter(
                row["cost_per_1k"],
                row[metric],
                color=family_color["original_lp"],
                marker=hedge_styles[mode]["marker"],
                s=110,
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
                label=f"original_lp ({mode})",
            )

        ax.set_xlabel("avg cost ($ per 1k requests)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{ylabel} vs cost — unified_pool (synthetic)")

    # Single legend (deduplicated) on the rightmost axis.
    handles, labels = axes[0].get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, lab in zip(handles, labels):
        seen.setdefault(lab, h)
    fig.legend(
        list(seen.values()),
        list(seen.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        "Pareto frontier: original_lp (Phase 3) vs budget_range_p* (current main)",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "pareto_frontier.pdf"
    out_png = OUT_DIR / "pareto_frontier.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Numeric table side-by-side: anchor at original_lp_hedge cost.
    table_cols = ["variant", "family", "hedge_mode", "budget_p", "p50_ms", "p99_ms",
                  "slo_violation_rate", "cost_per_1k"]
    table = df[table_cols].sort_values(["family", "hedge_mode", "budget_p"]).reset_index(drop=True)
    csv_path = OUT_DIR / "pareto_table.csv"
    table.to_csv(csv_path, index=False, float_format="%.4f")

    print(f"[ok] wrote {out_pdf}")
    print(f"[ok] wrote {out_png}")
    print(f"[ok] wrote {csv_path}")
    print()
    print("Pareto-relevant points (sorted by cost):")
    print(table.sort_values("cost_per_1k").to_string(index=False))


if __name__ == "__main__":
    main()
