#!/usr/bin/env python3
"""Reconstruct LP decision dynamics from a completed Phase 5 run.

This script replays the shared OnlineLatencyRouter used by `lp_mix` and
`smart_hedge` using:
  1. The final checkpoint's router sample store (includes warmup, probing, and
     request-derived samples with original timestamps), and
  2. The chronological request stream from `evaluation_log.csv`.

Goal:
  Explain why Alibaba-heavy windows occur by reconstructing:
  - raw LP weights
  - smoothed sampler weights
  - routed providers under SWRR
  - per-provider CDF at the SLO threshold

Usage:
    source .venv/bin/activate
    python -m experiment.scripts.analysis.reconstruct_lp_decisions
"""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_EVAL_LOG = Path(
    "/Users/realtmxi/Desktop/NSDI2027_RouteWise/"
    "experiment/results/phase5_qwen3_7d_clean/"
    "run_20260410_171625/evaluation_log.csv"
)
DEFAULT_PRICING = Path(
    "/Users/realtmxi/Desktop/NSDI2027_RouteWise/"
    "experiment/data/openrouter_qwen3_235b.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/realtmxi/Desktop/NSDI2027_RouteWise/"
    "experiment/results/phase5_qwen3_7d_clean/"
    "run_20260410_171624/analysis/lp_reconstruction"
)

TARGET_POLICIES = ("lp_mix", "smart_hedge")
FOCUS_PROVIDERS = ("WandB", "Alibaba", "Google")
SLO_SEC = 2.0
WINDOW_SEC = 15 * 60
LP_UPDATE_INTERVAL_SEC = 300.0
WEIGHT_SMOOTHING = 0.3
OBJECTIVE_INPUT_TOKENS = 200
OBJECTIVE_OUTPUT_TOKENS = 300


def load_router_module():
    """Load online_latency_router.py without importing experiment.strategies package."""
    module_path = (
        Path(__file__).resolve().parents[2] / "strategies" / "online_latency_router.py"
    )
    spec = importlib.util.spec_from_file_location(
        "online_latency_router_reconstruct",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_ROUTER_MODULE = load_router_module()
FailureMode = _ROUTER_MODULE.FailureMode
OnlineLatencyRouter = _ROUTER_MODULE.OnlineLatencyRouter
pre_filter = _ROUTER_MODULE.pre_filter
solve_lp_with_fallback = _ROUTER_MODULE.solve_lp_with_fallback

def load_pricing(
    path: Path,
    input_tokens: int = OBJECTIVE_INPUT_TOKENS,
    output_tokens: int = OBJECTIVE_OUTPUT_TOKENS,
) -> dict[str, float]:
    """Convert per-1M-token pricing to per-request LP costs."""
    with path.open() as f:
        raw = json.load(f)

    costs = {}
    for provider, prices in raw.items():
        input_price, output_price = prices
        costs[provider] = (
            input_price * input_tokens + output_price * output_tokens
        ) / 1_000_000.0
    return costs


def load_events(path: Path) -> pd.DataFrame:
    """Load routing events in original file order."""
    df = pd.read_csv(path)
    df["row_id"] = np.arange(len(df))
    df = df[df["policy"].isin(TARGET_POLICIES)].copy()
    df = df.sort_values(["timestamp", "row_id"]).reset_index(drop=True)
    df["window_start"] = np.floor(df["timestamp"] / WINDOW_SEC) * WINDOW_SEC
    df["window_dt"] = pd.to_datetime(df["window_start"], unit="s")
    return df


def build_probe_update_events(
    path: Path,
    gap_threshold_sec: float = 10.0,
) -> pd.DataFrame:
    """Infer forced LP updates that happen after each probing round."""
    df = pd.read_csv(path, usecols=["timestamp", "policy"])
    probes = df[df["policy"] == "_probing"].copy().sort_values("timestamp")
    if probes.empty:
        return pd.DataFrame(columns=["timestamp", "event_type", "policy"])

    probes["gap"] = probes["timestamp"].diff().fillna(np.inf)
    probes["round_id"] = (probes["gap"] > gap_threshold_sec).cumsum()
    rounds = probes.groupby("round_id")["timestamp"].max().reset_index(drop=True)
    return pd.DataFrame(
        {
            "timestamp": rounds.astype(float),
            "event_type": "force_update",
            "policy": "_probing",
        }
    )


def load_learning_events(path: Path, pricing: dict[str, float]) -> pd.DataFrame:
    """Load the chronological sample stream used to update the shared router.

    This matches runtime behavior:
    - include probing rows
    - include all policies except openrouter_auto
    - learn on actual_provider
    - add samples only after the corresponding request/probe has happened
    """
    df = pd.read_csv(
        path,
        usecols=[
            "timestamp",
            "policy",
            "actual_provider",
            "ttft_ms",
            "status",
        ],
    )
    df = df[df["policy"] != "openrouter_auto"].copy()
    df = df[df["actual_provider"].isin(pricing)].copy()
    df["error_type"] = df["status"].map(
        lambda status: None if status == "success" else "server_error"
    )
    df["error_type"] = df["error_type"].astype(object)
    df["ttft_for_router"] = np.where(df["status"] == "success", df["ttft_ms"], -1.0)
    return df.sort_values("timestamp").reset_index(drop=True)


def initialize_router(pricing: dict[str, float]) -> OnlineLatencyRouter:
    """Initialize an empty router for incremental replay."""
    router = OnlineLatencyRouter(
        costs=pricing,
        slo_sec=SLO_SEC,
        window_sec=WINDOW_SEC,
        failure_mode=FailureMode.INFINITY,
        kappa=0.0,
        weight_smoothing=WEIGHT_SMOOTHING,
        lp_update_interval=LP_UPDATE_INTERVAL_SEC,
    )
    return router


def snapshot_lp_state(
    router: OnlineLatencyRouter,
    current_time: float,
) -> dict[str, object]:
    """Capture LP state at the current update time."""
    eligible = pre_filter(router.profiles, current_time)
    if not eligible:
        eligible = list(router.profiles.keys())

    raw_weights, status = solve_lp_with_fallback(
        providers=eligible,
        profiles=router.profiles,
        costs=router.costs,
        slo_sec=router.slo_sec,
        current_time=current_time,
        relaxation_factors=router.relaxation_factors,
        kappa=router.kappa,
    )

    row: dict[str, object] = {
        "timestamp": current_time,
        "datetime": pd.to_datetime(current_time, unit="s"),
        "status": status,
        "eligible": ",".join(sorted(eligible)),
    }

    for provider in router.profiles:
        row[f"cdf_{provider}"] = router.profiles[provider].get_cdf_at(
            router.slo_sec, current_time
        )
        row[f"error_{provider}"] = router.profiles[provider].get_error_rate_before(
            current_time
        )
        row[f"raw_{provider}"] = raw_weights.get(provider, 0.0)
        row[f"sampler_{provider}"] = router.sampler.get_weights().get(provider, 0.0)
        row[f"p99_{provider}"] = router.profiles[provider].get_p99(current_time)

    return row


def replay_router(
    events: pd.DataFrame,
    probe_updates: pd.DataFrame,
    learning_events: pd.DataFrame,
    router: OnlineLatencyRouter,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay route() over lp_mix+smart_hedge events and collect update states."""
    route_events = events[["timestamp", "policy", "row_id", "window_dt", "selected_provider", "actual_provider"]].copy()
    route_events["event_type"] = "route"

    force_events = probe_updates.copy()
    force_events["row_id"] = -1
    force_events["window_dt"] = pd.to_datetime(
        np.floor(force_events["timestamp"] / WINDOW_SEC) * WINDOW_SEC,
        unit="s",
    )
    force_events["selected_provider"] = np.nan
    force_events["actual_provider"] = np.nan

    timeline = pd.concat([route_events, force_events], ignore_index=True, sort=False)
    timeline = timeline.sort_values(["timestamp", "event_type", "row_id"]).reset_index(drop=True)

    route_rows: list[dict[str, object]] = []
    update_rows: list[dict[str, object]] = []
    learning_idx = 0
    n_learning = len(learning_events)

    for event in timeline.itertuples(index=False):
        current_time = float(event.timestamp)
        include_equal = event.event_type == "force_update"
        while learning_idx < n_learning:
            sample = learning_events.iloc[learning_idx]
            sample_time = float(sample["timestamp"])
            if sample_time < current_time or (include_equal and sample_time <= current_time):
                error_type = sample["error_type"]
                if pd.isna(error_type):
                    error_type = None
                router.add_sample(
                    provider=sample["actual_provider"],
                    timestamp=sample_time,
                    ttft_ms=float(sample["ttft_for_router"]),
                    error_type=error_type,
                )
                learning_idx += 1
            else:
                break

        if event.event_type == "force_update":
            router.update_lp(current_time=current_time, force=True)
            snapshot = snapshot_lp_state(router, current_time)
            snapshot["trigger_policy"] = event.policy
            snapshot["predicted_selected"] = None
            update_rows.append(snapshot)
            continue

        previous_update = router.last_lp_update
        predicted = router.route(current_time=current_time)
        updated = router.last_lp_update != previous_update

        route_rows.append(
            {
                "timestamp": event.timestamp,
                "datetime": pd.to_datetime(event.timestamp, unit="s"),
                "window_dt": event.window_dt,
                "policy": event.policy,
                "logged_selected": event.selected_provider,
                "logged_actual": event.actual_provider,
                "predicted_selected": predicted,
                "matches_logged_selected": predicted == event.selected_provider,
                "lp_updated": updated,
            }
        )

        if updated:
            snapshot = snapshot_lp_state(router, current_time)
            snapshot["trigger_policy"] = event.policy
            snapshot["predicted_selected"] = predicted
            update_rows.append(snapshot)

    return pd.DataFrame(route_rows), pd.DataFrame(update_rows)


def compute_window_summary(
    events: pd.DataFrame,
    route_df: pd.DataFrame,
    update_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize actual shares, replayed shares, and LP state by 15-minute window."""
    actual = (
        events[events["policy"] == "lp_mix"]
        .groupby(["window_dt", "actual_provider"])
        .size()
        .rename("count")
        .reset_index()
    )
    actual_total = (
        events[events["policy"] == "lp_mix"]
        .groupby("window_dt")
        .size()
        .rename("window_total")
        .reset_index()
    )
    actual = actual.merge(actual_total, on="window_dt", how="left")
    actual["actual_share"] = actual["count"] / actual["window_total"]
    actual = actual.pivot_table(
        index="window_dt",
        columns="actual_provider",
        values="actual_share",
        fill_value=0.0,
    )
    actual.columns = [f"actual_{c}" for c in actual.columns]

    replay = (
        route_df[route_df["policy"] == "lp_mix"]
        .groupby(["window_dt", "predicted_selected"])
        .size()
        .rename("count")
        .reset_index()
    )
    replay_total = (
        route_df[route_df["policy"] == "lp_mix"]
        .groupby("window_dt")
        .size()
        .rename("window_total")
        .reset_index()
    )
    replay = replay.merge(replay_total, on="window_dt", how="left")
    replay["replay_share"] = replay["count"] / replay["window_total"]
    replay = replay.pivot_table(
        index="window_dt",
        columns="predicted_selected",
        values="replay_share",
        fill_value=0.0,
    )
    replay.columns = [f"replay_{c}" for c in replay.columns]

    updates = update_df.copy()
    updates["window_start"] = np.floor(updates["timestamp"] / WINDOW_SEC) * WINDOW_SEC
    updates["window_dt"] = pd.to_datetime(updates["window_start"], unit="s")

    agg_cols: dict[str, str] = {}
    for provider in FOCUS_PROVIDERS:
        agg_cols[f"raw_{provider}"] = "mean"
        agg_cols[f"sampler_{provider}"] = "mean"
        agg_cols[f"cdf_{provider}"] = "mean"
        agg_cols[f"p99_{provider}"] = "mean"
    updates_grouped = updates.groupby("window_dt").agg(agg_cols)
    updates_grouped.columns = [
        f"mean_{col}" for col in updates_grouped.columns.to_flat_index()
    ]
    updates_grouped["n_updates"] = updates.groupby("window_dt").size()

    status_mode = (
        updates.groupby("window_dt")["status"]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("status_mode")
    )
    updates_grouped = updates_grouped.join(status_mode, how="left")

    summary = actual.join(replay, how="outer").join(updates_grouped, how="outer")
    summary = summary.reset_index().sort_values("window_dt")
    return summary


def plot_weight_vs_share(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot actual share vs reconstructed weights for WandB and Alibaba."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    for axis, provider in zip(axes, ["WandB", "Alibaba"]):
        axis.plot(
            summary["window_dt"],
            summary.get(f"actual_{provider}", pd.Series(0.0, index=summary.index)),
            label=f"Actual lp_mix share: {provider}",
            linewidth=2.0,
            color="#1f77b4" if provider == "WandB" else "#ff7f0e",
        )
        axis.plot(
            summary["window_dt"],
            summary.get(f"replay_{provider}", pd.Series(0.0, index=summary.index)),
            label=f"Replay selected share: {provider}",
            linewidth=1.6,
            linestyle="--",
            color="#17becf" if provider == "WandB" else "#d62728",
        )
        axis.plot(
            summary["window_dt"],
            summary.get(
                f"mean_sampler_{provider}", pd.Series(np.nan, index=summary.index)
            ),
            label=f"Mean sampler weight: {provider}",
            linewidth=1.6,
            linestyle="-.",
            color="#9467bd" if provider == "WandB" else "#2ca02c",
        )
        axis.plot(
            summary["window_dt"],
            summary.get(
                f"mean_raw_{provider}", pd.Series(np.nan, index=summary.index)
            ),
            label=f"Mean raw LP weight: {provider}",
            linewidth=1.2,
            linestyle=":",
            color="#7f7f7f",
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("Share / Weight")
        axis.set_title(f"{provider}: Actual share vs reconstructed weights")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", fontsize=9)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.suptitle("Alibaba Reconstruction: Actual Share vs LP/Sampler Weights", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cdf_focus(update_df: pd.DataFrame, output_path: Path) -> None:
    """Plot CDF-at-SLO evolution for focus providers."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    for provider in FOCUS_PROVIDERS:
        axes[0].plot(
            update_df["datetime"],
            update_df[f"cdf_{provider}"],
            label=provider,
            linewidth=1.8,
        )
    axes[0].axhline(0.99, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("F(SLO=2.0s)")
    axes[0].set_title("CDF at SLO over LP update times")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    for provider in FOCUS_PROVIDERS:
        axes[1].plot(
            update_df["datetime"],
            update_df[f"p99_{provider}"],
            label=provider,
            linewidth=1.8,
        )
    axes[1].set_ylabel("P99 (s)")
    axes[1].set_title("P99 over LP update times")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    fig.suptitle("Alibaba Reconstruction: CDF and Tail Dynamics", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pricing = load_pricing(DEFAULT_PRICING)
    events = load_events(DEFAULT_EVAL_LOG)
    probe_updates = build_probe_update_events(DEFAULT_EVAL_LOG)
    learning_events = load_learning_events(DEFAULT_EVAL_LOG, pricing)
    router = initialize_router(pricing)

    route_df, update_df = replay_router(events, probe_updates, learning_events, router)
    summary = compute_window_summary(events, route_df, update_df)

    route_df.to_csv(output_dir / "router_replay_events.csv", index=False)
    update_df.to_csv(output_dir / "router_lp_updates.csv", index=False)
    summary.to_csv(output_dir / "router_window_summary.csv", index=False)

    plot_weight_vs_share(summary, output_dir / "alibaba_weight_vs_share.png")
    plot_cdf_focus(update_df, output_dir / "alibaba_cdf_focus.png")

    replay_accuracy = route_df["matches_logged_selected"].mean()
    print(f"Replay accuracy vs logged selected_provider: {replay_accuracy:.4f}")

    top_cols = [
        "window_dt",
        "actual_Alibaba",
        "replay_Alibaba",
        "mean_raw_Alibaba",
        "mean_sampler_Alibaba",
        "actual_WandB",
        "replay_WandB",
        "mean_raw_WandB",
        "mean_sampler_WandB",
        "mean_cdf_Alibaba",
        "mean_cdf_WandB",
        "mean_p99_Alibaba",
        "mean_p99_WandB",
        "n_updates",
        "status_mode",
    ]
    existing_cols = [c for c in top_cols if c in summary.columns]
    top_windows = (
        summary.sort_values("actual_Alibaba", ascending=False)
        .head(10)[existing_cols]
    )
    print("\nTop Alibaba-heavy windows:")
    print(top_windows.to_string(index=False))
    print(f"\nOutputs saved to {output_dir}")


if __name__ == "__main__":
    main()
