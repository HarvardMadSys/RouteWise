"""Block bootstrap utilities for offline counterfactual simulation.

Resamples 15-minute windows with replacement to preserve temporal
clustering in tails and cross-provider correlations within each window.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def assign_windows(
    df: pd.DataFrame,
    window_sec: int = 900,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Assign each row to a non-overlapping window index.

    Args:
        df: DataFrame with a timestamp column.
        window_sec: Window duration in seconds.
        ts_col: Name of the timestamp column.

    Returns:
        DataFrame with an added 'window_idx' column (0-based).
    """
    t0 = df[ts_col].min()
    df = df.copy()
    df["window_idx"] = ((df[ts_col] - t0) / window_sec).astype(int)
    return df


def block_bootstrap(
    df: pd.DataFrame,
    window_sec: int = 900,
    n_trials: int = 100,
    seed: int = 42,
    ts_col: str = "timestamp",
) -> Iterator[pd.DataFrame]:
    """Yield bootstrap trials by resampling window indices.

    Each trial samples T window indices with replacement, concatenates
    all observations from the sampled windows. All requests are kept
    (including those originally routed to the excluded provider); the
    exclusion only affects the candidate set for routing.

    Args:
        df: Full openrouter_auto DataFrame with columns:
            timestamp, actual_provider, ttft_ms, cost_usd, etc.
        window_sec: Window duration in seconds (default 15 min).
        n_trials: Number of bootstrap trials to generate.
        seed: Random seed for reproducibility.
        ts_col: Name of the timestamp column.

    Yields:
        DataFrame for each trial. The 'window_idx' column is remapped
        to sequential indices (0..T-1) for the trial, and a
        'sim_time' column provides monotonically increasing synthetic
        timestamps preserving within-window ordering.
    """
    df = assign_windows(df, window_sec, ts_col)
    unique_windows = sorted(df["window_idx"].unique())
    T = len(unique_windows)

    # Pre-group by window for fast lookups.
    grouped = {w: g for w, g in df.groupby("window_idx")}

    rng = np.random.default_rng(seed)

    for trial in range(n_trials):
        sampled_indices = rng.choice(unique_windows, size=T, replace=True)

        pieces = []
        for new_idx, orig_idx in enumerate(sampled_indices):
            chunk = grouped[orig_idx].copy()
            chunk["window_idx"] = new_idx
            # Create synthetic time: window_start + offset within window.
            window_start = new_idx * window_sec
            offsets = chunk[ts_col] - chunk[ts_col].min()
            chunk["sim_time"] = window_start + offsets
            pieces.append(chunk)

        trial_df = pd.concat(pieces, ignore_index=True)
        trial_df = trial_df.sort_values("sim_time").reset_index(drop=True)
        trial_df["trial"] = trial
        yield trial_df


def build_latency_pools(
    trial_df: pd.DataFrame,
    window_sec: int = 900,
    exclude_providers: set[str] | None = None,
    provider_col: str = "actual_provider",
    ttft_col: str = "ttft_ms",
) -> dict[int, dict[str, np.ndarray]]:
    """Build per-window, per-provider latency pools from a trial DataFrame.

    Args:
        trial_df: Bootstrap trial DataFrame with window_idx, provider, ttft.
        window_sec: Window duration (for documentation, not used in logic).
        exclude_providers: Providers to exclude from pools.
        provider_col: Column name for provider.
        ttft_col: Column name for TTFT in milliseconds.

    Returns:
        Dict[window_idx -> Dict[provider -> ndarray of TTFT in ms]].
        Only includes successful requests (non-NaN TTFT) from
        non-excluded providers.
    """
    exclude = exclude_providers or set()
    pools: dict[int, dict[str, np.ndarray]] = {}

    mask = (
        ~trial_df[provider_col].isin(exclude)
        & trial_df[ttft_col].notna()
        & (trial_df[ttft_col] > 0)
    )
    filtered = trial_df[mask]

    for (widx, prov), group in filtered.groupby(["window_idx", provider_col]):
        if widx not in pools:
            pools[widx] = {}
        pools[widx][prov] = group[ttft_col].values

    return pools


def get_pool_with_fallback(
    pools: dict[int, dict[str, np.ndarray]],
    window_idx: int,
    provider: str,
    max_lookback: int = 10,
) -> np.ndarray | None:
    """Get latency pool for a provider, falling back to past windows only.

    Args:
        pools: Per-window, per-provider latency pools.
        window_idx: Current window index.
        provider: Provider name.
        max_lookback: Maximum number of past windows to search.

    Returns:
        ndarray of TTFT values in ms, or None if no data found.
    """
    # Try current window first.
    if window_idx in pools and provider in pools[window_idx]:
        return pools[window_idx][provider]

    # Fall back to past windows only (causal).
    for lookback in range(1, max_lookback + 1):
        past_idx = window_idx - lookback
        if past_idx in pools and provider in pools[past_idx]:
            return pools[past_idx][provider]

    return None
