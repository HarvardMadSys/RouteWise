"""Caching utilities for experiment data and ILP results.

This module provides caching for:
1. Dataset loading (CSV -> Request objects): ~30s -> <1s
2. ILP solver results (assignments + schedules): minutes/hours -> instant

Cache keys include all relevant parameters to avoid stale cache issues.
"""

import datetime
import hashlib
import json
import logging
import pickle
from pathlib import Path

from experiment.data.schema import Request

logger = logging.getLogger(__name__)

# Default cache directory
CACHE_DIR = Path("experiment/cache")


def get_cache_dir() -> Path:
    """Get and create cache directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _hash_dict(d: dict | None) -> str:
    """Create a short hash of a dictionary for cache keys."""
    if d is None:
        return "none"
    json_str = json.dumps(d, sort_keys=True, default=str)
    return hashlib.md5(json_str.encode()).hexdigest()[:8]


# ============================================================
# Dataset Cache (CSV -> Request objects)
# ============================================================


def get_dataset_cache_path(dataset_name: str) -> Path:
    """Get cache path for dataset."""
    cache_dir = get_cache_dir() / "dataset"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{dataset_name}.pkl"


def load_cached_dataset(dataset_name: str) -> list[Request] | None:
    """Load dataset from cache if available.

    Args:
        dataset_name: Name of the dataset (e.g., "freeinference", "rednote")

    Returns:
        List of Request objects if cache exists, None otherwise
    """
    cache_path = get_dataset_cache_path(dataset_name)
    if cache_path.exists():
        # Log cache file info for debugging stale cache issues
        mtime = cache_path.stat().st_mtime
        cache_time = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Loading cached dataset from {cache_path} (cached at {cache_time})")

        with open(cache_path, "rb") as f:
            requests = pickle.load(f)
        logger.info(f"Loaded {len(requests)} requests from cache")
        return requests
    return None


def save_dataset_cache(dataset_name: str, requests: list[Request]) -> None:
    """Save dataset to cache.

    Args:
        dataset_name: Name of the dataset
        requests: List of Request objects to cache
    """
    cache_path = get_dataset_cache_path(dataset_name)
    logger.info(f"Saving dataset cache to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(requests, f)
    logger.info(f"Cached {len(requests)} requests")


# ============================================================
# ILP Result Cache (assignments + schedules)
# ============================================================


def get_ilp_cache_key(
    dataset_name: str,
    num_requests: int,
    delta: float,
    daily_quota: int,
    concurrency_limit: int,
    solver: str,
    latency_slo: int | None = None,
    model_pricing: dict | None = None,
    subscriptions: dict | None = None,
    model_override: str | None = None,
) -> str:
    """Generate cache key for ILP results.

    Args:
        dataset_name: Name of the dataset
        num_requests: Number of requests (to differentiate --limit runs)
        delta: Time slot size in seconds
        daily_quota: Daily quota limit (Q)
        concurrency_limit: Concurrency limit (C)
        solver: Solver name ("cbc" or "gurobi")
        latency_slo: Latency SLO in slots (0=zero-wait, None=unlimited)
        model_pricing: Model pricing dict (hashed)
        subscriptions: Subscription config (hashed)
        model_override: If set, all requests mapped to this model

    Returns:
        Cache key string
    """
    pricing_hash = _hash_dict(model_pricing)
    sub_hash = _hash_dict(subscriptions)
    slo_str = str(latency_slo) if latency_slo is not None else "none"
    model_str = f"_m{model_override}" if model_override else ""

    return f"{dataset_name}_n{num_requests}_d{delta}_Q{daily_quota}_C{concurrency_limit}_{solver}_slo{slo_str}_p{pricing_hash}_s{sub_hash}{model_str}"


def get_ilp_cache_path(cache_key: str) -> Path:
    """Get cache path for ILP results."""
    cache_dir = get_cache_dir() / "ilp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.json"


def load_cached_ilp_result(cache_key: str) -> dict | None:
    """Load ILP results from cache if available.

    Args:
        cache_key: Cache key from get_ilp_cache_key()

    Returns:
        Dictionary with 'assignments' and 'schedules' if cache exists, None otherwise
    """
    cache_path = get_ilp_cache_path(cache_key)
    if cache_path.exists():
        logger.info(f"Loading cached ILP result from {cache_path}")
        with open(cache_path) as f:
            data = json.load(f)

        # Convert string keys back to int (JSON limitation)
        assignments = {int(k): v for k, v in data["assignments"].items()}
        schedules = {int(k): tuple(v) for k, v in data.get("schedules", {}).items()}

        logger.info(f"Loaded ILP cache: {len(assignments)} assignments, {len(schedules)} schedules")
        return {"assignments": assignments, "schedules": schedules}
    return None


def save_ilp_cache(
    cache_key: str,
    assignments: dict[int, str],
    schedules: dict[int, tuple[int, int]],
) -> None:
    """Save ILP results to cache.

    Args:
        cache_key: Cache key from get_ilp_cache_key()
        assignments: Request ID -> provider type mapping
        schedules: Request ID -> (start_time, finish_time) mapping
    """
    cache_path = get_ilp_cache_path(cache_key)
    logger.info(f"Saving ILP cache to {cache_path}")

    # Convert int keys to string for JSON
    data = {
        "assignments": {str(k): v for k, v in assignments.items()},
        "schedules": {str(k): list(v) for k, v in schedules.items()},
    }

    with open(cache_path, "w") as f:
        json.dump(data, f)

    logger.info(f"Cached {len(assignments)} assignments, {len(schedules)} schedules")


def clear_cache(cache_type: str | None = None) -> None:
    """Clear cache files.

    Args:
        cache_type: "dataset", "ilp", or None for all
    """
    cache_dir = get_cache_dir()

    if cache_type is None or cache_type == "dataset":
        dataset_dir = cache_dir / "dataset"
        if dataset_dir.exists():
            for f in dataset_dir.glob("*.pkl"):
                f.unlink()
                logger.info(f"Deleted {f}")

    if cache_type is None or cache_type == "ilp":
        ilp_dir = cache_dir / "ilp"
        if ilp_dir.exists():
            for f in ilp_dir.glob("*.json"):
                f.unlink()
                logger.info(f"Deleted {f}")
