"""Data loader for CSV request logs."""

import csv
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Increase CSV field size limit for large log entries
csv.field_size_limit(sys.maxsize)

import numpy as np

from routewise.schemas import Request

logger = logging.getLogger(__name__)

# Model name normalization mapping
# Maps custom/variant model names to canonical names for consistent pricing
# Naming convention: {provider}-{version}-{tier} (e.g., claude-4.5-opus)
MODEL_NAME_MAPPING = {
    # Custom wrappers -> canonical models
    "custom-4.5-sonnet": "claude-4.5-sonnet",
    "custom-model-opus": "claude-4.5-opus",
    "custom-model-alpha": "gpt-5",  # Per user specification
    "custom-model-beta": "claude-4.1-opus",  # Per user specification (opus 4.1)
    "custom-glm-4.6": "glm-4.6",
    "custom-gemini-3-pro": "gemini-3-pro",
    "custom-gpt-5": "gpt-5",
    # Normalize claude naming variants (use claude-{version}-{tier} format)
    "claude-opus-4.1": "claude-4.1-opus",
    # DeepSeek variants
    "deepseek-chat": "deepseek-r1",
    # Freeinference specific -> canonical
    "freeinference-glm-4.6": "glm-4.6",
    # Add more mappings as needed
}


def normalize_model_name(model: str | None) -> str | None:
    """Normalize model name using the mapping table.

    Args:
        model: Original model name

    Returns:
        Normalized model name, or original if no mapping exists
    """
    if model is None:
        return None
    return MODEL_NAME_MAPPING.get(model, model)


class DataLoader:
    """Load historical request data from CSV files.

    Supports three formats (auto-detected):
    1. BurstGPT: Timestamp, Model, Request tokens, Response tokens, Total tokens
    2. Experiment: Timestamp, Model, Request tokens, Response tokens, Total tokens, Provider, Actual Cost
    3. HybridInference (native): timestamp, model_id, prompt_tokens, completion_tokens, total_tokens, provider, cost_usd

    Attributes:
        config: Configuration dictionary
    """

    def __init__(self, config: dict):
        """Initialize data loader.

        Args:
            config: Configuration dictionary containing dataset settings
        """
        self.config = config
        self.dataset_config = config.get("dataset", {})

    def load(
        self,
        filepath: str | Path,
        filter_model: str | None = None,
        model_override: str | None = None,
    ) -> list[Request]:
        """Load requests from CSV file.

        Auto-detects format based on available columns.

        Args:
            filepath: Path to CSV file
            filter_model: Optional model name to filter (e.g., "ChatGPT", "GPT-4").
                         If provided, only requests for this model are loaded.
            model_override: Optional model name to use for ALL requests.
                           Use this to treat a dataset as single-model workload.
                           E.g., model_override="deepseek-r1" maps all BurstGPT
                           requests to deepseek-r1 for pricing.

        Returns:
            List of Request objects, sorted by timestamp

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV format is invalid
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        logger.info(f"Loading dataset from {filepath}")
        if filter_model:
            logger.info(f"Filtering for model: {filter_model}")
        if model_override:
            logger.info(f"Model override: all requests will use model '{model_override}'")

        # Detect format by reading first row
        with open(filepath) as f:
            reader = csv.DictReader(f)
            first_row = next(reader, None)
            if first_row is None:
                raise ValueError("Empty CSV file")

            # Detect format
            if "prompt_tokens" in first_row:
                format_type = "hybridinference"
                has_provider = "provider" in first_row
                has_actual_cost = "cost_usd" in first_row
            elif "Request tokens" in first_row:
                format_type = "experiment"
                has_provider = "Provider" in first_row
                has_actual_cost = "Actual Cost" in first_row
            else:
                raise ValueError(
                    "Unknown CSV format. Expected columns like 'Request tokens' or 'prompt_tokens'"
                )

        # Log detected format
        if format_type == "hybridinference":
            logger.info(
                "Detected HybridInference native format (prompt_tokens, completion_tokens, cost_usd)"
            )
        elif has_provider:
            logger.info("Detected Experiment format with provider info")
        else:
            logger.info("Detected BurstGPT format (no provider info)")

        # Load data
        requests = []
        skipped = 0
        filtered_out = 0

        with open(filepath) as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                request = self._parse_row(
                    row, format_type, i, has_provider, has_actual_cost, model_override
                )
                if request:
                    # Apply model filter if specified
                    if filter_model and request.model != filter_model:
                        filtered_out += 1
                        continue

                    requests.append(request)
                else:
                    skipped += 1

                # Progress logging for large files
                if (i + 1) % 100000 == 0:
                    logger.info(f"Processed {i + 1} rows, {len(requests)} valid requests")

        if skipped > 0:
            logger.warning(f"Skipped {skipped} invalid rows")

        if filter_model and filtered_out > 0:
            logger.info(
                f"Filtered out {filtered_out} requests (not matching model '{filter_model}')"
            )

        # Sort by timestamp
        requests.sort(key=lambda r: r.timestamp)

        logger.info(f"Loaded {len(requests)} requests from {filepath}")

        return requests

    def _parse_row(
        self,
        row: dict,
        format_type: str,
        request_id: int,
        has_provider: bool = False,
        has_actual_cost: bool = False,
        model_override: str | None = None,
    ) -> Request | None:
        """Parse a single CSV row into Request object.

        Args:
            row: Dictionary from CSV reader
            format_type: Format type ("hybridinference" or "experiment")
            request_id: Unique identifier to assign to this request
            has_provider: Whether Provider column exists
            has_actual_cost: Whether Actual Cost column exists
            model_override: If set, use this model instead of the row's model

        Returns:
            Request object or None if row is invalid
        """
        try:
            # Parse based on format
            if format_type == "hybridinference":
                # HybridInference native format
                request_tokens = int(row["prompt_tokens"])
                response_tokens = int(row["completion_tokens"])
                total_tokens = int(row["total_tokens"])

                # Parse timestamp (multiple formats supported)
                timestamp_str = row.get("timestamp", "")
                if "T" in timestamp_str:  # ISO format: 2025-11-06T09:07:36.243526+00:00
                    dt = datetime.fromisoformat(timestamp_str)
                    timestamp = int(dt.timestamp())
                elif " " in timestamp_str:  # Space format: 2025-09-12 15:02:36 or with microseconds
                    # Remove timezone and microseconds if present
                    ts_clean = timestamp_str.split("+")[0].split(".")[0]
                    dt = datetime.strptime(ts_clean, "%Y-%m-%d %H:%M:%S")
                    timestamp = int(dt.timestamp())
                else:
                    timestamp = int(timestamp_str)

                # Parse optional fields and normalize model name
                model = normalize_model_name(row.get("model_id"))
                provider = row.get("provider")
                actual_cost = float(row.get("cost_usd", 0)) if row.get("cost_usd") else None

                # Parse latency fields for Phase 2
                latency_ms = int(row.get("latency_ms", 0)) if row.get("latency_ms") else None
                ttft_ms = int(row.get("ttft_ms", 0)) if row.get("ttft_ms") else None

            else:
                # Standard experiment format
                request_tokens = int(row["Request tokens"])
                response_tokens = int(row["Response tokens"])
                total_tokens = int(row["Total tokens"])
                timestamp = int(row["Timestamp"])

                # Parse optional fields and normalize model name
                model = normalize_model_name(row.get("Model"))
                provider = row.get("Provider") if has_provider else None
                actual_cost = float(row.get("Actual Cost", 0)) if has_actual_cost else None
                latency_ms = None
                ttft_ms = None

            # Skip rows with zero tokens (likely errors)
            if total_tokens == 0:
                return None

            # Apply model override if specified (treat dataset as single-model)
            if model_override:
                model = model_override

            return Request(
                id=request_id,
                timestamp=timestamp,
                request_tokens=request_tokens,
                response_tokens=response_tokens,
                total_tokens=total_tokens,
                model=model,
                provider=provider,
                actual_cost=actual_cost,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
            )
        except (KeyError, ValueError) as e:
            logger.debug(f"Skipping invalid row: {e}")
            return None

    def get_statistics(self, requests: list[Request]) -> dict:
        """Calculate dataset statistics.

        Args:
            requests: List of requests

        Returns:
            Dictionary with statistics
        """
        if not requests:
            return {
                "total_requests": 0,
                "num_days": 0,
                "avg_request_tokens": 0,
                "avg_response_tokens": 0,
                "total_tokens": 0,
            }

        stats = {
            "total_requests": len(requests),
            "num_days": requests[-1].day - requests[0].day + 1,  # Actual span of days
            "avg_request_tokens": np.mean([r.request_tokens for r in requests]),
            "avg_response_tokens": np.mean([r.response_tokens for r in requests]),
            "total_tokens": sum(r.total_tokens for r in requests),
            "min_timestamp": requests[0].timestamp,
            "max_timestamp": requests[-1].timestamp,
        }

        # Add model distribution if available
        models = [r.model for r in requests if r.model is not None]
        if models:
            stats["models"] = dict(Counter(models))

        # Add provider distribution if available
        providers = [r.provider for r in requests if r.provider is not None]
        if providers:
            stats["providers"] = dict(Counter(providers))

        # Add actual cost if available
        actual_costs = [r.actual_cost for r in requests if r.actual_cost is not None]
        if actual_costs:
            stats["total_actual_cost"] = sum(actual_costs)
            stats["avg_actual_cost"] = np.mean(actual_costs)

        return stats


__all__ = ["DataLoader", "MODEL_NAME_MAPPING", "normalize_model_name"]
