"""Window-based quota manager for Stage 1 fixed-window subscriptions.

This module implements the quota management for commercial subscriptions
with fixed time window constraints (e.g., 160 messages per 3 hours).
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PlanConfig:
    """Configuration for a subscription plan."""

    plan_id: str
    window_hours: float
    quota_per_window: int
    supported_models: list[str]
    monthly_fee: float = 0.0
    quota_unit: str = "message"
    notes: str = ""

    @property
    def window_seconds(self) -> float:
        """Window size in seconds."""
        return self.window_hours * 3600


@dataclass
class WindowQuotaManager:
    """Manage quota for multiple subscription plans with fixed time windows.

    Each plan has:
    - A set of supported models
    - A window size (e.g., 3 hours)
    - A quota per window (e.g., 160 messages)

    Requests are routed to plans based on model compatibility.
    Within each (plan, window), quota is shared across all compatible models.
    """

    plans: dict[str, PlanConfig] = field(default_factory=dict)
    model_to_plan: dict[str, str] = field(default_factory=dict)
    reference_time: float = 0.0  # t_0 for window alignment

    # Runtime state: tracks usage per (plan_id, window_index)
    usage: dict[tuple[str, int], int] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        plans_config: dict,
        reference_time: float = 0.0,
        active_plans: list[str] | None = None,
    ) -> "WindowQuotaManager":
        """Create manager from YAML config.

        Args:
            plans_config: Dictionary of plan configurations from offline-stage YAML
            reference_time: Reference timestamp for window alignment (default: 0)
            active_plans: List of plan IDs to activate. If None, all plans are used.

        Returns:
            Configured WindowQuotaManager instance
        """
        manager = cls(reference_time=reference_time)

        for plan_id, config in plans_config.items():
            # Skip if not in active_plans (when specified)
            if active_plans is not None and plan_id not in active_plans:
                continue

            plan = PlanConfig(
                plan_id=plan_id,
                window_hours=config["window_hours"],
                quota_per_window=config["quota_per_window"],
                supported_models=config.get("supported_models", []),
                monthly_fee=config.get("monthly_fee", 0.0),
                quota_unit=config.get("quota_unit", "message"),
                notes=config.get("notes", ""),
            )
            manager.plans[plan_id] = plan

            # Build model -> plan mapping
            for model in plan.supported_models:
                if model in manager.model_to_plan:
                    logger.warning(
                        f"Model '{model}' already mapped to plan '{manager.model_to_plan[model]}', "
                        f"overwriting with '{plan_id}'"
                    )
                manager.model_to_plan[model] = plan_id

        logger.info(
            f"WindowQuotaManager initialized with {len(manager.plans)} plans, "
            f"{len(manager.model_to_plan)} model mappings"
        )
        if active_plans:
            logger.info(f"Active plans: {list(manager.plans.keys())}")
        return manager

    def get_plan_for_model(self, model: str) -> str | None:
        """Get the plan ID for a given model.

        Args:
            model: Model name

        Returns:
            Plan ID if model is supported, None otherwise
        """
        return self.model_to_plan.get(model)

    def get_window_index(self, plan_id: str, timestamp: float) -> int:
        """Calculate the window index for a given timestamp.

        Args:
            plan_id: Plan identifier
            timestamp: Request timestamp (seconds since epoch)

        Returns:
            Window index (0-based)
        """
        plan = self.plans[plan_id]
        return int((timestamp - self.reference_time) // plan.window_seconds)

    def get_quota(self, plan_id: str) -> int:
        """Get the quota for a plan.

        Args:
            plan_id: Plan identifier

        Returns:
            Quota per window
        """
        return self.plans[plan_id].quota_per_window

    def get_usage(self, plan_id: str, window_index: int) -> int:
        """Get current usage for a (plan, window) pair.

        Args:
            plan_id: Plan identifier
            window_index: Window index

        Returns:
            Number of requests used in this window
        """
        return self.usage.get((plan_id, window_index), 0)

    def has_quota(self, plan_id: str, window_index: int, amount: int = 1) -> bool:
        """Check if quota is available.

        Args:
            plan_id: Plan identifier
            window_index: Window index
            amount: Amount of quota needed

        Returns:
            True if quota is available
        """
        current = self.get_usage(plan_id, window_index)
        quota = self.get_quota(plan_id)
        return current + amount <= quota

    def use_quota(self, plan_id: str, window_index: int, amount: int = 1) -> bool:
        """Use quota for a (plan, window) pair.

        Args:
            plan_id: Plan identifier
            window_index: Window index
            amount: Amount of quota to use

        Returns:
            True if quota was successfully used
        """
        if not self.has_quota(plan_id, window_index, amount):
            return False

        key = (plan_id, window_index)
        self.usage[key] = self.usage.get(key, 0) + amount
        return True

    def remaining(self, plan_id: str, window_index: int) -> int:
        """Get remaining quota for a (plan, window) pair.

        Args:
            plan_id: Plan identifier
            window_index: Window index

        Returns:
            Remaining quota
        """
        return max(0, self.get_quota(plan_id) - self.get_usage(plan_id, window_index))

    def reset(self) -> None:
        """Reset all usage counters."""
        self.usage.clear()

    def get_statistics(self) -> dict:
        """Get usage statistics across all plans and windows.

        Returns:
            Dictionary with usage statistics
        """
        stats = {
            "total_used": sum(self.usage.values()),
            "by_plan": {},
        }

        for plan_id in self.plans:
            plan_usage = {k: v for k, v in self.usage.items() if k[0] == plan_id}
            if plan_usage:
                usages = list(plan_usage.values())
                quota = self.get_quota(plan_id)
                stats["by_plan"][plan_id] = {
                    "total_used": sum(usages),
                    "windows_count": len(usages),
                    "avg_usage": sum(usages) / len(usages),
                    "max_usage": max(usages),
                    "avg_utilization": sum(usages) / len(usages) / quota if quota > 0 else 0,
                }

        return stats


def build_model_plan_mapping(plans_config: dict) -> dict[str, str]:
    """Build a mapping from model names to plan IDs.

    Args:
        plans_config: Plans configuration from offline-stage YAML

    Returns:
        Dictionary mapping model name to plan ID
    """
    mapping = {}
    for plan_id, config in plans_config.items():
        for model in config.get("supported_models", []):
            mapping[model] = plan_id
    return mapping
