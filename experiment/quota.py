"""Quota manager for subscription providers."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class QuotaManager:
    """Track and manage subscription quota.

    Attributes:
        daily_quota_per_sub: Quota per subscription per day
        num_subscriptions: Number of subscription accounts
        total_daily_quota: Total daily quota across all subscriptions
        current_day: Current day number
        used_today: Quota used today
        total_used: Total quota used across all days
        daily_usage: Dictionary mapping day to usage
    """

    def __init__(self, daily_quota: int, num_subscriptions: int = 1):
        """Initialize quota manager.

        Args:
            daily_quota: Quota per subscription per day
            num_subscriptions: Number of subscription accounts
        """
        self.daily_quota_per_sub = daily_quota
        self.num_subscriptions = num_subscriptions
        self.total_daily_quota = daily_quota * num_subscriptions

        # State
        self.current_day = 0
        self.used_today = 0
        self.total_used = 0

        # Statistics
        self.daily_usage: dict[int, int] = {}

    def reset_if_new_day(self, day: int) -> None:
        """Reset quota if entering a new day.

        Args:
            day: Current day number (0-indexed)
        """
        if day > self.current_day:
            # Record previous day's usage
            self.daily_usage[self.current_day] = self.used_today

            # Reset for new day
            self.current_day = day
            self.used_today = 0

    def has_quota(self, amount: int = 1) -> bool:
        """Check if quota is available.

        Args:
            amount: Amount of quota needed

        Returns:
            True if quota is available
        """
        return self.used_today + amount <= self.total_daily_quota

    def use_quota(self, amount: int = 1) -> bool:
        """Try to use quota.

        Args:
            amount: Amount of quota to use

        Returns:
            True if quota was successfully used, False otherwise
        """
        if not self.has_quota(amount):
            return False

        self.used_today += amount
        self.total_used += amount
        return True

    @property
    def remaining_today(self) -> int:
        """Remaining quota for today."""
        return max(0, self.total_daily_quota - self.used_today)

    @property
    def utilization_today(self) -> float:
        """Quota utilization for today (0.0 to 1.0)."""
        if self.total_daily_quota == 0:
            return 0.0
        return self.used_today / self.total_daily_quota

    def finalize(self) -> None:
        """Finalize quota tracking by recording the last day's usage.

        Call this at the end of simulation to ensure the last day is recorded.
        """
        if self.used_today > 0 or self.current_day not in self.daily_usage:
            self.daily_usage[self.current_day] = self.used_today

    def get_statistics(self) -> dict:
        """Get quota usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        if not self.daily_usage:
            return {
                "total_used": self.total_used,
                "avg_daily_usage": 0,
                "max_daily_usage": 0,
                "min_daily_usage": 0,
                "avg_utilization": 0,
                "days_quota_exhausted": 0,
            }

        usages = list(self.daily_usage.values())
        # Edge case: quota disabled (e.g., All-API baseline).
        # Avoid division-by-zero and report utilization as 0.
        if self.total_daily_quota <= 0:
            return {
                "total_used": self.total_used,
                "avg_daily_usage": float(np.mean(usages)) if usages else 0,
                "max_daily_usage": max(usages) if usages else 0,
                "min_daily_usage": min(usages) if usages else 0,
                "avg_utilization": 0,
                "days_quota_exhausted": 0,
            }
        return {
            "total_used": self.total_used,
            "avg_daily_usage": float(np.mean(usages)),
            "max_daily_usage": max(usages),
            "min_daily_usage": min(usages),
            "avg_utilization": float(np.mean(usages)) / self.total_daily_quota,
            "days_quota_exhausted": sum(1 for u in usages if u >= self.total_daily_quota),
        }
