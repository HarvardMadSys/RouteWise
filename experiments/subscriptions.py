"""Shared subscription-plan facts for experiment harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import yaml

DEFAULT_BILLING_PERIOD_DAYS = 30.0
DEFAULT_SUBSCRIPTION_PLANS_PATH = Path(__file__).with_name("subscription_plans.yaml")


@dataclass(frozen=True)
class QuotaWindow:
    """One fixed-window quota constraint in a subscription plan."""

    name: str
    quota_requests: int
    quota_window_sec: float


@dataclass(frozen=True)
class ModelClassResolution:
    """Resolved concurrency class and cost for one trace/provider model id."""

    model_class: str
    cost: int
    matched_via: Literal["override"] = "override"


@dataclass(frozen=True)
class SubscriptionPlan:
    """Canonical plan facts shared by simulation, live eval, and offline code."""

    plan_id: str
    display_name: str
    monthly_fee_usd: float | None
    quota_windows: tuple[QuotaWindow, ...]
    subscription_counts: tuple[int, ...]
    eligible_sections: tuple[str, ...]
    cost_claim_allowed: bool
    source: str
    transport: str | None = None
    notes: str = ""
    tier: str = "quota"
    billing_mode: str = "subscription"
    concurrency_allotment: int | None = None
    model_concurrency_costs_by_class: MappingProxyType[str, int] = MappingProxyType({})
    model_class_overrides: MappingProxyType[str, str] = MappingProxyType({})

    @property
    def subscription_cost_known(self) -> bool:
        """Return whether fixed-fee cost is known for this plan."""
        return self.monthly_fee_usd is not None

    def resolve_model_class(self, model_id: str | None) -> str | None:
        """Resolve a trace/provider model id to this plan's concurrency class."""
        resolution = self.resolve_model_class_with_cost(model_id)
        return None if resolution is None else resolution.model_class

    def resolve_model_class_with_cost(
        self,
        model_id: str | None,
    ) -> ModelClassResolution | None:
        """Resolve a model id and expose the matched concurrency cost."""
        if not self.model_concurrency_costs_by_class:
            return None
        if model_id is None:
            return None
        model_class = self.model_class_overrides.get(str(model_id))
        if model_class is None:
            return None
        try:
            cost = self.model_concurrency_costs_by_class[model_class]
        except KeyError as exc:
            raise ValueError(
                f"plan {self.plan_id!r}: unknown concurrency model class {model_class!r}"
            ) from exc
        return ModelClassResolution(
            model_class=model_class,
            cost=cost,
        )

    def concurrency_cost_for_model(self, model_id: str | None) -> int:
        """Return weighted concurrency cost for a trace/provider model id."""
        resolution = self.resolve_model_class_with_cost(model_id)
        if resolution is None:
            raise ValueError(f"plan {self.plan_id!r}: no concurrency model class resolved")
        return resolution.cost


def load_subscription_plans(
    path: Path | None = None,
) -> dict[str, SubscriptionPlan]:
    """Load and validate canonical subscription-plan facts."""
    resolved = DEFAULT_SUBSCRIPTION_PLANS_PATH if path is None else Path(path)
    return _load_subscription_plans_cached(resolved.resolve())


@cache
def _load_subscription_plans_cached(path: Path) -> dict[str, SubscriptionPlan]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    plans_raw = raw.get("plans")
    if not isinstance(plans_raw, dict) or not plans_raw:
        raise ValueError(f"{path}: expected a non-empty top-level 'plans' map")
    return {
        str(plan_id): _parse_plan(str(plan_id), payload)
        for plan_id, payload in plans_raw.items()
    }


def _parse_plan(plan_id: str, payload: Any) -> SubscriptionPlan:
    if not isinstance(payload, dict):
        raise ValueError(f"plan {plan_id!r}: expected a mapping")

    monthly_fee = payload.get("monthly_fee_usd")
    if monthly_fee is not None:
        monthly_fee = float(monthly_fee)
        if monthly_fee < 0.0:
            raise ValueError(f"plan {plan_id!r}: monthly_fee_usd must be >= 0")

    cost_claim_allowed = bool(payload.get("cost_claim_allowed", False))
    if cost_claim_allowed and monthly_fee is None:
        raise ValueError(
            f"plan {plan_id!r}: cost_claim_allowed=true requires monthly_fee_usd"
        )

    tier = str(payload.get("tier", "quota"))
    if tier not in {"quota", "concurrency"}:
        raise ValueError(
            f"plan {plan_id!r}: tier must be 'quota' or 'concurrency', got {tier!r}"
        )
    billing_mode = str(payload.get("billing_mode", "subscription"))
    if billing_mode != "subscription":
        raise ValueError(
            f"plan {plan_id!r}: expected billing_mode='subscription', got {billing_mode!r}"
        )
    quota_windows = _parse_quota_windows(
        plan_id,
        payload.get("quota_windows"),
        required=tier == "quota",
    )
    (
        concurrency_allotment,
        model_concurrency_costs_by_class,
        model_class_overrides,
    ) = _parse_concurrency_fields(plan_id, payload, required=tier == "concurrency")
    subscription_counts = _positive_int_tuple(
        plan_id,
        "subscription_counts",
        payload.get("subscription_counts"),
    )
    eligible_sections = _string_tuple(payload.get("eligible_sections"), field="eligible_sections")

    return SubscriptionPlan(
        plan_id=plan_id,
        display_name=str(payload.get("display_name") or plan_id),
        monthly_fee_usd=monthly_fee,
        quota_windows=quota_windows,
        concurrency_allotment=concurrency_allotment,
        model_concurrency_costs_by_class=model_concurrency_costs_by_class,
        model_class_overrides=model_class_overrides,
        subscription_counts=subscription_counts,
        eligible_sections=eligible_sections,
        cost_claim_allowed=cost_claim_allowed,
        source=str(payload.get("source") or ""),
        transport=(
            str(payload["transport"])
            if payload.get("transport") is not None
            else None
        ),
        notes=str(payload.get("notes") or ""),
        tier=tier,
        billing_mode=billing_mode,
    )


def _parse_quota_windows(
    plan_id: str,
    raw: Any,
    *,
    required: bool,
) -> tuple[QuotaWindow, ...]:
    if raw is None:
        if required:
            raise ValueError(f"plan {plan_id!r}: quota_windows must be a non-empty list")
        return ()
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"plan {plan_id!r}: quota_windows must be a non-empty list")
    windows: list[QuotaWindow] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"plan {plan_id!r}: quota_windows[{index}] must be a map")
        try:
            quota_requests = int(item["quota_requests"])
            quota_window_sec = float(item["quota_window_sec"])
        except KeyError as exc:
            raise ValueError(
                f"plan {plan_id!r}: quota_windows[{index}] is missing {exc.args[0]!r}"
            ) from exc
        if quota_requests <= 0:
            raise ValueError(
                f"plan {plan_id!r}: quota_windows[{index}].quota_requests must be > 0"
            )
        if quota_window_sec <= 0.0:
            raise ValueError(
                f"plan {plan_id!r}: quota_windows[{index}].quota_window_sec must be > 0"
            )
        windows.append(
            QuotaWindow(
                name=str(item.get("name") or f"window_{index + 1}"),
                quota_requests=quota_requests,
                quota_window_sec=quota_window_sec,
            )
        )
    return tuple(windows)


def _parse_concurrency_fields(
    plan_id: str,
    payload: dict[str, Any],
    *,
    required: bool,
) -> tuple[int | None, MappingProxyType[str, int], MappingProxyType[str, str]]:
    raw_allotment = payload.get("concurrency_allotment")
    raw_costs = payload.get("model_concurrency_costs_by_class")
    raw_overrides = payload.get("model_class_overrides")
    if not required and raw_allotment is None and raw_costs is None:
        return None, MappingProxyType({}), MappingProxyType({})
    if raw_allotment is None:
        raise ValueError(f"plan {plan_id!r}: concurrency_allotment is required")
    concurrency_allotment = int(raw_allotment)
    if concurrency_allotment <= 0:
        raise ValueError(f"plan {plan_id!r}: concurrency_allotment must be > 0")
    if not isinstance(raw_costs, dict) or not raw_costs:
        raise ValueError(
            f"plan {plan_id!r}: model_concurrency_costs_by_class must be a non-empty map"
        )
    costs: dict[str, int] = {}
    for model_class, raw_cost in raw_costs.items():
        concurrency_cost = int(raw_cost)
        if concurrency_cost <= 0:
            raise ValueError(
                f"plan {plan_id!r}: concurrency cost for {model_class!r} must be > 0"
            )
        costs[str(model_class)] = concurrency_cost
    if raw_overrides is None:
        overrides: dict[str, str] = {}
    elif isinstance(raw_overrides, dict):
        overrides = {
            str(model_id): str(model_class)
            for model_id, model_class in raw_overrides.items()
        }
    else:
        raise ValueError(f"plan {plan_id!r}: model_class_overrides must be a map")
    unknown_override_classes = sorted(set(overrides.values()) - set(costs))
    if unknown_override_classes:
        raise ValueError(
            f"plan {plan_id!r}: model_class_overrides reference unknown classes: "
            f"{unknown_override_classes}"
        )
    return (
        concurrency_allotment,
        MappingProxyType(costs),
        MappingProxyType(overrides),
    )


def _positive_int_tuple(plan_id: str, field: str, raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"plan {plan_id!r}: {field} must be a non-empty list")
    values = tuple(int(item) for item in raw)
    if any(value <= 0 for value in values):
        raise ValueError(f"plan {plan_id!r}: {field} values must be > 0")
    if len(set(values)) != len(values):
        raise ValueError(f"plan {plan_id!r}: {field} values must be unique")
    return values


def _string_tuple(raw: Any, *, field: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a list when provided")
    return tuple(str(item) for item in raw)


def subscription_fixed_cost_usd(
    plan: SubscriptionPlan,
    *,
    subscription_count: int,
    trace_days: float,
    billing_period_days: float = DEFAULT_BILLING_PERIOD_DAYS,
) -> float:
    """Return prorated fixed fee for one plan/count over a trace span."""
    if plan.monthly_fee_usd is None:
        return 0.0
    if subscription_count <= 0:
        raise ValueError(f"subscription_count must be > 0, got {subscription_count}")
    if trace_days < 0.0:
        raise ValueError(f"trace_days must be >= 0, got {trace_days}")
    return (
        float(plan.monthly_fee_usd)
        * int(subscription_count)
        * (float(trace_days) / float(billing_period_days))
    )


__all__ = [
    "DEFAULT_BILLING_PERIOD_DAYS",
    "DEFAULT_SUBSCRIPTION_PLANS_PATH",
    "ModelClassResolution",
    "QuotaWindow",
    "SubscriptionPlan",
    "load_subscription_plans",
    "subscription_fixed_cost_usd",
]
