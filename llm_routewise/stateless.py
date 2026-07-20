"""Stateless RouteWise budget routing.

This module exposes the smallest useful part of RouteWise: callers provide
already-priced candidates, the library solves the cost-constrained latency LP,
and one provider is sampled from the resulting mixture.  No observations or
other state are retained between calls.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping  # noqa: TC003 - public runtime annotations
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Protocol

from llm_routewise.core.lp import (
    BudgetLPCandidate,
    cost_tiebroken_objective,
    solve_budget_lp,
)
from llm_routewise.errors import RouteWiseError, ValidationError


class _RandomSource(Protocol):
    def random(self) -> float: ...


@dataclass(frozen=True, slots=True)
class Candidate:
    """An eligible provider with caller-computed request cost and latency."""

    name: str
    cost_usd: float
    latency_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValidationError("candidate name must be a non-empty string")
        object.__setattr__(
            self,
            "cost_usd",
            _finite_non_negative(self.cost_usd, name="cost_usd"),
        )
        object.__setattr__(
            self,
            "latency_ms",
            _finite_non_negative(self.latency_ms, name="latency_ms"),
        )


@dataclass(frozen=True, slots=True)
class RouteOnceResult:
    """Immutable result returned by :func:`route_once`."""

    provider: str
    weights: Mapping[str, float]
    budget_usd: float

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider:
            raise ValidationError("provider must be a non-empty string")
        try:
            copied_weights = dict(self.weights)
        except (TypeError, ValueError) as exc:
            raise ValidationError("weights must be a mapping") from exc
        if not copied_weights:
            raise ValidationError("weights must not be empty")
        for name, value in copied_weights.items():
            if not isinstance(name, str) or not name:
                raise ValidationError("weight names must be non-empty strings")
            copied_weights[name] = _finite_non_negative(value, name=f"weights[{name!r}]")
        total = sum(copied_weights.values())
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValidationError("weights must sum to 1")
        if self.provider not in copied_weights or copied_weights[self.provider] <= 0.0:
            raise ValidationError("provider must have positive weight")
        object.__setattr__(self, "weights", MappingProxyType(copied_weights))
        object.__setattr__(
            self,
            "budget_usd",
            _finite_non_negative(self.budget_usd, name="budget_usd"),
        )


def route_once(
    candidates: Iterable[Candidate],
    *,
    alpha: float,
    seed: int | None = None,
    rng: _RandomSource | None = None,
) -> RouteOnceResult:
    """Solve the RouteWise budget LP and sample one provider.

    Eligibility filtering and cost computation are intentionally the caller's
    responsibility.  Pass a reusable ``rng`` for repeated calls; passing the
    same fixed ``seed`` with identical inputs intentionally replays the same
    draw on every call.
    """

    candidate_tuple = _validated_candidates(candidates)
    alpha_value = _unit_interval(alpha, name="alpha")
    if seed is not None and rng is not None:
        raise ValidationError("seed and rng are mutually exclusive")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValidationError("seed must be an integer or None")
    if rng is not None and not callable(getattr(rng, "random", None)):
        raise ValidationError("rng must expose a callable random() method")

    costs = [candidate.cost_usd for candidate in candidate_tuple]
    latencies = [candidate.latency_ms for candidate in candidate_tuple]
    c_min = min(costs)
    c_max = max(costs)
    budget = c_min + alpha_value * (c_max - c_min)
    objective = cost_tiebroken_objective(latencies, costs)
    lp_result = solve_budget_lp(
        [
            BudgetLPCandidate(
                candidate.name,
                objective=objective[index],
                effective_cost=candidate.cost_usd,
            )
            for index, candidate in enumerate(candidate_tuple)
        ],
        budget=budget,
    )
    if not lp_result.feasible or not lp_result.weights:
        raise RouteWiseError("budget LP unexpectedly found no feasible provider")

    random_source: _RandomSource = rng if rng is not None else random.Random(seed)
    draw = _draw(random_source)
    provider = _sample(lp_result.weights, draw)
    return RouteOnceResult(
        provider=provider,
        weights=lp_result.weights,
        budget_usd=budget,
    )


def _validated_candidates(candidates: Iterable[Candidate]) -> tuple[Candidate, ...]:
    try:
        values = tuple(candidates)
    except TypeError as exc:
        raise ValidationError("candidates must be an iterable of Candidate objects") from exc
    if not values:
        raise ValidationError("candidates must not be empty")
    if any(not isinstance(candidate, Candidate) for candidate in values):
        raise ValidationError("candidates must contain only Candidate objects")
    names = [candidate.name for candidate in values]
    if len(set(names)) != len(names):
        raise ValidationError("candidate names must be unique")
    return values


def _draw(rng: _RandomSource) -> float:
    try:
        raw_draw = rng.random()
    except Exception as exc:
        raise ValidationError("rng.random() failed") from exc
    if not isinstance(raw_draw, Real) or isinstance(raw_draw, bool):
        raise ValidationError("rng.random() must return a real number in [0, 1)")
    draw = float(raw_draw)
    if not math.isfinite(draw) or not 0.0 <= draw < 1.0:
        raise ValidationError("rng.random() must return a finite number in [0, 1)")
    return draw


def _sample(weights: Mapping[str, float], draw: float) -> str:
    cumulative = 0.0
    last_name: str | None = None
    for name, weight in weights.items():
        last_name = name
        cumulative += float(weight)
        if draw < cumulative:
            return name
    if last_name is None:  # pragma: no cover - solve_budget_lp guarantees support
        raise RouteWiseError("budget LP returned an empty mixture")
    return last_name


def _finite_non_negative(value: object, *, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValidationError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValidationError(f"{name} must be a finite non-negative number")
    return result


def _unit_interval(value: object, *, name: str) -> float:
    result = _finite_non_negative(value, name=name)
    if result > 1.0:
        raise ValidationError(f"{name} must be within [0, 1]")
    return result


__all__ = ["Candidate", "RouteOnceResult", "route_once"]
