"""Tests for the public stateless routing API."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from llm_routewise.errors import RouteWiseError, ValidationError
from llm_routewise.stateless import Candidate, RouteOnceResult, route_once


class StubRng:
    def __init__(self, *draws: object) -> None:
        self._draws = iter(draws)
        self.calls = 0

    def random(self) -> object:
        self.calls += 1
        return next(self._draws)


def _mixed_candidates() -> list[Candidate]:
    return [
        Candidate("fast_expensive", cost_usd=3.0, latency_ms=100.0),
        Candidate("slow_cheap", cost_usd=1.0, latency_ms=300.0),
    ]


def test_route_once_solves_budget_mixture_and_samples_it() -> None:
    result = route_once(_mixed_candidates(), alpha=0.25, rng=StubRng(0.10))

    assert result.provider == "fast_expensive"
    assert result.budget_usd == pytest.approx(1.5)
    assert result.weights == pytest.approx({"fast_expensive": 0.25, "slow_cheap": 0.75})


def test_route_once_draws_from_later_mixture_segment() -> None:
    result = route_once(_mixed_candidates(), alpha=0.25, rng=StubRng(0.75))

    assert result.provider == "slow_cheap"


def test_route_once_calls_reusable_rng_once_per_call() -> None:
    rng = StubRng(0.10, 0.90)

    first = route_once(_mixed_candidates(), alpha=0.25, rng=rng)
    second = route_once(_mixed_candidates(), alpha=0.25, rng=rng)

    assert first.provider == "fast_expensive"
    assert second.provider == "slow_cheap"
    assert rng.calls == 2


def test_route_once_fixed_seed_replays_same_draw() -> None:
    providers = [route_once(_mixed_candidates(), alpha=0.5, seed=7).provider for _ in range(5)]

    assert len(set(providers)) == 1


def test_route_once_default_rng_is_not_shared_between_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []

    class RecordingRandom(StubRng):
        def __init__(self, seed: int | None) -> None:
            super().__init__(0.0)
            created.append(seed)

    monkeypatch.setattr("llm_routewise.stateless.random.Random", RecordingRandom)

    route_once(_mixed_candidates(), alpha=0.5)
    route_once(_mixed_candidates(), alpha=0.5)

    assert created == [None, None]


def test_route_once_uses_cost_tiebreak_for_equal_latency() -> None:
    result = route_once(
        [
            Candidate("expensive", cost_usd=3.0, latency_ms=100.0),
            Candidate("cheap", cost_usd=1.0, latency_ms=100.0),
        ],
        alpha=1.0,
        rng=StubRng(0.0),
    )

    assert result.provider == "cheap"
    assert result.weights == {"cheap": 1.0}


def test_route_once_one_candidate_is_one_hot() -> None:
    result = route_once(
        [Candidate("only", cost_usd=2, latency_ms=200)],
        alpha=0,
        rng=StubRng(0.99),
    )

    assert result == RouteOnceResult("only", {"only": 1.0}, 2.0)


def test_candidate_and_result_are_immutable() -> None:
    candidate = Candidate("p", cost_usd=1, latency_ms=2)
    result = route_once([candidate], alpha=0.5, seed=1)

    assert isinstance(result.weights, MappingProxyType)
    with pytest.raises(FrozenInstanceError):
        candidate.cost_usd = 7  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.weights["p"] = 0.5  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.provider = "other"  # type: ignore[misc]


@pytest.mark.parametrize("name", ["", None, 7])
def test_candidate_rejects_invalid_name(name: object) -> None:
    with pytest.raises(ValidationError):
        Candidate(name, cost_usd=1.0, latency_ms=1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cost_usd", -1.0),
        ("cost_usd", float("nan")),
        ("cost_usd", float("inf")),
        ("cost_usd", True),
        ("latency_ms", -1.0),
        ("latency_ms", float("nan")),
        ("latency_ms", float("inf")),
        ("latency_ms", "1"),
    ],
)
def test_candidate_rejects_invalid_numeric_fields(field: str, value: object) -> None:
    kwargs: dict[str, object] = {"cost_usd": 1.0, "latency_ms": 1.0, field: value}

    with pytest.raises(ValidationError):
        Candidate("p", **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("alpha", [-0.01, 1.01, float("nan"), float("inf"), True, "0.5"])
def test_route_once_rejects_invalid_alpha(alpha: object) -> None:
    with pytest.raises(ValidationError):
        route_once(_mixed_candidates(), alpha=alpha)  # type: ignore[arg-type]


def test_route_once_rejects_empty_candidates() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        route_once([], alpha=0.5)


def test_route_once_rejects_duplicate_names_without_drawing() -> None:
    rng = StubRng(0.0)

    with pytest.raises(ValidationError, match="unique"):
        route_once(
            [Candidate("p", 1, 2), Candidate("p", 2, 1)],
            alpha=0.5,
            rng=rng,
        )

    assert rng.calls == 0


def test_route_once_rejects_non_candidate_values() -> None:
    with pytest.raises(ValidationError, match="Candidate"):
        route_once([object()], alpha=0.5)  # type: ignore[list-item]


@pytest.mark.parametrize("seed", [1.5, "7", True])
def test_route_once_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValidationError, match="seed"):
        route_once(_mixed_candidates(), alpha=0.5, seed=seed)  # type: ignore[arg-type]


def test_route_once_rejects_seed_and_rng_together() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        route_once(_mixed_candidates(), alpha=0.5, seed=7, rng=StubRng(0.0))


@pytest.mark.parametrize("draw", [-0.1, 1.0, float("nan"), float("inf"), True, "0.5"])
def test_route_once_rejects_invalid_rng_draw(draw: object) -> None:
    with pytest.raises(ValidationError, match=r"\[0, 1\)"):
        route_once(_mixed_candidates(), alpha=0.5, rng=StubRng(draw))


def test_route_once_wraps_rng_failure_as_validation_error() -> None:
    class FailingRng:
        def random(self) -> float:
            raise RuntimeError("broken")

    with pytest.raises(ValidationError, match="failed") as error:
        route_once(_mixed_candidates(), alpha=0.5, rng=FailingRng())

    assert isinstance(error.value, RouteWiseError)
    assert isinstance(error.value.__cause__, RuntimeError)


def test_route_once_rejects_object_without_random_method() -> None:
    with pytest.raises(ValidationError, match="random"):
        route_once(_mixed_candidates(), alpha=0.5, rng=object())  # type: ignore[arg-type]


def test_route_once_accepts_generator_candidates() -> None:
    result = route_once(
        (candidate for candidate in _mixed_candidates()),
        alpha=0.25,
        rng=StubRng(0.99),
    )

    assert result.provider == "slow_cheap"
