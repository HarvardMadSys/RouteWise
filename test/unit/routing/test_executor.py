from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Any

import pytest

from routing.executor import RouteExecutor
from serving.adapters.base import BaseAdapter, ModelConfig

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


def _cfg(mid: str, provider: str = "p") -> ModelConfig:
    return ModelConfig(
        id=mid,
        name=mid,
        provider=provider,
        base_url="http://test",
        context_length=8192,
        max_output_length=4096,
    )


class _EchoAdapter(BaseAdapter):
    async def chat_completion(self, messages: list[dict[str, Any]], **params) -> dict[str, Any]:
        text = params.get("content", "ok")
        return self.format_response(content=text, model=self.config.id)

    async def stream_chat_completion(
        self, messages: list[dict[str, Any]], **params
    ) -> AsyncGenerator[str, None]:  # pragma: no cover - streaming covered elsewhere
        yield self.format_stream_chunk(model=self.config.id, content="ok")


class _FailAdapter(BaseAdapter):
    async def chat_completion(self, messages: list[dict[str, Any]], **params) -> dict[str, Any]:
        raise RuntimeError("fail")

    async def stream_chat_completion(
        self, messages: list[dict[str, Any]], **params
    ) -> AsyncGenerator[str, None]:  # pragma: no cover
        raise RuntimeError("fail")


@pytest.mark.unit
def test_weighted_selection_distribution():
    exe = RouteExecutor()
    a = _EchoAdapter(_cfg("m", provider="A"))
    b = _EchoAdapter(_cfg("m", provider="B"))
    exe.register_route("m", [(a, 0.8), (b, 0.2)])

    # Seed RNG for reproducibility
    random.seed(42)
    picks = {"A": 0, "B": 0}
    for _ in range(10000):
        chosen = exe._select_adapter("m")  # type: ignore[attr-defined]
        assert chosen is not None
        picks[chosen.config.provider] += 1

    frac_a = picks["A"] / 10000
    frac_b = picks["B"] / 10000
    # Allow small tolerance around target weights
    assert 0.77 <= frac_a <= 0.83
    assert 0.17 <= frac_b <= 0.23


@pytest.mark.unit
def test_weighted_selection_chi_square():
    """Validate distribution with a chi-square test at 95% confidence without SciPy."""
    exe = RouteExecutor()
    a = _EchoAdapter(_cfg("m", provider="A"))
    b = _EchoAdapter(_cfg("m", provider="B"))
    exe.register_route("m", [(a, 0.8), (b, 0.2)])

    random.seed(7)
    n = 10000
    picks = {"A": 0, "B": 0}
    for _ in range(n):
        chosen = exe._select_adapter("m")  # type: ignore[attr-defined]
        assert chosen is not None
        picks[chosen.config.provider] += 1

    observed = [picks["A"], picks["B"]]
    expected = [0.8 * n, 0.2 * n]
    chi2 = sum(((o - e) ** 2) / e for o, e in zip(observed, expected, strict=False))
    # df=1, alpha=0.05 => critical value ~3.841
    assert chi2 < 3.841


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_on_primary_failure():
    exe = RouteExecutor()
    primary = _FailAdapter(_cfg("m", provider="primary"))
    backup = _EchoAdapter(_cfg("m", provider="backup"))
    exe.register_route("m", [(primary, 0.9), (backup, 0.1)])

    # Force primary selection by fixing RNG
    random_state = random.random
    try:
        random.random = lambda: 0.01  # always pick primary (weight 0.9)
        resp = await exe.chat_completion("m", messages=[{"role": "user", "content": "hi"}])
    finally:
        random.random = random_state
    assert resp["choices"][0]["message"]["content"] == "ok"
    assert resp["_routing"]["provider"] == "backup"
    assert resp["_routing"].get("fallback") is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_route_configured_raises():
    exe = RouteExecutor()
    with pytest.raises(ValueError):
        await exe.chat_completion("unknown", messages=[{"role": "user", "content": "hi"}])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrent_route_selection():
    """Verify concurrent selections do not race or corrupt state."""
    exe = RouteExecutor()
    echo = _EchoAdapter(_cfg("m"))
    exe.register_route("m", [(echo, 1.0)])

    async def do_one():
        r = await exe.chat_completion("m", messages=[{"role": "user", "content": "x"}])
        return r["choices"][0]["message"]["content"]

    results = await asyncio.gather(*(do_one() for _ in range(100)))
    assert all(res == "ok" for res in results)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multiple_fallback_chain():
    """Primary and secondary fail; tertiary succeeds."""
    exe = RouteExecutor()
    p1 = _FailAdapter(_cfg("m", provider="p1"))
    p2 = _FailAdapter(_cfg("m", provider="p2"))
    p3 = _EchoAdapter(_cfg("m", provider="p3"))
    exe.register_route("m", [(p1, 0.6), (p2, 0.3), (p3, 0.1)])

    # Force selecting the primary first
    orig = exe._select_adapter  # type: ignore[attr-defined]
    try:
        exe._select_adapter = lambda model_id: p1  # type: ignore[assignment]
        resp = await exe.chat_completion("m", messages=[{"role": "user", "content": "hi"}])
    finally:
        exe._select_adapter = orig  # type: ignore[assignment]

    assert resp["choices"][0]["message"]["content"] == "ok"
    assert resp["_routing"]["provider"] == "p3"
    assert resp["_routing"].get("fallback") is True


@pytest.mark.unit
@pytest.mark.perf
def test_route_selection_performance():
    """Ensure adapter selection is fast enough for basic regression budgets."""
    exe = RouteExecutor()
    a = _EchoAdapter(_cfg("m", provider="A"))
    b = _EchoAdapter(_cfg("m", provider="B"))
    exe.register_route("m", [(a, 0.8), (b, 0.2)])

    import time

    start = time.perf_counter()
    for _ in range(10000):
        _ = exe._select_adapter("m")  # type: ignore[attr-defined]
    elapsed = time.perf_counter() - start
    assert elapsed < 0.1
