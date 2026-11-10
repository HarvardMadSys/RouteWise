"""Contract tests for caller-supplied locality identities."""

from __future__ import annotations

from llm_routewise.facade import Provider, Router


class DeterministicClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _warm(router: Router, provider: str) -> None:
    for _ in range(5):
        router.observe(provider, ttft_ms=100.0)


def test_credential_identity_shares_evidence_between_unrelated_prompts() -> None:
    """A credential hash cannot isolate prefix-locality observations."""
    clock = DeterministicClock()
    router = Router(
        [
            Provider("A", price_in=2.0, price_out=1.0, price_cached=0.2),
            Provider("B", price_in=1.0, price_out=1.0),
        ],
        cold_start="require_observations",
        seed=1,
        clock=clock,
    )
    _warm(router, "A")
    _warm(router, "B")
    router._locality_estimator.record(
        "A", "api-key-hash", cached_tokens=90, input_tokens=100, now=clock.now
    )

    # Unrelated conversations that reuse the credential hash inherit the
    # same observation. A conversation-specific key remains isolated.
    credential_scoped = router.route(
        input_tokens=100,
        affinity_key="api-key-hash",
        estimated_output_tokens=10,
    )
    conversation_scoped = router.route(
        input_tokens=100,
        affinity_key="conversation:unrelated",
        estimated_output_tokens=10,
    )

    assert credential_scoped._estimated_cached_tokens["A"] == 90
    assert conversation_scoped._estimated_cached_tokens["A"] == 0
