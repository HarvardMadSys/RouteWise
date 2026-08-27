"""Integrate llm-routewise into an application: route, dispatch, report, adapt.

This is a copyable application adapter, not a benchmark or a simulator. It
routes five requests across two deterministic in-process backends and reports
every outcome back to the router:

- Requests 1-2 cover both providers once (cold-start exploration).
- Request 3 goes to the provider with the faster observed TTFT.
- Request 4 times out before the first token. A pre-TTFT health failure
  enters latency learning as one synthetic 60,000 ms observation in the
  routing objective's mean; success-only TTFT percentiles ignore it, so the
  final stats still report premium at 240 ms and list the timeout under
  errors instead.
- Request 5 routes to the healthy provider because that penalty changed the
  objective. This sequential two-provider example does not reach the default
  three-failure cooldown.

Reporting the outcome (``completed`` or ``failed``) is what releases the
router's per-attempt bookkeeping; the ``settle`` call afterwards only closes
the billing state so no attempt is left unsettled.

The script needs no network access, API keys, or third-party packages, and
its output is deterministic.
"""

from __future__ import annotations

from typing import NamedTuple

import llm_routewise as rw


class Response(NamedTuple):
    ttft_ms: float
    output_tokens: int


class MockBackend:
    """Deterministic stand-in for one provider's API endpoint."""

    def __init__(self, name: str, *, ttft_ms: float) -> None:
        self.name = name
        self.ttft_ms = ttft_ms
        self._fail_next = False

    def fail_next(self) -> None:
        self._fail_next = True

    def complete(self, prompt: str) -> Response:
        if self._fail_next:
            self._fail_next = False
            raise TimeoutError(f"{self.name} produced no first token before the deadline")
        return Response(ttft_ms=self.ttft_ms, output_tokens=110)


def route_request(
    router: rw.Router,
    backends: dict[str, MockBackend],
    prompt: str,
    *,
    input_tokens: int,
) -> tuple[rw.Decision, str]:
    """Route one request, dispatch it, and report the outcome to the router."""
    decision = router.route(input_tokens=input_tokens, estimated_output_tokens=120)
    backend = backends[decision.provider]
    try:
        response = backend.complete(prompt)
    except TimeoutError:
        decision.failed(kind="health", code="timeout")
        decision.settle(cost_usd=0.0)
        return decision, "timeout"
    decision.completed(ttft_ms=response.ttft_ms, output_tokens=response.output_tokens)
    return decision, "ok"


def main() -> None:
    router = rw.Router(
        [
            rw.Provider("premium", price_in=3.00, price_out=15.00),
            rw.Provider("budget", price_in=0.15, price_out=0.60),
        ],
        alpha=1.0,  # Full cost budget: optimize latency across the whole price range.
        seed=7,
    )
    backends = {
        "premium": MockBackend("premium", ttft_ms=240.0),
        "budget": MockBackend("budget", ttft_ms=900.0),
    }

    selections = []
    for request_id in range(1, 6):
        if request_id == 4:
            backends["premium"].fail_next()
        decision, outcome = route_request(
            router,
            backends,
            f"prompt {request_id}",
            input_tokens=800,
        )
        selections.append(decision.provider)
        print(f"request {request_id}: {decision.provider:7s} {outcome}")
        if request_id in (3, 5):
            print(f"  {decision.explain()}")

    assert selections == ["budget", "premium", "premium", "premium", "budget"]

    print("\nrouter stats:")
    for name, snapshot in router.stats().providers.items():
        errors = {
            code: count
            for kind in ("health", "request")
            for code, count in snapshot["errors"][kind].items()
        }
        print(
            f"  {name:7s} selections={snapshot['primary_selections']} "
            f"ttft_p50_ms={snapshot['ttft_p50_ms']} errors={errors or 'none'} "
            f"spend=${snapshot['calculated_spend_usd']:.6f} "
            f"unsettled_attempts={snapshot['unsettled_attempts']}"
        )


if __name__ == "__main__":
    main()
