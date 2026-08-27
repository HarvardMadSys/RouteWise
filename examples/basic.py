"""Run one RouteWise decision without network access or third-party packages."""

from __future__ import annotations

import llm_routewise as rw


def main() -> None:
    router = rw.Router(
        [
            rw.Provider("fast", price_in=3.00, price_out=15.00),
            rw.Provider("cheap", price_in=0.15, price_out=0.60),
        ],
        alpha=0.25,
        seed=7,
    )

    decision = router.route(
        input_tokens=800,
        estimated_output_tokens=120,
    )
    decision.completed(ttft_ms=240.0, output_tokens=110)

    print(f"selected provider: {decision.provider}")
    print(f"routing weights: {dict(decision.weights)}")
    print(f"expected cost: ${decision.expected_cost_usd:.6f}")


if __name__ == "__main__":
    main()
