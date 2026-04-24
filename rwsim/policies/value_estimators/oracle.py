"""Oracle predictor that uses ground-truth output length from trace replay.

This predictor "cheats" by reading the true response_tokens from the request
object. It serves as an upper bound on what any online predictor can achieve,
isolating the impact of prediction error on end-to-end routing cost.

Usage:
    This predictor is used in the oracle sensitivity experiment to answer:
    "If we had perfect output length prediction, how much would routing improve?"

    The gap between PD-Oracle and PD-EMA quantifies the cost of prediction error.
    The gap between PD-Oracle and Optimal quantifies the cost of online decisions.
"""

from rwsim.policies.value_estimators.base import (
    OutputTokenPredictor,
    QuantilePrediction,
)
from rwsim.schemas import Request


class OracleOutputPredictor(OutputTokenPredictor):
    """Oracle predictor using true output length from trace data.

    Returns the actual response_tokens as the prediction for all quantiles.
    This represents the best possible predictor -- one that knows the exact
    output length before routing.

    Note: This violates the causal constraint (cannot know output length
    before generation), so it is only used as an analytical upper bound.
    """

    def predict(self, request: Request) -> QuantilePrediction:
        """Return true output length as prediction.

        Args:
            request: The incoming request (with ground-truth response_tokens)

        Returns:
            QuantilePrediction with all quantiles set to true output length
        """
        true_len = max(float(request.response_tokens), 1.0)
        return QuantilePrediction(
            q10=true_len,
            q50=true_len,
            q90=true_len,
            is_warmed_up=True,
        )

    def update(self, request: Request) -> None:
        """No-op: oracle has no state to update."""
        pass

    def reset(self) -> None:
        """No-op: oracle has no state to reset."""
        pass

    @property
    def is_warmed_up(self) -> bool:
        """Oracle is always warmed up."""
        return True
