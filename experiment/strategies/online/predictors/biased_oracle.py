"""Biased oracle predictor for misprediction robustness analysis.

This predictor wraps the ground-truth output length with a controlled
distortion (systematic bias and/or log-normal multiplicative noise).
It is used to evaluate how sensitive the cost router is to predictor
miscalibration.

Two knobs:
    bias_factor (float, default 1.0):
        Multiplicative systematic bias. `predicted = true * bias_factor`.
        Preserves relative ordering of requests -- isolates the effect
        of a uniform miscalibration on shadow-price routing.
    noise_std (float, default 0.0):
        Standard deviation of log-normal multiplicative noise.
        `predicted = true * bias_factor * exp(noise_std * Z)` with
        Z ~ N(0, 1). Breaks ordering -- isolates the effect of
        per-request prediction variance.

The two knobs can be combined. When both are zero (bias=1.0, noise=0.0),
this is equivalent to OracleOutputPredictor.
"""

from __future__ import annotations

import math

import numpy as np

from experiment.data.schema import Request
from experiment.predictors.base import (
    OutputTokenPredictor,
    QuantilePrediction,
)


class BiasedOraclePredictor(OutputTokenPredictor):
    """Oracle predictor with configurable systematic bias and noise.

    The predicted output length for a request with ground-truth
    `response_tokens = y` is:

        y_hat = max(1.0, y * bias_factor * exp(noise_std * Z))

    where Z ~ N(0, 1) is drawn independently for each prediction.

    Attributes:
        bias_factor: Multiplicative systematic bias applied to all predictions.
        noise_std: Standard deviation of log-normal multiplicative noise.
        _rng: NumPy random generator for reproducibility.
    """

    def __init__(
        self,
        bias_factor: float = 1.0,
        noise_std: float = 0.0,
        seed: int = 42,
    ) -> None:
        """Initialize biased oracle predictor.

        Args:
            bias_factor: Multiplicative bias (e.g., 0.5 under-estimates by 50%).
            noise_std: Log-normal noise std. 0.0 disables noise.
            seed: RNG seed for noise draws.
        """
        if bias_factor <= 0.0:
            raise ValueError(f"bias_factor must be positive, got {bias_factor}")
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")

        self.bias_factor = float(bias_factor)
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def predict(self, request: Request) -> QuantilePrediction:
        """Return distorted ground-truth output length as prediction.

        Args:
            request: Incoming request with ground-truth response_tokens.

        Returns:
            QuantilePrediction with all quantiles set to the distorted value.
        """
        true_len = max(float(request.response_tokens), 1.0)
        multiplier = self.bias_factor
        if self.noise_std > 0.0:
            z = float(self._rng.standard_normal())
            multiplier *= math.exp(self.noise_std * z)
        predicted = max(1.0, true_len * multiplier)
        return QuantilePrediction(
            q10=predicted,
            q50=predicted,
            q90=predicted,
            is_warmed_up=True,
        )

    def update(self, request: Request) -> None:
        """No-op: biased oracle has no state to update."""
        pass

    def reset(self) -> None:
        """Reset RNG to the initial seed for reproducible runs."""
        self._rng = np.random.default_rng(self.seed)

    @property
    def is_warmed_up(self) -> bool:
        """Biased oracle is always warmed up."""
        return True
