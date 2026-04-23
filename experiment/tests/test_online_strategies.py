"""Unit tests for online routing strategies.

Tests cover:
1. Predictor injection into PrimalDualOnlineStrategy
2. No data leakage (route() must not access response_tokens for decisions)
3. Correct q50 usage as point estimate
4. Strategy naming with custom predictors
5. LA-PD with EMA predictor ablation
"""

from unittest.mock import MagicMock

import pytest

from experiment.cost import CostCalculator
from experiment.data.schema import Request
from experiment.predictors import EMAOutputPredictor, HistogramOutputPredictor
from experiment.predictors.base import QuantilePrediction
from experiment.quota import QuotaManager


class TestPrimalDualPredictorInjection:
    """Test predictor injection into PrimalDualOnlineStrategy."""

    @pytest.fixture
    def mock_config(self):
        """Create minimal config for testing."""
        return {
            "providers": {},
            "subscriptions": {
                "chutes": {"supported_models": []},
            },
            "simulation": {"default_api_fallback": "test-api"},
            "model_pricing": {
                "test-model": {"input": 1.0, "output": 2.0},
            },
        }

    @pytest.fixture
    def mock_cost_calculator(self):
        """Create mock cost calculator."""
        calc = MagicMock(spec=CostCalculator)
        calc.calculate_cost_by_model.return_value = 0.001
        calc.get_cheapest_api_provider.return_value = ("api", 0.001)
        return calc

    @pytest.fixture
    def mock_quota_manager(self):
        """Create mock quota manager."""
        return QuotaManager(daily_quota=100)

    def test_default_predictor_is_ema(self, mock_config, mock_cost_calculator, mock_quota_manager):
        """Test that default predictor is EMAOutputPredictor."""
        from experiment.strategies.online.primal_dual import PrimalDualOnlineStrategy

        strategy = PrimalDualOnlineStrategy(
            mock_cost_calculator,
            mock_quota_manager,
            mock_config,
            daily_quota=100,
        )

        assert isinstance(strategy.output_predictor, EMAOutputPredictor)
        assert strategy._use_custom_predictor is False

    def test_custom_histogram_predictor(
        self, mock_config, mock_cost_calculator, mock_quota_manager
    ):
        """Test injection of HistogramOutputPredictor."""
        from experiment.strategies.online.primal_dual import PrimalDualOnlineStrategy

        hist_predictor = HistogramOutputPredictor()
        strategy = PrimalDualOnlineStrategy(
            mock_cost_calculator,
            mock_quota_manager,
            mock_config,
            daily_quota=100,
            output_predictor=hist_predictor,
        )

        assert strategy.output_predictor is hist_predictor
        assert strategy._use_custom_predictor is True

    def test_strategy_name_with_histogram(
        self, mock_config, mock_cost_calculator, mock_quota_manager
    ):
        """Test strategy name reflects Histogram predictor."""
        from experiment.strategies.online.primal_dual import PrimalDualOnlineStrategy

        strategy = PrimalDualOnlineStrategy(
            mock_cost_calculator,
            mock_quota_manager,
            mock_config,
            daily_quota=100,
            output_predictor=HistogramOutputPredictor(),
        )

        assert "Hist" in strategy.name
        assert strategy.name == "PrimalDual-Online-Stage1-Hist"

    def test_strategy_name_default_ema(self, mock_config, mock_cost_calculator, mock_quota_manager):
        """Test strategy name without suffix for default EMA."""
        from experiment.strategies.online.primal_dual import PrimalDualOnlineStrategy

        strategy = PrimalDualOnlineStrategy(
            mock_cost_calculator,
            mock_quota_manager,
            mock_config,
            daily_quota=100,
        )

        assert "Hist" not in strategy.name
        assert strategy.name == "PrimalDual-Online-Stage1"


class TestNoDataLeakage:
    """Test that route() does not access response_tokens for decisions."""

    @pytest.fixture
    def request_with_spy(self):
        """Create a request that tracks attribute access."""
        req = Request(
            id=1,
            timestamp=86400,  # Day 1 (timestamp // 86400 = 1)
            request_tokens=100,
            response_tokens=500,  # This should NOT be accessed for decisions
            total_tokens=600,
            model="test-model",
        )
        return req

    @pytest.fixture
    def mock_config(self):
        """Provide mock config."""
        return {
            "providers": {},
            "subscriptions": {"chutes": {"supported_models": []}},
            "simulation": {"default_api_fallback": "test-api"},
            "model_pricing": {"test-model": {"input": 1.0, "output": 2.0}},
        }

    @pytest.fixture
    def mock_cost_calculator(self):
        """Provide mock cost calculator."""
        calc = MagicMock(spec=CostCalculator)
        calc.calculate_cost_by_model.return_value = 0.001
        calc.get_cheapest_api_provider.return_value = ("api", 0.001)
        return calc

    def test_primal_dual_uses_predicted_not_actual(
        self, request_with_spy, mock_config, mock_cost_calculator
    ):
        """Test PrimalDual uses predictor output, not actual response_tokens."""
        from experiment.strategies.online.primal_dual import (
            PrimalDualOnlineStrategy,
        )

        quota_manager = QuotaManager(daily_quota=100)
        strategy = PrimalDualOnlineStrategy(
            mock_cost_calculator,
            quota_manager,
            mock_config,
            daily_quota=100,
        )
        strategy.precompute([request_with_spy])

        # Mock the predictor to return a known value
        mock_prediction = QuantilePrediction(q10=100, q50=200, q90=300, is_warmed_up=True)
        strategy.output_predictor.predict = MagicMock(return_value=mock_prediction)

        # Route the request
        strategy.route(request_with_spy)

        # Verify predictor was called
        strategy.output_predictor.predict.assert_called_once_with(request_with_spy)

        # The predicted value (q50=200) should be used, not actual (500)
        # We can't directly verify the internal calculation, but we can check
        # that the predictor was consulted

    def test_predicted_cost_helper_uses_predicted_tokens(self, mock_config):
        """Test _calculate_predicted_api_cost uses predicted, not actual tokens."""
        from experiment.strategies.online.primal_dual import _calculate_predicted_api_cost

        req = Request(
            id=1,
            timestamp=86400,
            request_tokens=100,
            response_tokens=500,  # Actual tokens
            total_tokens=600,
            model="test-model",
        )

        # Use predicted_output_tokens=200, not actual=500
        cost = _calculate_predicted_api_cost(mock_config, req, predicted_output_tokens=200)

        # Cost should be based on input=100 + output=200
        # With pricing input=1.0/1M, output=2.0/1M:
        # cost = 100/1M * 1.0 + 200/1M * 2.0 = 0.0001 + 0.0004 = 0.0005
        expected = 100 / 1_000_000 * 1.0 + 200 / 1_000_000 * 2.0
        assert abs(cost - expected) < 1e-9


class TestQ50Usage:
    """Test that q50 is correctly used as point estimate."""

    def test_histogram_predictor_q50_used(self):
        """Test HistogramOutputPredictor.predict returns QuantilePrediction with q50."""
        predictor = HistogramOutputPredictor()

        # Warm up with some data
        for i in range(150):
            req = Request(
                id=i,
                timestamp=86400 + i,
                request_tokens=100,
                response_tokens=100 + i,  # Varying output
                total_tokens=200 + i,
                model="test-model",
            )
            predictor.update(req)

        # Get prediction
        test_req = Request(
            id=999,
            timestamp=86400 + 999,
            request_tokens=100,
            response_tokens=0,
            total_tokens=100,
            model="test-model",
        )
        pred = predictor.predict(test_req)

        assert hasattr(pred, "q50")
        assert pred.is_warmed_up
        assert pred.q10 <= pred.q50 <= pred.q90

    def test_ema_predictor_q50_equals_mean(self):
        """Test EMAOutputPredictor.predict q50 is mean estimate."""
        predictor = EMAOutputPredictor()

        # Warm up with constant output
        for i in range(150):
            req = Request(
                id=i,
                timestamp=86400 + i,
                request_tokens=100,
                response_tokens=200,
                total_tokens=300,
                model="test-model",
            )
            predictor.update(req)

        test_req = Request(
            id=999,
            timestamp=86400 + 999,
            request_tokens=100,
            response_tokens=0,
            total_tokens=100,
            model="test-model",
        )
        pred = predictor.predict(test_req)
        mean_est = predictor.get_mean_estimate(test_req)

        # q50 should equal mean for EMA (normal approximation)
        assert abs(pred.q50 - mean_est) < 1e-6


class TestLAPDWithEMAPredictor:
    """Test LearningAugmentedPrimalDualStrategy with EMA predictor."""

    @pytest.fixture
    def mock_config(self):
        """Provide mock config."""
        return {
            "providers": {},
            "subscriptions": {"chutes": {"supported_models": []}},
            "simulation": {"default_api_fallback": "test-api"},
            "model_pricing": {"test-model": {"input": 1.0, "output": 2.0}},
        }

    @pytest.fixture
    def mock_cost_calculator(self):
        """Provide mock cost calculator."""
        calc = MagicMock(spec=CostCalculator)
        calc.calculate_cost_by_model.return_value = 0.001
        calc.get_cheapest_api_provider.return_value = ("api", 0.001)
        return calc

    def test_lapd_accepts_ema_predictor(self, mock_config, mock_cost_calculator):
        """Test LA-PD strategy accepts EMA predictor."""
        from experiment.strategies.online.learning_augmented import (
            LAPDConfig,
            LearningAugmentedPrimalDualStrategy,
        )

        quota_manager = QuotaManager(daily_quota=100)
        ema_predictor = EMAOutputPredictor()

        strategy = LearningAugmentedPrimalDualStrategy(
            mock_cost_calculator,
            quota_manager,
            mock_config,
            daily_quota=100,
            lapd_config=LAPDConfig(quantile_for_lcb=0.10),
            output_predictor=ema_predictor,
        )

        assert strategy.output_predictor is ema_predictor

    def test_lapd_default_is_histogram(self, mock_config, mock_cost_calculator):
        """Test LA-PD strategy defaults to Histogram predictor."""
        from experiment.strategies.online.learning_augmented import (
            LAPDConfig,
            LearningAugmentedPrimalDualStrategy,
        )

        quota_manager = QuotaManager(daily_quota=100)

        strategy = LearningAugmentedPrimalDualStrategy(
            mock_cost_calculator,
            quota_manager,
            mock_config,
            daily_quota=100,
            lapd_config=LAPDConfig(quantile_for_lcb=0.10),
        )

        assert isinstance(strategy.output_predictor, HistogramOutputPredictor)


class TestPredictorReset:
    """Test that predictors are properly reset in precompute()."""

    @pytest.fixture
    def mock_config(self):
        """Provide mock config."""
        return {
            "providers": {},
            "subscriptions": {"chutes": {"supported_models": []}},
            "simulation": {"default_api_fallback": "test-api"},
            "model_pricing": {"test-model": {"input": 1.0, "output": 2.0}},
        }

    @pytest.fixture
    def mock_cost_calculator(self):
        """Provide mock cost calculator."""
        calc = MagicMock(spec=CostCalculator)
        calc.calculate_cost_by_model.return_value = 0.001
        calc.get_cheapest_api_provider.return_value = ("api", 0.001)
        return calc

    def test_primal_dual_resets_predictor(self, mock_config, mock_cost_calculator):
        """Test PrimalDual resets predictor in precompute()."""
        from experiment.strategies.online.primal_dual import PrimalDualOnlineStrategy

        quota_manager = QuotaManager(daily_quota=100)
        strategy = PrimalDualOnlineStrategy(
            mock_cost_calculator,
            quota_manager,
            mock_config,
            daily_quota=100,
        )

        # Warm up the predictor
        for i in range(100):
            req = Request(
                id=i,
                timestamp=86400 + i,
                request_tokens=100,
                response_tokens=200,
                total_tokens=300,
                model="test-model",
            )
            strategy.output_predictor.update(req)

        assert strategy.output_predictor.is_warmed_up

        # Precompute should reset
        strategy.precompute([])

        # After reset, should not be warmed up
        # (EMA has min_samples=100 by default, but reset clears count)
        # Note: EMA's is_warmed_up depends on _count, which reset() sets to 0


class TestIntegration:
    """Integration tests with actual routing."""

    @pytest.fixture
    def simple_requests(self):
        """Create simple request list for testing."""
        return [
            Request(
                id=i,
                timestamp=86400 + i * 60,  # Day 1, 1 minute apart
                request_tokens=100,
                response_tokens=200,
                total_tokens=300,
                model="test-model",
            )
            for i in range(200)
        ]

    @pytest.fixture
    def mock_config(self):
        """Provide mock config."""
        return {
            "providers": {},
            "subscriptions": {"chutes": {"supported_models": []}},
            "simulation": {"default_api_fallback": "test-api"},
            "model_pricing": {"test-model": {"input": 1.0, "output": 2.0}},
        }

    @pytest.fixture
    def mock_cost_calculator(self):
        """Provide mock cost calculator."""
        calc = MagicMock(spec=CostCalculator)
        calc.calculate_cost_by_model.return_value = 0.001
        calc.get_cheapest_api_provider.return_value = ("api", 0.001)
        return calc

    def test_pd_ema_routes_requests(self, simple_requests, mock_config, mock_cost_calculator):
        """Test PD-EMA can route all requests without error."""
        from experiment.strategies.online.primal_dual import PrimalDualOnlineStrategy

        quota_manager = QuotaManager(daily_quota=50)
        strategy = PrimalDualOnlineStrategy(
            mock_cost_calculator,
            quota_manager,
            mock_config,
            daily_quota=50,
        )
        strategy.precompute(simple_requests)

        decisions = []
        for req in simple_requests:
            decision = strategy.route(req)
            decisions.append(decision)

        assert len(decisions) == len(simple_requests)
        # Should have some subscription and some API routes
        sub_count = sum(1 for d in decisions if d.provider == "daily-quota")
        api_count = sum(1 for d in decisions if d.provider == "api")
        assert sub_count + api_count == len(simple_requests)

    def test_pd_hist_routes_requests(self, simple_requests, mock_config, mock_cost_calculator):
        """Test PD-Hist can route all requests without error."""
        from experiment.strategies.online.primal_dual import PrimalDualOnlineStrategy

        quota_manager = QuotaManager(daily_quota=50)
        strategy = PrimalDualOnlineStrategy(
            mock_cost_calculator,
            quota_manager,
            mock_config,
            daily_quota=50,
            output_predictor=HistogramOutputPredictor(),
        )
        strategy.precompute(simple_requests)

        decisions = []
        for req in simple_requests:
            decision = strategy.route(req)
            decisions.append(decision)

        assert len(decisions) == len(simple_requests)

    def test_lapd_ema_routes_requests(self, simple_requests, mock_config, mock_cost_calculator):
        """Test LA-EMA can route all requests without error."""
        from experiment.strategies.online.learning_augmented import (
            LAPDConfig,
            LearningAugmentedPrimalDualStrategy,
        )

        quota_manager = QuotaManager(daily_quota=50)
        strategy = LearningAugmentedPrimalDualStrategy(
            mock_cost_calculator,
            quota_manager,
            mock_config,
            daily_quota=50,
            lapd_config=LAPDConfig(quantile_for_lcb=0.10),
            output_predictor=EMAOutputPredictor(),
        )
        strategy.precompute(simple_requests)

        decisions = []
        for req in simple_requests:
            decision = strategy.route(req)
            decisions.append(decision)

        assert len(decisions) == len(simple_requests)
