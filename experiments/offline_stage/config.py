"""Configuration management for experiments."""

import logging
from pathlib import Path
from typing import Any

import yaml

from experiments.subscriptions import SubscriptionPlan, load_subscription_plans
from routewise.offline.schemas import ProviderConfig, ProviderType

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "experiment.yaml"


class ExperimentConfig:
    """Experiment configuration manager.

    Loads and parses experiment.yaml configuration file, similar to
    the design of serving/config.py.

    Attributes:
        config_path: Path to configuration file
        raw_config: Raw YAML configuration
        models: List of model configurations
        providers: Dictionary of provider configurations
        simulation: Simulation settings
        strategies: Strategy configurations
        dataset: Dataset settings
        output: Output settings
    """

    def __init__(self, config_path: str | Path):
        """Initialize configuration.

        Args:
            config_path: Path to experiment.yaml
        """
        self.config_path = Path(config_path)
        self.raw_config = self._load_yaml()

        # Parse configuration sections
        self.models = self._parse_models()
        self.providers = self._build_providers()
        self.simulation = self.raw_config.get("simulation", {})
        self.strategies = self.raw_config.get("strategies", [])
        self.dataset = self.raw_config.get("dataset", {})
        self.output = self.raw_config.get("output", {})
        self.model_pricing = self.raw_config.get("model_pricing", {})

        logger.info(f"Loaded configuration from {self.config_path}")
        logger.info(f"  Providers: {len(self.providers)}")
        logger.info(f"  Strategies: {len(self.strategies)}")

    def _load_yaml(self) -> dict[str, Any]:
        """Load YAML configuration file.

        Returns:
            Parsed YAML as dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _parse_models(self) -> list[dict]:
        """Parse models section from config.

        Returns:
            List of model configurations
        """
        return self.raw_config.get("models", [])

    def _build_providers(self) -> dict[str, ProviderConfig]:
        """Build ProviderConfig objects from models.

        Returns:
            Dictionary mapping provider ID to ProviderConfig
        """
        providers = {}

        for model in self.models:
            model_id = model["id"]
            model_type = model.get("type", "api")
            pricing = model.get("pricing", {})

            if model_type == "subscription":
                plan_id = model.get("subscription_plan")
                if plan_id:
                    monthly_fee, daily_quota, _window_sec = (
                        self._daily_quota_subscription_facts(str(plan_id))
                    )
                else:
                    monthly_fee = float(pricing.get("monthly_fee", 0))
                    daily_quota = int(pricing.get("daily_quota", 0))
                providers[model_id] = ProviderConfig(
                    name=model["name"],
                    type=ProviderType.SUBSCRIPTION,
                    monthly_fee=monthly_fee,
                    daily_quota=daily_quota,
                )
            else:
                # API provider - convert from per-1M to per-1K
                providers[model_id] = ProviderConfig(
                    name=model["name"],
                    type=ProviderType.API,
                    input_price_per_1k=float(pricing.get("prompt", 0)) / 1000.0,
                    output_price_per_1k=float(pricing.get("completion", 0)) / 1000.0,
                )

        return providers

    def get_provider(self, provider_id: str) -> ProviderConfig:
        """Get provider configuration by ID.

        Args:
            provider_id: Provider identifier

        Returns:
            Provider configuration

        Raises:
            ValueError: If provider not found
        """
        if provider_id not in self.providers:
            raise ValueError(f"Provider {provider_id} not found in config")
        return self.providers[provider_id]

    def get_subscription_provider(self) -> ProviderConfig:
        """Get the first subscription provider.

        Returns:
            First subscription provider configuration

        Raises:
            ValueError: If no subscription provider is configured
        """
        # Find first subscription provider
        for _provider_id, provider in self.providers.items():
            if provider.is_subscription():
                return provider

        raise ValueError("No subscription provider found in configuration")

    def get_api_providers(self) -> dict[str, ProviderConfig]:
        """Get all API providers.

        Returns:
            Dictionary of API providers
        """
        return {pid: p for pid, p in self.providers.items() if p.is_api()}

    def _daily_quota_subscription_facts(
        self,
        plan_id: str,
    ) -> tuple[float, int, float]:
        plan = self._subscription_plan(plan_id)
        if plan.monthly_fee_usd is None:
            raise ValueError(
                f"subscription plan {plan_id!r} cannot back offline_stage "
                "cost accounting because monthly_fee_usd is null"
            )
        window = self._single_daily_window(plan)
        return (float(plan.monthly_fee_usd), int(window.quota_requests), 86400.0)

    def _subscription_plan(self, plan_id: str) -> SubscriptionPlan:
        plans = load_subscription_plans()
        try:
            return plans[plan_id]
        except KeyError as exc:
            known = ", ".join(sorted(plans))
            raise ValueError(
                f"unknown subscription_plan {plan_id!r}; known plans: {known}"
            ) from exc

    @staticmethod
    def _single_daily_window(plan: SubscriptionPlan):
        if len(plan.quota_windows) != 1:
            raise ValueError(
                f"offline_stage legacy daily_quota path only supports one quota "
                f"window, but {plan.plan_id!r} has {len(plan.quota_windows)}"
            )
        window = plan.quota_windows[0]
        if abs(float(window.quota_window_sec) - 86400.0) > 1e-9:
            raise ValueError(
                f"offline_stage legacy daily_quota path requires a 86400s window, "
                f"but {plan.plan_id!r} has {window.quota_window_sec:g}s"
            )
        return window

    def _canonical_subscriptions(self) -> dict[str, Any]:
        subscriptions: dict[str, Any] = {}
        for name, entry in self.raw_config.get("subscriptions", {}).items():
            if not isinstance(entry, dict):
                subscriptions[name] = entry
                continue
            resolved = dict(entry)
            plan_id = resolved.get("subscription_plan")
            if plan_id:
                monthly_fee, daily_quota, window_sec = (
                    self._daily_quota_subscription_facts(str(plan_id))
                )
                resolved.setdefault("type", "daily_quota")
                resolved["monthly_fee"] = monthly_fee
                resolved["daily_quota"] = daily_quota
                resolved["quota_window_sec"] = window_sec
            subscriptions[name] = resolved
        return subscriptions

    def to_dict(self) -> dict:
        """Convert to dictionary for passing to strategies.

        Returns:
            Configuration as dictionary
        """
        return {
            "providers": self.providers,
            "simulation": self.simulation,
            "dataset": self.dataset,
            "output": self.output,
            "model_pricing": self.model_pricing,
            "subscriptions": self._canonical_subscriptions(),
            "plans": self.raw_config.get("plans", {}),
            "active_plans": self.raw_config.get("active_plans"),
        }


def load_default_config() -> ExperimentConfig:
    """Load the canonical offline/stage experiment config."""
    return ExperimentConfig(DEFAULT_CONFIG_PATH)
