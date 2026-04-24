"""Configuration management for experiments."""

import logging
from pathlib import Path
from typing import Any

import yaml

from legacy.experiment.data.schema import ProviderConfig, ProviderType

logger = logging.getLogger(__name__)


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
                providers[model_id] = ProviderConfig(
                    name=model["name"],
                    type=ProviderType.SUBSCRIPTION,
                    monthly_fee=float(pricing.get("monthly_fee", 0)),
                    daily_quota=int(pricing.get("daily_quota", 0)),
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
            "subscriptions": self.raw_config.get("subscriptions", {}),
            "plans": self.raw_config.get("plans", {}),
            "active_plans": self.raw_config.get("active_plans"),
        }
