"""Tests for bootstrap module."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import pytest

from routing.executor import RouteExecutor
from routing.manager import RoutingManager
from serving.servers import bootstrap
from serving.servers.deps import AppServices


class TestBootstrapInitialization:
    """Test bootstrap initialization functions."""

    @pytest.mark.asyncio
    async def test_initialize_returns_app_services(self, mock_env):
        """Test that initialize returns properly typed AppServices."""
        with (
            patch("serving.servers.bootstrap._init_db_logger", return_value=None),
            patch("serving.servers.bootstrap._init_router_and_models", new=AsyncMock()),
            patch("serving.servers.bootstrap._apply_routing_manager", return_value=None),
            patch("serving.servers.bootstrap._configure_rate_limiter"),
        ):
            services = await bootstrap.initialize()

            assert isinstance(services, AppServices)
            assert isinstance(services.router, RouteExecutor)
            assert services.db_logger is None  # Disabled in mock_env
            assert services.rate_limiter is None  # Disabled in mock_env
            assert services.routing_manager is None

    @pytest.mark.asyncio
    async def test_initialize_with_database(self, mock_env, monkeypatch):
        """Test initialization with PostgreSQL database enabled."""
        monkeypatch.setenv("DB_ENABLED", "true")
        monkeypatch.setenv("DB_HOST", "testhost")
        monkeypatch.setenv("DB_NAME", "testdb")
        monkeypatch.setenv("DB_USER", "testuser")
        monkeypatch.setenv("DB_PASSWORD", "testpass")

        with (
            patch("serving.servers.bootstrap._init_router_and_models", new=AsyncMock()),
            patch("serving.servers.bootstrap._apply_routing_manager", return_value=None),
            patch("serving.servers.bootstrap._configure_rate_limiter"),
            patch("serving.servers.bootstrap.DatabaseLogger") as MockDBLogger,
        ):
            mock_logger = AsyncMock()
            MockDBLogger.return_value = mock_logger

            services = await bootstrap.initialize()

            assert services.db_logger is not None
            mock_logger.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_rate_limiter(self, mock_env, monkeypatch):
        """Test initialization with rate limiter enabled."""
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")

        with (
            patch("serving.servers.bootstrap._init_db_logger", return_value=None),
            patch("serving.servers.bootstrap._init_router_and_models", new=AsyncMock()),
            patch("serving.servers.bootstrap._apply_routing_manager", return_value=None),
        ):
            services = await bootstrap.initialize()

            assert services.rate_limiter is not None
            await services.rate_limiter._persist_state()

    @pytest.mark.asyncio
    async def test_initialize_loads_models_yaml(self, mock_env, temp_models_yaml, monkeypatch):
        """Test that models.yaml is loaded and registered."""
        monkeypatch.setenv("MODELS_CONFIG", temp_models_yaml)
        monkeypatch.setenv("LOCAL_BASE_URL", "http://localhost:8001")

        with (
            patch("serving.servers.bootstrap._init_db_logger", return_value=None),
            patch("serving.servers.bootstrap._apply_routing_manager", return_value=None),
            patch("serving.servers.bootstrap._configure_rate_limiter"),
        ):
            services = await bootstrap.initialize()

            # Check that routes were registered
            assert len(services.router.routes) > 0
            assert "test-model-1" in services.router.routes
            assert "test-alias-1" in services.router.routes

    @pytest.mark.asyncio
    async def test_initialize_with_routing_manager(self, mock_env, temp_routing_yaml, monkeypatch):
        """Test initialization with routing manager."""
        monkeypatch.setenv("ROUTING_CONFIG", temp_routing_yaml)
        monkeypatch.setenv("LOCAL_BASE_URL", "http://localhost:8001")

        with (
            patch("serving.servers.bootstrap._init_db_logger", return_value=None),
            patch("serving.servers.bootstrap._init_router_and_models", new=AsyncMock()),
            patch("serving.servers.bootstrap._configure_rate_limiter"),
        ):
            services = await bootstrap.initialize()

            # Routing manager should be initialized
            assert services.routing_manager is not None


class TestBootstrapShutdown:
    """Test bootstrap shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up_resources(self, app_services):
        """Test that shutdown properly cleans up all resources."""
        # Mock cleanup methods
        app_services.db_logger.cleanup = AsyncMock()
        app_services.rate_limiter._persist_state = AsyncMock()

        # Add routing manager with health monitor
        routing_manager = MagicMock()
        routing_manager.shutdown = AsyncMock()
        app_services.routing_manager = routing_manager

        await bootstrap.shutdown(app_services)

        # Verify cleanup was called
        app_services.db_logger.cleanup.assert_called_once()
        app_services.rate_limiter._persist_state.assert_called_once()
        routing_manager.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_none_services(self):
        """Test that shutdown handles None values gracefully."""
        services = AppServices(
            router=RouteExecutor(), db_logger=None, rate_limiter=None, routing_manager=None
        )

        # Should not raise any errors
        await bootstrap.shutdown(services)

    @pytest.mark.asyncio
    async def test_shutdown_handles_exceptions(self, app_services):
        """Test that shutdown continues even if cleanup fails."""
        # Make cleanup raise an exception
        app_services.db_logger.cleanup = AsyncMock(side_effect=Exception("DB cleanup failed"))
        app_services.rate_limiter._persist_state = AsyncMock()

        # Should not raise, but should log the error
        with patch("serving.servers.bootstrap.logger") as mock_logger:
            await bootstrap.shutdown(app_services)

            # Check that error was logged
            mock_logger.error.assert_called()
            # But other cleanup still happened
            app_services.rate_limiter._persist_state.assert_called_once()


class TestBootstrapHelpers:
    """Test bootstrap helper functions."""

    def test_init_db_logger_postgres(self, monkeypatch):
        """Test PostgreSQL database logger initialization."""
        monkeypatch.setenv("DB_HOST", "testhost")
        monkeypatch.setenv("DB_PORT", "5433")
        monkeypatch.setenv("DB_NAME", "testdb")
        monkeypatch.setenv("DB_USER", "testuser")
        monkeypatch.setenv("DB_PASSWORD", "testpass")

        logger = bootstrap._init_db_logger()

        assert logger is not None
        assert isinstance(logger, bootstrap.DatabaseLogger)
        assert logger.db_config["host"] == "testhost"
        assert logger.db_config["database"] == "testdb"

    def test_init_db_logger_disabled(self, monkeypatch):
        """Test database logger when disabled."""
        monkeypatch.setenv("DB_ENABLED", "false")

        logger = bootstrap._init_db_logger()

        assert logger is None

    @pytest.mark.asyncio
    async def test_init_router_with_remote_models(self, monkeypatch, tmp_path):
        """Router should register purely remote models from YAML."""
        models_yaml = tmp_path / "remote_models.yaml"
        models_yaml.write_text(
            """
models:
  - id: remote-model
    name: Remote Only Model
    provider: llama
    context_length: 8192
    max_output_length: 4096
    route:
      - kind: llama
        weight: 1.0
        base_url: https://api.example.com
        api_key: test-key
"""
        )

        monkeypatch.setenv("MODELS_CONFIG", str(models_yaml))
        monkeypatch.setenv("OFFLOAD", "0")

        router = RouteExecutor()

        await bootstrap._init_router_and_models(router)

        assert "remote-model" in router.routes
        assert len(router.routes["remote-model"].adapters) == 1

    @pytest.mark.asyncio
    async def test_init_router_with_offload_no_local(self, monkeypatch, tmp_path):
        """Hard OFFLOAD should keep remote-only models untouched."""
        models_yaml = tmp_path / "remote_models.yaml"
        models_yaml.write_text(
            """
models:
  - id: remote-model
    name: Remote Only Model
    provider: llama
    context_length: 8192
    max_output_length: 4096
    route:
      - kind: llama
        weight: 1.0
        base_url: https://api.example.com
        api_key: test-key
"""
        )

        monkeypatch.setenv("MODELS_CONFIG", str(models_yaml))
        monkeypatch.setenv("LOCAL_BASE_URL", "http://localhost:8001")
        monkeypatch.setenv("OFFLOAD", "1")

        router = RouteExecutor()

        await bootstrap._init_router_and_models(router)

        # No local adapters to remove; remote model should remain intact
        assert "remote-model" in router.routes
        assert len(router.routes["remote-model"].adapters) == 1

    @pytest.mark.asyncio
    async def test_init_router_with_deepseek(self, monkeypatch, tmp_path):
        """DeepSeek should be loaded from YAML, not env fallback."""
        models_yaml = tmp_path / "models_deepseek.yaml"
        models_yaml.write_text(
            """
models:
  - id: deepseek-chat
    name: DeepSeek Chat
    provider: deepseek
    base_url: https://api.deepseek.com/v1
    api_key: test-key
    context_length: 65536
    max_output_length: 8192
    supports_tools: true
    supports_structured_output: true
    supported_params: [temperature, top_p, max_tokens, stop, frequency_penalty, presence_penalty]
    route:
      - kind: deepseek
        weight: 1.0
        base_url: https://api.deepseek.com/v1
        api_key: test-key
"""
        )
        monkeypatch.setenv("MODELS_CONFIG", str(models_yaml))

        router = RouteExecutor()
        await bootstrap._init_router_and_models(router)
        assert "deepseek-chat" in router.routes

    @pytest.mark.asyncio
    async def test_init_router_with_gemini(self, monkeypatch, tmp_path):
        """Gemini should be loaded from YAML, not env fallback."""
        models_yaml = tmp_path / "models_gemini.yaml"
        models_yaml.write_text(
            """
models:
  - id: gemini-2.5-flash
    name: Gemini 2.5 Flash
    provider: gemini
    provider_model_id: "gemini-2.5-flash"
    base_url: https://generativelanguage.googleapis.com/v1beta
    api_key: test-key
    context_length: 1048576
    max_output_length: 8192
    supports_tools: true
    supports_structured_output: true
    supported_params: [temperature, top_p, top_k, max_tokens, stop]
    route:
      - kind: gemini
        weight: 1.0
        base_url: https://generativelanguage.googleapis.com/v1beta
        api_key: test-key
        provider_model_id: "gemini-2.5-flash"
"""
        )
        monkeypatch.setenv("MODELS_CONFIG", str(models_yaml))

        router = RouteExecutor()
        await bootstrap._init_router_and_models(router)
        assert "gemini-2.5-flash" in router.routes

    def test_configure_rate_limiter_deepseek(self, monkeypatch, mock_rate_limiter):
        """Test rate limiter configuration for DeepSeek."""
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        monkeypatch.delenv("GLM_TPH_LIMIT", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setenv("DEEPSEEK_TPD_LIMIT", "500000")

        bootstrap._configure_rate_limiter(mock_rate_limiter)

        mock_rate_limiter.configure.assert_called()
        configs = [call.args[0] for call in mock_rate_limiter.configure.call_args_list]
        deepseek_cfg = next((cfg for cfg in configs if cfg.model_id == "deepseek-chat"), None)
        assert deepseek_cfg is not None
        assert deepseek_cfg.capacity_tokens == 500000
        assert deepseek_cfg.window_seconds == 86400

    def test_configure_rate_limiter_gemini(self, monkeypatch, mock_rate_limiter):
        """Test rate limiter configuration for Gemini (both models)."""
        # Clear any existing keys first
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        monkeypatch.delenv("GLM_TPH_LIMIT", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_TPM_LIMIT", "2000000")

        bootstrap._configure_rate_limiter(mock_rate_limiter)

        # Check that configure was called for both Gemini models
        mock_rate_limiter.configure.assert_called()
        calls = mock_rate_limiter.configure.call_args_list

        # Find both Gemini model configs
        gemini_flash_config = None
        gemini_preview_config = None
        for call in calls:
            config = call.args[0]
            if config.model_id == "gemini-2.5-flash":
                gemini_flash_config = config
            elif config.model_id == "gemini-2.5-flash-preview-09-2025":
                gemini_preview_config = config

        # Verify both models are configured with same policy
        assert gemini_flash_config is not None, "Gemini flash config not found"
        assert gemini_flash_config.capacity_tokens == 2000000
        assert gemini_flash_config.window_seconds == 60

        assert gemini_preview_config is not None, "Gemini preview config not found"
        assert gemini_preview_config.capacity_tokens == 2000000
        assert gemini_preview_config.window_seconds == 60

    def test_configure_rate_limiter_glm(self, monkeypatch, mock_rate_limiter):
        """Test rate limiter configuration for GLM."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("ZAI_API_KEY", "glm-test-key")
        monkeypatch.setenv("GLM_TPH_LIMIT", "750000")

        bootstrap._configure_rate_limiter(mock_rate_limiter)

        mock_rate_limiter.configure.assert_called()
        configs = [call.args[0] for call in mock_rate_limiter.configure.call_args_list]
        glm_config = next((cfg for cfg in configs if cfg.model_id == "glm-4.5"), None)
        assert glm_config is not None
        assert glm_config.capacity_tokens == 750000
        assert glm_config.window_seconds == 3600


class TestBootstrapErrorHandling:
    """Test error handling in bootstrap."""

    @pytest.mark.asyncio
    async def test_initialize_handles_model_yaml_error(self, mock_env, monkeypatch):
        """Test that initialize continues if models.yaml fails to load."""
        monkeypatch.setenv("MODELS_CONFIG", "/nonexistent/models.yaml")

        with (
            patch("serving.servers.bootstrap._init_db_logger", return_value=None),
            patch("serving.servers.bootstrap._apply_routing_manager", return_value=None),
            patch("serving.servers.bootstrap._configure_rate_limiter"),
            patch("serving.servers.bootstrap.logger") as mock_logger,
        ):
            services = await bootstrap.initialize()

            # Should still return services
            assert isinstance(services, AppServices)
            # Should log a warning about missing models config
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_initialize_handles_routing_manager_error(self, mock_env, monkeypatch):
        """Test that initialize continues if routing manager fails."""
        monkeypatch.setenv("ROUTING_CONFIG", "/invalid/routing.yaml")

        with (
            patch("serving.servers.bootstrap._init_db_logger", return_value=None),
            patch("serving.servers.bootstrap._init_router_and_models", new=AsyncMock()),
            patch("serving.servers.bootstrap._configure_rate_limiter"),
            patch("serving.servers.bootstrap.logger") as mock_logger,
        ):
            services = await bootstrap.initialize()

            # Should still return services
            assert isinstance(services, AppServices)
            # Routing manager is optional, so None is acceptable
            assert services.routing_manager is None or isinstance(
                services.routing_manager, RoutingManager
            )
            # Should log a warning about routing config failure/missing
            mock_logger.warning.assert_called()
