"""Shared fixtures for server tests."""

import asyncio
import contextlib
import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient

from routing.executor import RouteExecutor
from serving.servers.deps import AppServices
from serving.servers.rate_limiter import PersistentRateLimiter
from serving.storage.database import DatabaseLogger


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables for testing."""
    test_env = {
        "DB_ENABLED": "false",  # Disable DB in tests by default
        "RATE_LIMIT_ENABLED": "0",  # Disable rate limiting in tests
        "MODELS_CONFIG": "test/fixtures/test_models.yaml",
        "ROUTING_CONFIG": "test/fixtures/test_routing.yaml",
        "LOCAL_BASE_URL": "http://localhost:8001",
        "OFFLOAD": "0",
    }
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    return test_env


@pytest.fixture
def mock_router():
    """Create a mock RouteExecutor with test routes."""
    router = RouteExecutor()

    # Create mock adapter
    mock_adapter = MagicMock()
    mock_adapter.config.provider = "test"
    mock_adapter.config.base_url = "http://test.local"
    mock_adapter.config.context_length = 8192
    mock_adapter.config.max_output_length = 4096
    mock_adapter.config.supported_params = ["temperature", "max_tokens"]
    mock_adapter.config.supports_tools = False
    mock_adapter.config.supports_structured_output = False
    mock_adapter.config.id = "test-model"
    mock_adapter.config.name = "Test Model"
    mock_adapter.config.quantization = "bf16"
    mock_adapter.config.input_modalities = ["text"]
    mock_adapter.config.output_modalities = ["text"]
    mock_adapter.config.pricing = {"prompt": "0", "completion": "0"}

    # Register test route
    router.register_route("test-model", [(mock_adapter, 1.0)])

    return router


@pytest.fixture
def mock_db_logger():
    """Create a mock database logger."""
    logger = MagicMock(spec=DatabaseLogger)
    logger.initialize = AsyncMock()
    logger.cleanup = AsyncMock()
    logger.log_request = AsyncMock()
    logger.get_stats = AsyncMock(return_value=[])
    return logger


@pytest.fixture
def mock_rate_limiter():
    """Create a mock rate limiter."""
    limiter = MagicMock(spec=PersistentRateLimiter)
    limiter.initialize = AsyncMock()
    limiter._persist_state = AsyncMock()
    limiter.acquire_tokens = AsyncMock(return_value=(True, {}))
    limiter.release_tokens = AsyncMock()
    limiter.get_status = MagicMock(
        return_value={
            "configured": True,
            "capacity": 1000000,
            "tokens_available": 1000000,
            "window_seconds": 60,
        }
    )
    limiter.get_metrics = MagicMock(return_value={})
    limiter.reset_circuit_breaker = MagicMock()
    return limiter


@pytest_asyncio.fixture
async def app_services(mock_router, mock_db_logger, mock_rate_limiter):
    """Create AppServices instance for testing."""
    services = AppServices(
        router=mock_router,
        db_logger=mock_db_logger,
        rate_limiter=mock_rate_limiter,
        routing_manager=None,
    )
    yield services

    # Cleanup
    if services.db_logger:
        with contextlib.suppress(Exception):
            # Suppress teardown errors to avoid masking test results
            await services.db_logger.cleanup()
    if services.rate_limiter:
        with contextlib.suppress(Exception):
            # Suppress teardown errors to avoid masking test results
            await services.rate_limiter._persist_state()


@pytest_asyncio.fixture
async def test_app(app_services):
    """Create a FastAPI app instance for testing."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = app_services
        yield
        # Cleanup handled by app_services fixture

    app = FastAPI(title="Test API Server", version="2.0.0", lifespan=lifespan)

    # Import and register routes from new modular structure
    from serving.servers.routers import health, models

    # Include routers (they have their own routes defined)
    app.include_router(health.router)
    app.include_router(models.router)

    return app


@pytest_asyncio.fixture
async def test_client(test_app):
    """Create an async test client."""
    from httpx import ASGITransport

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def temp_models_yaml(tmp_path):
    """Create a temporary models.yaml for testing."""
    models_yaml = tmp_path / "test_models.yaml"
    content = """
models:
  - id: test-model-1
    name: Test Model 1
    provider: vllm
    base_url: ${LOCAL_BASE_URL}
    context_length: 8192
    max_output_length: 4096
    aliases: ["test-alias-1"]
    route:
      - kind: vllm
        weight: 1.0
        base_url: ${LOCAL_BASE_URL}

  - id: test-model-2
    name: Test Model 2
    provider: llama
    base_url: http://remote.test
    api_key: test-key
    context_length: 16384
    max_output_length: 8192
    aliases: ["test-alias-2"]
    route:
      - kind: llama
        weight: 1.0
        base_url: http://remote.test
        api_key: test-key
"""
    models_yaml.write_text(content)
    return str(models_yaml)


@pytest.fixture
def temp_routing_yaml(tmp_path):
    """Create a temporary routing.yaml for testing."""
    routing_yaml = tmp_path / "test_routing.yaml"
    content = """
routing_strategy: fixed
routing_parameter:
  local_fraction: 0.7
timeout: 2
health_check: 0
local_deployment:
  - endpoint: ${LOCAL_BASE_URL:-http://localhost:8001}
    models:
      - test-model-1
remote_deployment:
  - endpoint: http://remote.test
    models:
      - test-model-2
"""
    routing_yaml.write_text(content)
    return str(routing_yaml)
