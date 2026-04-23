"""Tests for dependency injection module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from routing.executor import RouteExecutor
from serving.servers.deps import (
    AppServices,
    get_db_logger,
    get_rate_limiter,
    get_router,
    get_services,
)
from serving.servers.rate_limiter import PersistentRateLimiter
from serving.storage.database import DatabaseLogger


class TestAppServices:
    """Test AppServices dataclass."""

    def test_app_services_creation(self):
        """Test creating AppServices with all fields."""
        router = RouteExecutor()
        rate_limiter = MagicMock(spec=PersistentRateLimiter)
        db_logger = MagicMock(spec=DatabaseLogger)
        routing_manager = MagicMock()

        services = AppServices(
            router=router,
            rate_limiter=rate_limiter,
            db_logger=db_logger,
            routing_manager=routing_manager,
        )

        assert services.router is router
        assert services.rate_limiter is rate_limiter
        assert services.db_logger is db_logger
        assert services.routing_manager is routing_manager

    def test_app_services_with_defaults(self):
        """Test creating AppServices with only required fields."""
        router = RouteExecutor()

        services = AppServices(router=router)

        assert services.router is router
        assert services.rate_limiter is None
        assert services.db_logger is None
        assert services.routing_manager is None

    def test_app_services_type_annotations(self):
        """Test that AppServices has proper type annotations."""
        annotations = AppServices.__annotations__

        assert "router" in annotations
        assert "rate_limiter" in annotations
        assert "db_logger" in annotations
        assert "routing_manager" in annotations


class TestDependencyFunctions:
    """Test dependency injection functions."""

    def test_get_services(self, app_services):
        """Test get_services returns services from app.state."""
        app = FastAPI()
        app.state.services = app_services

        request = Mock(spec=Request)
        request.app = app

        result = get_services(request)

        assert result is app_services

    def test_get_router(self, app_services):
        """Test get_router extracts router from services."""

        # Create a mock that simulates the Depends behavior
        def mock_get_services():
            return app_services

        result = get_router(mock_get_services())

        assert result is app_services.router

    def test_get_rate_limiter(self, app_services):
        """Test get_rate_limiter extracts rate limiter from services."""

        def mock_get_services():
            return app_services

        result = get_rate_limiter(mock_get_services())

        assert result is app_services.rate_limiter

    def test_get_db_logger(self, app_services):
        """Test get_db_logger extracts database logger from services."""

        def mock_get_services():
            return app_services

        result = get_db_logger(mock_get_services())

        assert result is app_services.db_logger

    def test_get_rate_limiter_when_none(self):
        """Test get_rate_limiter returns None when not configured."""
        services = AppServices(router=RouteExecutor())

        def mock_get_services():
            return services

        result = get_rate_limiter(mock_get_services())

        assert result is None

    def test_get_db_logger_when_none(self):
        """Test get_db_logger returns None when not configured."""
        services = AppServices(router=RouteExecutor())

        def mock_get_services():
            return services

        result = get_db_logger(mock_get_services())

        assert result is None


class TestDependencyIntegration:
    """Test dependency injection in FastAPI context."""

    def test_dependencies_in_fastapi_route(self, app_services):
        """Test using dependencies in actual FastAPI routes."""
        from fastapi import Depends

        app = FastAPI()
        app.state.services = app_services

        @app.get("/test-router")
        def test_router_endpoint(router: RouteExecutor = Depends(get_router)):
            return {"has_router": router is not None}

        @app.get("/test-limiter")
        def test_limiter_endpoint(limiter=Depends(get_rate_limiter)):
            return {"has_limiter": limiter is not None}

        @app.get("/test-logger")
        def test_logger_endpoint(logger=Depends(get_db_logger)):
            return {"has_logger": logger is not None}

        client = TestClient(app)

        # Test router dependency
        response = client.get("/test-router")
        assert response.status_code == 200
        assert response.json()["has_router"] is True

        # Test rate limiter dependency
        response = client.get("/test-limiter")
        assert response.status_code == 200
        assert response.json()["has_limiter"] is True

        # Test db logger dependency
        response = client.get("/test-logger")
        assert response.status_code == 200
        assert response.json()["has_logger"] is True

    def test_dependencies_with_missing_state(self):
        """Test that dependencies fail gracefully when app.state is not set."""
        app = FastAPI()
        # Don't set app.state.services

        request = Mock(spec=Request)
        request.app = app

        with pytest.raises(AttributeError):
            get_services(request)

    def test_nested_dependency_chain(self, app_services):
        """Test that nested dependencies work correctly."""
        from fastapi import Depends

        app = FastAPI()
        app.state.services = app_services

        # Create a route that uses nested dependencies
        @app.get("/test-nested")
        def test_nested_endpoint(
            services: AppServices = Depends(get_services),
            router: RouteExecutor = Depends(get_router),
        ):
            return {
                "services_router_match": services.router is router,
                "router_routes": len(router.routes),
            }

        client = TestClient(app)

        response = client.get("/test-nested")
        assert response.status_code == 200
        data = response.json()
        assert data["services_router_match"] is True
        assert data["router_routes"] >= 0


class TestDependencyEdgeCases:
    """Test edge cases and error conditions."""

    def test_get_services_with_none_state(self):
        """Test get_services when app.state exists but services is None."""
        app = FastAPI()
        app.state.services = None

        request = Mock(spec=Request)
        request.app = app

        result = get_services(request)

        assert result is None

    def test_dependency_with_async_route(self, app_services):
        """Test dependencies work with async routes."""
        from fastapi import Depends

        app = FastAPI()
        app.state.services = app_services

        @app.get("/test-async")
        async def test_async_endpoint(router: RouteExecutor = Depends(get_router)):
            return {"is_router": isinstance(router, RouteExecutor)}

        client = TestClient(app)

        response = client.get("/test-async")
        assert response.status_code == 200
        assert response.json()["is_router"] is True

    def test_multiple_requests_share_services(self, app_services):
        """Test that multiple requests share the same services instance."""
        from fastapi import Depends

        app = FastAPI()
        app.state.services = app_services

        service_ids = []

        @app.get("/test-shared")
        def test_shared_endpoint(services: AppServices = Depends(get_services)):
            service_ids.append(id(services))
            return {"service_id": id(services)}

        client = TestClient(app)

        # Make multiple requests
        response1 = client.get("/test-shared")
        response2 = client.get("/test-shared")

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Services should be the same instance
        assert service_ids[0] == service_ids[1]
