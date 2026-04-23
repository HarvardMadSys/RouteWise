.PHONY: help test-all test-phase1 test-phase2 test-phase3 clean

# Load environment variables from .env file (if exists)
# This allows CHUTES_API_KEY and other vars to be loaded automatically
-include ../../.env
export

# Default target
help:
	@echo "E2E Test Makefile"
	@echo "================="
	@echo ""
	@echo "Automated Testing (Recommended):"
	@echo "  make test-phase1         - Test local models directly (no gateway)"
	@echo "  make test-phase2-auto    - Auto start gateway, test, then cleanup"
	@echo "  make test-phase3-auto    - Auto start gateway, test, then cleanup"
	@echo "  make test-all-auto       - Run all phases (fully automated)"
	@echo ""
	@echo "Manual Testing (requires gateway running in another terminal):"
	@echo "  make test-phase2         - Test gateway routing to local models"
	@echo "  make test-phase3         - Test hybrid routing (40% local + 60% Chutes)"
	@echo "  make test-all            - Run all phases (manual mode)"
	@echo ""
	@echo "Gateway Management (for manual testing):"
	@echo "  make start-gateway-phase2  - Start gateway in foreground (Phase 2)"
	@echo "  make start-gateway-phase3  - Start gateway in foreground (Phase 3)"
	@echo "  make stop-gateway          - Stop any running gateway"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean               - Clean up test artifacts"
	@echo ""
	@echo "Environment variables:"
	@echo "  CHUTES_API_KEY       - Auto-loaded from ../../.env (required for Phase 3)"
	@echo "  CHUTES_BASE_URL      - Auto-loaded from ../../.env (default: https://llm.chutes.ai)"
	@echo "  GATEWAY_PORT         - Gateway port (default: 10081)"
	@echo ""
	@echo "Examples:"
	@echo "  make test-phase3-auto              # One command, fully automated"
	@echo "  GATEWAY_PORT=8080 make test-all-auto  # Use custom port"
	@echo ""

# Configuration
GATEWAY_PORT ?= 10081
GATEWAY_URL = http://localhost:$(GATEWAY_PORT)
E2E_CONFIG = ../fixtures/e2e_config.yaml
PYTHON = uv run python
PYTEST = uv run pytest
GATEWAY_PID_FILE = /tmp/hybridInference_gateway_$(GATEWAY_PORT).pid
GATEWAY_LOG_FILE = /tmp/hybridInference_gateway_$(GATEWAY_PORT).log

# Phase 1: Direct local model testing
test-phase1:
	@echo "========================================="
	@echo "Phase 1: Direct Local Model Testing"
	@echo "========================================="
	@echo "Testing Llama 3.2 3B (port 8001) and Qwen3 Coder 30B (port 8003)"
	@echo ""
	E2E_CONFIG=$(E2E_CONFIG) \
	$(PYTEST) test_phase1_direct_local.py -v -s

# Phase 2: Gateway routing to local models (manual mode - requires gateway running)
test-phase2:
	@echo "========================================="
	@echo "Phase 2: Gateway Routing to Local Models"
	@echo "========================================="
	@echo "Starting gateway on port $(GATEWAY_PORT)..."
	@echo ""
	@# Check if gateway is already running
	@if lsof -Pi :$(GATEWAY_PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "✓ Gateway already running on port $(GATEWAY_PORT)"; \
	else \
		echo "✗ Gateway not running. Please start it with:"; \
		echo "  make start-gateway-phase2"; \
		exit 1; \
	fi
	@echo ""
	GATEWAY_BASE_URL=$(GATEWAY_URL) \
	$(PYTEST) test_phase2_gateway_routing.py -v -s

# Phase 2: Automated mode - auto start/stop gateway
test-phase2-auto:
	@echo "========================================="
	@echo "Phase 2: Gateway Routing (Automated)"
	@echo "========================================="
	@$(MAKE) --no-print-directory _start-gateway-phase2-bg
	@sleep 3  # Wait for gateway to be ready
	@echo ""
	@echo "Running Phase 2 tests..."
	@GATEWAY_BASE_URL=$(GATEWAY_URL) \
	$(PYTEST) test_phase2_gateway_routing.py -v -s; \
	TEST_EXIT_CODE=$$?; \
	$(MAKE) --no-print-directory _stop-gateway-bg; \
	exit $$TEST_EXIT_CODE

# Phase 3: Hybrid routing (manual mode - requires gateway running)
test-phase3:
	@echo "========================================="
	@echo "Phase 3: Hybrid Routing (40% Local + 60% Chutes)"
	@echo "========================================="
	@# Check Chutes API key
	@if [ -z "$(CHUTES_API_KEY)" ]; then \
		echo "✗ CHUTES_API_KEY not set"; \
		echo "  Please set it: export CHUTES_API_KEY=your-key"; \
		exit 1; \
	fi
	@# Check if gateway is running
	@if lsof -Pi :$(GATEWAY_PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "✓ Gateway already running on port $(GATEWAY_PORT)"; \
	else \
		echo "✗ Gateway not running. Please start it with:"; \
		echo "  make start-gateway-phase3"; \
		exit 1; \
	fi
	@echo ""
	GATEWAY_BASE_URL=$(GATEWAY_URL) \
	$(PYTEST) test_phase3_hybrid_routing.py -v -s

# Phase 3: Automated mode - auto start/stop gateway
test-phase3-auto:
	@echo "========================================="
	@echo "Phase 3: Hybrid Routing (Automated)"
	@echo "========================================="
	@if [ -z "$(CHUTES_API_KEY)" ]; then \
		echo "✗ CHUTES_API_KEY not set"; \
		echo "  Hint: It should be loaded from ../../.env"; \
		exit 1; \
	fi
	@$(MAKE) --no-print-directory _start-gateway-phase3-bg
	@sleep 3  # Wait for gateway to be ready
	@echo ""
	@echo "Running Phase 3 tests..."
	@GATEWAY_BASE_URL=$(GATEWAY_URL) \
	$(PYTEST) test_phase3_hybrid_routing.py -v -s; \
	TEST_EXIT_CODE=$$?; \
	$(MAKE) --no-print-directory _stop-gateway-bg; \
	exit $$TEST_EXIT_CODE

# Run all phases sequentially (automated mode)
test-all-auto: test-phase1 test-phase2-auto test-phase3-auto
	@echo ""
	@echo "========================================="
	@echo "✓ All E2E Tests Completed (Automated)"
	@echo "========================================="

# Run all phases sequentially (manual mode - requires gateway running)
test-all: test-phase1 test-phase2 test-phase3
	@echo ""
	@echo "========================================="
	@echo "✓ All E2E Tests Completed"
	@echo "========================================="

# Internal: Start gateway for Phase 2 in background
_start-gateway-phase2-bg:
	@echo "Starting gateway for Phase 2 in background..."
	@# Stop any existing gateway first
	@$(MAKE) --no-print-directory _stop-gateway-bg 2>/dev/null || true
	@cd ../.. && \
	MODELS_CONFIG=test/fixtures/e2e_config.yaml \
	DB_ENABLED=false \
	RATE_LIMIT_ENABLED=0 \
	$(PYTHON) -m uvicorn serving.servers.app:app \
		--port $(GATEWAY_PORT) \
		--host 0.0.0.0 \
		> $(GATEWAY_LOG_FILE) 2>&1 & \
	echo $$! > $(GATEWAY_PID_FILE)
	@echo "✓ Gateway started (PID: $$(cat $(GATEWAY_PID_FILE)))"
	@echo "  Log file: $(GATEWAY_LOG_FILE)"

# Internal: Start gateway for Phase 3 in background
_start-gateway-phase3-bg:
	@echo "Starting gateway for Phase 3 in background..."
	@# Stop any existing gateway first
	@$(MAKE) --no-print-directory _stop-gateway-bg 2>/dev/null || true
	@cd ../.. && \
	MODELS_CONFIG=test/fixtures/e2e_config.yaml \
	CHUTES_BASE_URL=$(CHUTES_BASE_URL) \
	CHUTES_API_KEY=$(CHUTES_API_KEY) \
	DB_ENABLED=false \
	RATE_LIMIT_ENABLED=0 \
	$(PYTHON) -m uvicorn serving.servers.app:app \
		--port $(GATEWAY_PORT) \
		--host 0.0.0.0 \
		> $(GATEWAY_LOG_FILE) 2>&1 & \
	echo $$! > $(GATEWAY_PID_FILE)
	@echo "✓ Gateway started (PID: $$(cat $(GATEWAY_PID_FILE)))"
	@echo "  Log file: $(GATEWAY_LOG_FILE)"

# Internal: Stop background gateway
_stop-gateway-bg:
	@if [ -f $(GATEWAY_PID_FILE) ]; then \
		PID=$$(cat $(GATEWAY_PID_FILE)); \
		echo "Stopping gateway (PID: $$PID)..."; \
		kill $$PID 2>/dev/null || true; \
		rm -f $(GATEWAY_PID_FILE); \
		echo "✓ Gateway stopped"; \
	else \
		pkill -f "uvicorn.*$(GATEWAY_PORT)" 2>/dev/null || true; \
	fi

# Helper: Start gateway for Phase 2 (foreground - for manual testing)
start-gateway-phase2:
	@echo "Starting gateway for Phase 2..."
	@cd ../.. && \
	MODELS_CONFIG=test/fixtures/e2e_config.yaml \
	DB_ENABLED=false \
	RATE_LIMIT_ENABLED=0 \
	$(PYTHON) -m uvicorn serving.servers.app:app \
		--port $(GATEWAY_PORT) \
		--host 0.0.0.0

# Helper: Start gateway for Phase 3 (foreground - for manual testing)
start-gateway-phase3:
	@if [ -z "$(CHUTES_API_KEY)" ]; then \
		echo "✗ CHUTES_API_KEY not set"; \
		exit 1; \
	fi
	@echo "Starting gateway for Phase 3..."
	@cd ../.. && \
	MODELS_CONFIG=test/fixtures/e2e_config.yaml \
	CHUTES_BASE_URL=$(CHUTES_BASE_URL) \
	CHUTES_API_KEY=$(CHUTES_API_KEY) \
	DB_ENABLED=false \
	RATE_LIMIT_ENABLED=0 \
	$(PYTHON) -m uvicorn serving.servers.app:app \
		--port $(GATEWAY_PORT) \
		--host 0.0.0.0

# Helper: Stop gateway
stop-gateway:
	@echo "Stopping gateway on port $(GATEWAY_PORT)..."
	@$(MAKE) --no-print-directory _stop-gateway-bg
	@pkill -f "uvicorn.*$(GATEWAY_PORT)" 2>/dev/null || echo "No additional gateway process found"

# Clean up
clean:
	@echo "Cleaning up test artifacts..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -f $(GATEWAY_PID_FILE) 2>/dev/null || true
	@rm -f $(GATEWAY_LOG_FILE) 2>/dev/null || true
	@echo "✓ Clean complete"
