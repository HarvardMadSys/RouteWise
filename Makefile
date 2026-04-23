.PHONY: help format lint test test-verbose test-cov setup-dev clean check all

# Default target
.DEFAULT_GOAL := help

# Allow overriding uv run flags, e.g.:
#   make lint UV_RUN="uv run --active"
UV_RUN ?= uv run

# Colors for terminal output
RESET := \033[0m
BOLD := \033[1m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m

help:  ## Show this help message
	@echo "$(BOLD)Available targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'

format:  ## Format code with ruff (configured for Google style)
	@echo "$(YELLOW)Running formatter...$(RESET)"
	$(UV_RUN) ruff format .
	$(UV_RUN) ruff check --fix .
	@echo "$(GREEN)✓ Code formatted$(RESET)"

lint:  ## Run linters (ruff, pydocstyle)
	@echo "$(YELLOW)Running linters...$(RESET)"
	$(UV_RUN) ruff check --no-fix .
	$(UV_RUN) pydocstyle
	@echo "$(GREEN)✓ Linting passed$(RESET)"

test:  ## Run unit/integration tests (exclude external)
	@echo "$(YELLOW)Running tests (not external)...$(RESET)"
	$(UV_RUN) pytest -q -m "not external"
	@echo "$(GREEN)✓ Tests passed$(RESET)"

test-verbose: ## Run tests with verbose output (exclude external)
	$(UV_RUN) pytest -vv -m "not external"

test-cov:  ## Run tests with coverage (exclude external)
	$(UV_RUN) pytest -m "not external" --cov=. --cov-report=term-missing --cov-report=html

test-e2e: ## Run external/E2E tests (may require local server)
	$(UV_RUN) pytest -m external -vv

check: lint test  ## Run all checks (lint, test)
	@echo "$(GREEN)✓ All checks passed$(RESET)"

all: format check  ## Format code and run all checks

setup-dev:  ## Set up development environment
	@echo "$(YELLOW)Setting up development environment...$(RESET)"
	@# Create venv if it doesn't exist; keep idempotent
	[ -d .venv ] || uv venv -p 3.10
	@# Install package in editable mode
	uv pip install -e .
	@# Install requirements.txt if it exists
	[ -f requirements.txt ] && uv pip install -r requirements.txt || true
	@# Sync development dependencies from pyproject.toml
	uv sync --group dev
	$(UV_RUN) pre-commit install
	@echo "$(GREEN)✓ Development environment ready$(RESET)"

clean:  ## Clean build artifacts and cache
	@echo "$(YELLOW)Cleaning up...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	@echo "$(GREEN)✓ Cleanup complete$(RESET)"
