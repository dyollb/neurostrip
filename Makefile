.PHONY: help install install-dev format lint test clean pre-commit-install pre-commit-run setup-dev
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	.venv/bin/pip install -e .

install-dev: ## Install the package with development dependencies
	.venv/bin/pip install -e ".[dev]"

format: ## Format code with ruff
	.venv/bin/ruff format src tests

lint: ## Run ruff linter
	.venv/bin/ruff check src tests

fix: ## Run ruff linter with auto-fix
	.venv/bin/ruff check --fix src tests

type-check: ## Run mypy type checker
	.venv/bin/mypy src

security: ## Run bandit security scanner
	.venv/bin/bandit -r src

test: ## Run tests with pytest
	.venv/bin/python -m pytest tests/ -v

test-cov: ## Run tests with coverage
	.venv/bin/python -m pytest tests/ -v --cov=neurostrip --cov-report=html --cov-report=term

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

pre-commit-install: ## Install pre-commit hooks
	.venv/bin/pre-commit install

pre-commit-run: ## Run pre-commit on all files
	.venv/bin/pre-commit run --all-files

check: ## Run all checks (format, lint, type-check, test)
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test

setup-dev: ## Set up development environment
	$(MAKE) install-dev
	$(MAKE) pre-commit-install
	@echo ""
	@echo "🎉 Development environment setup complete!"
	@echo ""
	@echo "Available commands:"
	@echo "  make format     - Format code with ruff"
	@echo "  make lint       - Run ruff linter"
	@echo "  make fix        - Run ruff linter with auto-fix"
	@echo "  make type-check - Run mypy type checker"
	@echo "  make test       - Run tests"
	@echo "  make check      - Run all checks"
	@echo ""
	@echo "Pre-commit hooks are installed and will run automatically on commit."
