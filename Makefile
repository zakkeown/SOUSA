.PHONY: help install install-dev install-all lint format test test-fast test-cov clean pre-commit setup-hooks

# Default target
help:
	@echo "SOUSA Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install base package"
	@echo "  make install-dev    Install with dev dependencies"
	@echo "  make install-all    Install with all optional dependencies"
	@echo "  make setup-hooks    Install pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  make lint           Run ruff linter"
	@echo "  make format         Format code with black"
	@echo "  make pre-commit     Run all pre-commit hooks"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-fast      Run core tests only (no audio)"
	@echo "  make test-cov       Run tests with coverage report"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove build artifacts and caches"

# Setup
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	@echo ""
	@echo ">>> Run 'make setup-hooks' to install pre-commit hooks"

install-all:
	pip install -e ".[dev,hub,cloud,ml]"

setup-hooks:
	pip install pre-commit
	pre-commit install
	@echo "Pre-commit hooks installed!"

# Development
lint:
	ruff check dataset_gen/ tests/ scripts/

format:
	black dataset_gen/ tests/ scripts/
	ruff check --fix dataset_gen/ tests/ scripts/

pre-commit:
	pre-commit run --all-files

# Testing
test:
	pytest -v

test-fast:
	pytest tests/test_rudiments.py tests/test_profiles.py tests/test_labels.py -v

test-cov:
	pytest --cov=dataset_gen --cov-report=html --cov-report=term-missing
	@echo ""
	@echo "Coverage report: htmlcov/index.html"

# Cleanup
clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .ruff_cache/ htmlcov/
	rm -rf .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
