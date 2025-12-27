.PHONY: help setup env lint test format sync notebook clean

PYTHON = python3
PIP = $(PYTHON) -m pip
VENV = .venv

# Default command: display help
help:
	@echo "Available commands:"
	@echo "  make install   - Install only production dependencies"
	@echo "  make setup     - Install production and dev dependencies + pre-commit"
	@echo "  make env       - Create virtual environment and install dependencies"
	@echo "  make lint      - Run linters (pylint, mypy) on src and notebooks"
	@echo "  make test      - Run tests with coverage"
	@echo "  make format    - Auto-format code (black, isort)"
	@echo "  make sync      - Manually sync notebooks with scripts via Jupytext"
	@echo "  make notebook  - Launch Jupyter Lab server"
	@echo "  make clean     - Remove all temporary files and checkpoints"

# Check if running inside a virtual environment
check-venv:
ifndef VIRTUAL_ENV
	$(error VIRTUAL_ENV is not set. Please activate your virtual environment (e.g., source $(VENV)/bin/activate))
endif

install: check-venv
	@echo "Installing production dependencies..."
	$(PIP) install -r requirements.txt

setup: install
	@echo "Installing development dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	pre-commit install

env:
	@echo "Creating virtual environment in $(VENV)..."
	$(PYTHON) -m venv $(VENV)
	@echo "To activate: source $(VENV)/bin/activate && make setup"

lint:
	@echo "Running linters..."
	pylint src/
	mypy src/
	nbqa pylint notebooks/
	nbqa mypy notebooks/

test:
	@echo "Running tests..."
	pytest --cov=src tests/

format:
	@echo "Formatting code..."
	black .
	isort .
	nbqa black notebooks/
	nbqa isort notebooks/

sync:
	@echo "Syncing notebooks via Jupytext..."
	jupytext --sync notebooks/*.ipynb

notebook:
	@echo "Launching Jupyter Lab..."
	jupyter lab

clean:
	@echo "Cleaning up..."
	rm -rf .pytest_cache .mypy_cache .coverage .pytest_cache .ipynb_checkpoints
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
