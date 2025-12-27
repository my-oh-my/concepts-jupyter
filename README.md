# Market Analysis Concepts (Jupyter)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional environment for exploring and developing market analysis concepts using Jupyter Notebooks and structured Python code.

## 🚀 Overview

This repository provides a robust framework for financial data analysis. It combines the interactivity of Jupyter Notebooks with the maintainability of a proper Python package.

### Key Features
- **Modular Architecture**: Core logic is encapsulated in a `src/` package for reuse across notebooks and tests.
- **Automated Data I/O**: Integrated loaders for Stooq and other financial data providers.
- **Statistical Utilities**: Built-in tools for calculating Rate of Return (RoR), identifying extremes, and correlation analysis.
- **Clean Version Control**: Uses **Jupytext** to synchronize `.ipynb` notebooks with `.py` scripts, ensuring readable git diffs.
- **Quality Assurance**: Integrated with Pylint, MyPy, and Pre-commit hooks to maintain high standards.

## 📁 Project Structure

```text
.
├── notebooks/        # Jupyter notebooks and their synced .py counterparts
├── src/              # Shared Python source code
│   ├── analysis/     # Statistical and technical analysis modules
│   ├── dataio/       # Data loading and processing utilities
│   └── config.py     # Centralized configuration
├── data/             # Local data storage (ignored by git)
│   └── raw/          # Original, immutable data files
├── tests/            # Test suite (placeholder)
├── Makefile          # Common development tasks
├── pyproject.toml    # Project metadata and dependencies
└── .pre-commit-config.yaml # Automated code quality checks
```

## 🛠️ Getting Started

### 1. Prerequisite
Ensure you have Python 3.10+ installed.

### 2. Setup
Clone the repository and run the automated setup:

```bash
make setup
```

This will:
- Create a virtual environment (`.venv`).
- Install production and development dependencies.
- Install the project in editable mode (`pip install -e .`).
- Set up pre-commit hooks.

### 3. Environment Variables
Copy the example environment file and adjust if needed:

```bash
cp .env.example .env
```

## 💻 Development Workflow

The project uses a `Makefile` to simplify common tasks:

| Command | Description |
|---------|-------------|
| `make env` | Create a fresh virtual environment |
| `make setup` | Full project setup (env + deps + pre-commit) |
| `make install` | Install only production dependencies |
| `make format` | Auto-format code with Black and Isort |
| `make lint` | Run Pylint and MyPy checks |
| `make test` | Run tests with coverage reports |
| `make sync` | Manually sync notebooks with scripts via Jupytext |
| `make notebook` | Launch Jupyter Lab |
| `make clean` | Remove temporary files and caches |

### Jupyter & Jupytext
We use **Jupytext** to pair `.ipynb` files with `.py` scripts.
- **Always commit the `.py` version** of the notebook.
- If you edit the `.py` file externally, run `make sync` to update the `.ipynb`.
- This ensures your logic is always versioned in a human-readable format.

## 📊 Shared Code
When a utility or analysis function becomes stable, move it from the notebook to the `src/` directory. This makes it available for other notebooks and automated tests.

---
*Created for specialized market analysis and algorithmic exploration.*
