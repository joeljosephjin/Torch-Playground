.PHONY: help install install-dev test lint format type-check clean train-quick train-demo

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and dependencies
	pip install -e .
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -e .[dev]
	pip install -r requirements.txt
	pre-commit install

test:  ## Run tests with coverage
	pytest --cov=. --cov-report=term-missing --cov-report=html

test-quick:  ## Run tests without coverage
	pytest -v

lint:  ## Run all linting checks
	flake8 .
	black --check .
	isort --check-only .

format:  ## Format code with black and isort
	black .
	isort .

type-check:  ## Run type checking with mypy
	mypy . --ignore-missing-imports

clean:  ## Clean up temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

train-quick:  ## Quick training test (1 epoch, small batch)
	python main.py --epochs 1 --batch-size 4 --model SimpleModel --dataset cifar_10 --learning-rate 0.01

train-demo:  ## Demo training (5 epochs, medium batch)
	python main.py --epochs 5 --batch-size 16 --model AVModel --dataset cifar_10 --learning-rate 0.01 --log-interval 1

train-densenet:  ## Train DenseNet model
	python main.py --epochs 50 --batch-size 64 --learning-rate 0.1 --model DenseNet3 --dataset cifar_10 --use-wandb

check-all: lint type-check test  ## Run all checks (lint, type-check, test)

setup-dev: install-dev  ## Setup development environment
	@echo "Development environment setup complete!"
	@echo "Run 'make help' to see available commands."

# CI/CD commands
ci-test:  ## Run CI tests (used by GitHub Actions)
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	black --check .
	isort --check-only .
	pytest --cov=. --cov-report=xml