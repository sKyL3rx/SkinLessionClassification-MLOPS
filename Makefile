.PHONY: setup prepare split train eval export benchmark serve mlflow-ui test lint format dvc-repro clean 

VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
CONFIG ?= params.yaml
EXP_NAME ?= default_run

setup:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

prepare:
	$(PYTHON) scripts/prepare_ham10000.py --config $(CONFIG)

split:
	$(PYTHON) scripts/make_grouped_splits.py --config $(CONFIG)

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

eval:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

export:
	$(PYTHON) scripts/export_onnx.py --config $(CONFIG)

benchmark:
	$(PYTHON) scripts/benchmark_inference.py --onnx-path deployment/onnx/model.onnx

archive-run:
	$(PYTHON) scripts/archive_run.py --experiment-name $(EXP_NAME) --dvc-add

test:
	$(PYTHON) -m pytest


dvc-repro:
	$(PYTHON) -m dvc repro

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check . --fix


lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
	rm -rf build dist *.egg-info src/*.egg-info
	rm -rf htmlcov .coverage coverage.xml


