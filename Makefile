.PHONY: \
	setup verify lint format test test-fast \
	dvc-pull dvc-push pipeline stage \
	prepare split train evaluate export \
	evaluate-onnx calibrate triage error-analysis \
	bundle benchmark check-bundle \
	serve-api serve-demo \
	docker-build docker-up docker-down \
	clean all

UV ?= uv
RUN := $(UV) run --locked
CONFIG ?= params.yaml
STAGE ?=
MODEL_BUNDLE_DIR ?= artifacts/deployment/model
ONNX_PROVIDER ?= cuda

all: verify pipeline check-bundle

setup:
	$(UV) sync --locked --all-extras

verify: lint test-fast

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	
format:
	$(RUN) ruff check . --fix
	$(RUN) ruff format .

test:
	$(RUN) pytest

test-fast:
	$(RUN) pytest \
		tests \
		-m "not artifact and not data and not slow" \
		-q

dvc-pull:
	$(RUN) dvc pull

dvc-push:
	$(RUN) dvc push

pipeline:
	$(RUN) dvc repro

stage:
	test -n "$(STAGE)"
	$(RUN) dvc repro "$(STAGE)"

prepare:
	$(RUN) dvc repro prepare_data

split:
	$(RUN) dvc repro make_splits

train:
	$(RUN) dvc repro train

evaluate:
	$(RUN) dvc repro \
		evaluate_val \
		evaluate_test

export:
	$(RUN) dvc repro export_onnx

evaluate-onnx:
	$(RUN) dvc repro \
		evaluate_onnx_val \
		evaluate_onnx_test

calibrate:
	$(RUN) dvc repro calibrate

triage:
	$(RUN) dvc repro \
		triage_val \
		triage_test

error-analysis:
	$(RUN) dvc repro \
		error_analysis_test

bundle:
	$(RUN) dvc repro \
		build_deployment_bundle

benchmark:
	$(RUN) dvc repro \
		benchmark_onnx

check-bundle:
	test -f $(MODEL_BUNDLE_DIR)/model.onnx
	test -f $(MODEL_BUNDLE_DIR)/model.metadata.json
	test -f $(MODEL_BUNDLE_DIR)/temperature.json
	test -f $(MODEL_BUNDLE_DIR)/decision.json
	test -f $(MODEL_BUNDLE_DIR)/manifest.json

serve-api: check-bundle
	MODEL_BUNDLE_DIR=$(MODEL_BUNDLE_DIR) \
	ONNX_PROVIDER=$(ONNX_PROVIDER) \
	$(RUN) uvicorn \
		lesion_ml.api.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload

serve-demo:
	$(RUN) python \
		-m lesion_ml.demo.gradio_app

docker-build: check-bundle
	docker compose \
		-f deployment/compose.yaml \
		build \
		--no-cache \
		api

docker-up: check-bundle
	docker compose \
		-f deployment/compose.yaml \
		up api

docker-down:
	docker compose \
		-f deployment/compose.yaml \
		down

clean:
	find . -type d \
		-name "__pycache__" \
		-prune \
		-exec rm -rf {} +

	find . -type f \
		\( -name "*.pyc" -o -name "*.pyo" \) \
		-delete

	rm -rf \
		.pytest_cache \
		.ruff_cache \
		.mypy_cache \
		htmlcov \
		build \
		dist

	rm -rf \
		*.egg-info \
		src/*.egg-info

	rm -f \
		.coverage \
		coverage.xml