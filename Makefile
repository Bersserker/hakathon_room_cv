.PHONY: setup lint test smoke-train train weak-labels-v1 format pre-commit-install mlflow-ui

UV ?= uv
CONFIG ?= configs/model/image_baseline_v1.yaml
FOLD ?= 0

setup:
	$(UV) sync --extra dev

lint:
	$(UV) run ruff check .

test:
	$(UV) run pytest

splits:
	$(UV) run python scripts/data02_build_splits.py

manifest:
	$(UV) run python src/datasets/make_manifest.py


smoke-train:
	$(UV) run python src/training/train_image.py --config $(CONFIG) --fold $(FOLD) --debug

train:
	$(UV) run python src/training/train_image.py --config $(CONFIG) --fold $(FOLD)

weak-labels-v1:
	$(UV) run python scripts/build_weak_labels_v1.py

format:
	$(UV) run ruff format .
	$(UV) run ruff check . --fix

pre-commit-install:
	$(UV) run pre-commit install

mlflow-ui:
	$(UV) run mlflow ui --backend-store-uri sqlite:///artifacts/logs/mlflow.db


seconf_model:
	$(UV) un python src/training/train_image.py --config configs/model/model2_v1.yaml --all-folds