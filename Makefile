.PHONY: setup lint test smoke-train train infer validate-submission weak-labels-v1 weak-images-v1 adversarial-validation format pre-commit-install mlflow-ui

UV ?= uv
CONFIG ?= configs/model/image_baseline_v1.yaml
RELEASE_CONFIG ?= configs/release/rc1.yaml
SUBMISSION ?= releases/rc1/submission.csv
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

infer:
	$(UV) run python -m src.inference.predict --config $(RELEASE_CONFIG)

validate-submission:
	$(UV) run python -m src.inference.validate_submission --submission $(SUBMISSION) --test-csv data/raw/test_df.csv --class-mapping configs/data/class_mapping.yaml

weak-labels-v1:
	$(UV) run python scripts/build_weak_labels_v1.py

weak-images-v1:
	$(UV) run python scripts/build_weak_images_v1.py --max-added-per-class 5=180 6=80 11=200 --weak-weight 0.35 --max-texts 0 --drop-catalog --drop-person

adversarial-validation:
	$(UV) run python scripts/run_adversarial_validation.py

format:
	$(UV) run ruff format .
	$(UV) run ruff check . --fix

pre-commit-install:
	$(UV) run pre-commit install

mlflow-ui:
	$(UV) run mlflow ui --backend-store-uri sqlite:///artifacts/logs/mlflow.db


seconf_model:
	$(UV) un python src/training/train_image.py --config configs/model/model2_v1.yaml --all-folds