.PHONY: setup lint test preprocess splits splits-preprocess smoke-train train train-release infer validate-submission full-pipeline weak-labels-v1 weak-images-v1 adversarial-validation format pre-commit-install mlflow-ui

UV ?= uv
CONFIG ?= configs/model/image_baseline_v1.yaml
RELEASE_CONFIG ?= configs/release/rc1.yaml
RELEASE_TRAIN_CONFIG ?= configs/release_training/train_cv03_balanced_sampler.yaml
SUBMISSION ?= releases/rc1/submission.csv
PREPROCESS_DIR ?= data/preprocess
FULL_CONFIG ?= configs/model/cv03_focal_loss_preprocess.yaml
FULL_RELEASE_CONFIG ?= configs/release/full_pipeline.yaml
FULL_SUBMISSION ?= releases/full_pipeline/submission.csv
FOLD ?= 0

setup:
	$(UV) sync --extra dev

lint:
	$(UV) run ruff check .

test:
	$(UV) run pytest

preprocess:
	$(UV) run python scripts/data01_preprocess.py --raw-dir data/raw --out-dir $(PREPROCESS_DIR) --report-md reports/preprocess_leakage_report.md

splits:
	$(UV) run python scripts/data02_build_splits.py

splits-preprocess:
	$(UV) run python scripts/data02_build_splits.py --train-csv $(PREPROCESS_DIR)/train_df.csv --val-csv $(PREPROCESS_DIR)/val_df.csv --manifest $(PREPROCESS_DIR)/data_manifest.parquet --output-json $(PREPROCESS_DIR)/splits_v1.json --report-md reports/leakage_report_preprocess.md

manifest:
	$(UV) run python src/datasets/make_manifest.py


smoke-train:
	$(UV) run python src/training/train_image.py --config $(CONFIG) --fold $(FOLD) --debug

train:
	$(UV) run python src/training/train_image.py --config $(CONFIG) --fold $(FOLD)

train-release:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(UV) run python src/training/train_release.py --config $(RELEASE_TRAIN_CONFIG)

infer:
	$(UV) run python -m src.inference.predict --config $(RELEASE_CONFIG)

validate-submission:
	$(UV) run python -m src.inference.validate_submission --submission $(SUBMISSION) --test-csv data/raw/test_df.csv --class-mapping configs/data/class_mapping.yaml

full-pipeline: preprocess splits-preprocess
	$(UV) run python src/training/train_image.py --config $(FULL_CONFIG) --all-folds
	$(UV) run python -m src.inference.predict --config $(FULL_RELEASE_CONFIG)
	$(UV) run python -m src.inference.validate_submission --submission $(FULL_SUBMISSION) --test-csv $(PREPROCESS_DIR)/test_df.csv --class-mapping configs/data/class_mapping.yaml

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