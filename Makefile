.PHONY: help setup setup-cuda lint test format pre-commit-install \
	check-raw-data download-data manifest splits label-audit data preprocess prepare-data \
	weak-labels-v1 weak-images-v1 prepare-weak-data \
	smoke-train train train-all train-large pipeline full-pipeline \
	infer validate-submission run demo mlflow-ui

.NOTPARALLEL: pipeline full-pipeline run

UV ?= uv
DATA_DIR ?= data/raw
TRAIN_CSV ?= $(DATA_DIR)/train_df.csv
VAL_CSV ?= $(DATA_DIR)/val_df.csv
TEST_CSV ?= $(DATA_DIR)/test_df.csv
MANIFEST ?= data/processed/data_manifest.parquet
SPLITS_JSON ?= data/splits/splits_v1.json
LEAKAGE_REPORT ?= reports/leakage_report.md
LABEL_AUDIT_REPORT ?= reports/label_quality_ratio.md
RATIO_BINS ?= tables/ratio_bins.csv
LOW_CONSENSUS ?= tables/low_consensus_samples.csv

CONFIG ?= configs/model/image_baseline_v1.yaml
LARGE_CONFIG ?= configs/model/cv04_convnext_small_earlystop_gpu.yaml
RELEASE_CONFIG ?= configs/release/rc1.yaml
SUBMISSION ?= releases/rc1/submission.csv
CLASS_MAPPING ?= configs/data/class_mapping.yaml
FOLD ?= 0
N_FOLDS ?= 5

WEAK_MAX_ADDED_PER_CLASS ?= 5=180 6=80 11=200
WEAK_WEIGHT ?= 0.35

help:
	@echo "Основной пайплайн:"
	@echo "  make prepare-data   # check raw csv -> download/check images -> manifest -> splits -> label audit"
	@echo "  make smoke-train    # prepare-data + debug training на одном fold"
	@echo "  make train          # prepare-data + обучение одного fold, FOLD=0 CONFIG=..."
	@echo "  make train-all      # prepare-data + обучение всех folds"
	@echo "  make pipeline       # prepare-data -> train-all -> infer -> validate-submission"
	@echo ""
	@echo "Отдельные шаги:"
	@echo "  make download-data  # распаковать архивы/скачать недостающие image из csv и собрать manifest"
	@echo "  make splits         # пересобрать leakage-safe splits"
	@echo "  make weak-images-v1 # подготовить weak images"
	@echo "  make infer          # собрать submission по RELEASE_CONFIG"
	@echo "  make run            # infer + validate-submission"
	@echo "  make demo           # запустить Gradio demo"

setup:
	$(UV) sync --extra dev

setup-cuda:
	$(UV) sync --extra dev --extra cuda

lint:
	$(UV) run ruff check .

test:
	$(UV) run pytest

format:
	$(UV) run ruff format .
	$(UV) run ruff check . --fix

pre-commit-install:
	$(UV) run pre-commit install

check-raw-data:
	@test -f "$(TRAIN_CSV)" || (echo "Missing $(TRAIN_CSV)" >&2; exit 1)
	@test -f "$(VAL_CSV)" || (echo "Missing $(VAL_CSV)" >&2; exit 1)
	@test -f "$(TEST_CSV)" || (echo "Missing $(TEST_CSV)" >&2; exit 1)

# Builds $(MANIFEST), reports/data_integrity.md, reports/missing_files.csv.
# Also unpacks split archives and downloads missing images from csv URLs when possible.
download-data: check-raw-data
	$(UV) run python src/datasets/make_manifest.py --base-dir $(DATA_DIR)

manifest: download-data

splits: manifest
	$(UV) run python scripts/data02_build_splits.py \
		--train-csv $(TRAIN_CSV) \
		--val-csv $(VAL_CSV) \
		--manifest $(MANIFEST) \
		--output-json $(SPLITS_JSON) \
		--report-md $(LEAKAGE_REPORT) \
		--n-folds $(N_FOLDS)

label-audit: splits
	$(UV) run python scripts/data03_audit_label_quality_ratio.py \
		--train-csv $(TRAIN_CSV) \
		--val-csv $(VAL_CSV) \
		--splits-json $(SPLITS_JSON) \
		--report-md $(LABEL_AUDIT_REPORT) \
		--ratio-bins-csv $(RATIO_BINS) \
		--low-consensus-csv $(LOW_CONSENSUS)

prepare-data: splits label-audit

data: prepare-data

preprocess: prepare-data

weak-labels-v1: manifest
	$(UV) run python scripts/build_weak_labels_v1.py

weak-images-v1: splits
	$(UV) run python scripts/build_weak_images_v1.py \
		--max-added-per-class $(WEAK_MAX_ADDED_PER_CLASS) \
		--weak-weight $(WEAK_WEIGHT) \
		--max-texts 0 \
		--drop-catalog \
		--drop-person

prepare-weak-data: weak-labels-v1 weak-images-v1

smoke-train: prepare-data
	$(UV) run python src/training/train_image.py --config $(CONFIG) --fold $(FOLD) --debug

train: prepare-data
	$(UV) run python src/training/train_image.py --config $(CONFIG) --fold $(FOLD)

train-all: prepare-data
	$(UV) run python src/training/train_image.py --config $(CONFIG) --all-folds

train-large: prepare-data
	$(UV) run python src/training/train_image.py --config $(LARGE_CONFIG) --all-folds

pipeline: prepare-data train-all infer validate-submission

full-pipeline: pipeline

infer:
	$(UV) run python -m src.inference.predict --config $(RELEASE_CONFIG)

validate-submission:
	$(UV) run python -m src.inference.validate_submission \
		--submission $(SUBMISSION) \
		--test-csv $(TEST_CSV) \
		--class-mapping $(CLASS_MAPPING)

run: infer validate-submission

demo:
	$(UV) run python demo/app.py --config $(RELEASE_CONFIG)

mlflow-ui:
	$(UV) run mlflow ui --backend-store-uri sqlite:///artifacts/logs/mlflow.db
