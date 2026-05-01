# Architecture

## Обзор

Проект построен как reproducible ML pipeline вокруг фиксированных YAML-конфигов, CLI entrypoints и файловых артефактов.

Высокоуровневый поток:

```text
raw csv/images
  -> manifest/data integrity
  -> leakage-safe splits
  -> train image experiments
  -> OOF/shadow reports
  -> experiment registry / diagnostics
  -> release config
  -> inference
  -> validated submission
```

## Структура репозитория

```text
configs/      YAML-конфиги данных, моделей, postprocess и release
src/          основной Python-код
scripts/      тонкие CLI-обёртки и report builders
reports/      markdown/csv отчёты
artifacts/    checkpoints, OOF predictions, MLflow logs
data/         raw/interim/processed data
releases/     submission/prediction outputs
notebooks/    исследовательские ноутбуки
tests/        unit/regression tests
```

## Data layer

### Manifest

Entry point:

```bash
uv run python src/datasets/make_manifest.py
```

Ответственность:

- проверить наличие изображений;
- проверить открываемость PIL;
- посчитать `content_hash`;
- сохранить `data/processed/data_manifest.parquet`;
- записать data quality отчёты в `reports/`.

### Split builder

Entry point:

```bash
uv run python scripts/data02_build_splits.py
```

Core module: `src/datasets/data02_build_splits.py`.

Контракт:

- вход: `train_df.csv`, `val_df.csv`, manifest;
- выход: `data/splits/splits_v1.json`;
- `train_df` превращается в 5 development folds;
- `val_df` фиксируется как отдельный shadow holdout;
- rows с невалидным manifest status не попадают в train/shadow;
- leakage checks выполняются по `item_id`, `image_id_ext`, `image`, `content_hash`.

### Label quality / ratio audit

Entry point:

```bash
uv run python scripts/data03_audit_label_quality_ratio.py
```

Core module: `src/datasets/data03_audit_label_quality_ratio.py`.

Ответственность:

- анализ `ratio` как proxy label consensus;
- disputed/low-consensus flags;
- рекомендации по sample weighting;
- отчёт `reports/label_quality_ratio.md` и таблицы в `tables/`.

### Weak labels

Entry point:

```bash
make weak-labels-v1
```

Core module: `src/datasets/weak_labels_v1.py`.

Weak labels проходят дедупликацию и audit, но не включены в RC1 по умолчанию.

### Weak images v1

Entry point:

```bash
make weak-images-v1
```

Core module: `src/datasets/weak_images_v1.py`.

Pipeline читает heuristic CSV и legacy image folder, применяет hard gates (`max_texts`, `person_found`, `is_catalog`, min size), удаляет leakage по `image_id_ext`/`sha256`, дедуплицирует weak-кандидаты, выбирает top-N по class quotas и копирует curated subset в `data/raw/weak_images/weak_images/`.

Выходы:

- `data/processed/weak_downloaded_v1.csv` — manifest для train-only injection;
- `reports/weak_images_download_report.md` — counts по drop reasons и selected rows.

Training включает эти rows только при `experiment.weak_label_flag: true`; valid/shadow/OOF остаются на original split rows.

## Training layer

Entry point:

```bash
uv run python src/training/train_image.py --config <config.yaml> --fold 0
uv run python src/training/train_image.py --config <config.yaml> --all-folds
```

Основные модули:

- `src/training/config_loader.py` — OmegaConf-compatible загрузка YAML, нормализация artifact roots, базовая валидация.
- `src/training/train_image.py` — dataset, transforms, model creation, train loop, evaluation, MLflow logging, OOF/shadow export.

### Training config contract

Ключевые секции model config:

- `data`: paths, `splits_json`, column names, `num_classes`, image size.
- `model`: `backbone`, `pretrained`, `whitelist`.
- `train`: epochs, batch size, workers, LR, AMP, seed.
- `augmentation`: torchvision/albumentations policy and label smoothing.
- `experiment`: loss, sampler, ratio policy, feature flags, version.
- `artifacts`: OOF dir, report path, roots.
- `mlflow`: tracking URI, experiment name, artifact root.

### Model creation

Models are created through `timm.create_model` with `num_classes=20`.
Backbone must be in config whitelist if whitelist is present.

Current supported families:

- ConvNeXt tiny: `convnext_tiny.in12k_ft_in1k`.
- EfficientNet-B0: `efficientnet_b0`.
- ResNet-50: `resnet50`.

### Data loading

`RoomDataset` reads rows from split records and loads images by `image_id_ext` from the configured train or shadow image directory.

Transforms:

- train: resize/crop/flip/color jitter or safer augmentation policy;
- val/shadow: resize + center crop + ImageNet normalization;
- optional Albumentations pipeline exists for experiment configs.

### Losses and imbalance controls

Supported controls in `train_image.py` include:

- `loss: ce`;
- `loss: weighted_ce`;
- `loss: ratio_ce` with sample weights from `ratio`;
- class weight policies such as raw inverse, sqrt inverse, effective number;
- samplers: shuffle, balanced, class-aware mixture, repeat-factor.

All serious changes must preserve `splits_v1` and must not train on shadow holdout.

### Outputs

For every run/fold:

- checkpoint: `artifacts/checkpoints/roomclf_<backbone>_fold<fold>_<image_size>_<version>.ckpt`;
- fold OOF parquet: `artifacts/oof/<run>/fold<fold>_oof_predictions.parquet`;
- fold shadow parquet: `artifacts/oof/<run>/fold<fold>_shadow_predictions.parquet`;
- MLflow params/metrics/artifacts.

For `--all-folds`:

- `artifacts/oof/<run>/oof_predictions.parquet`;
- `artifacts/oof/<run>/shadow_holdout_predictions.parquet`;
- `artifacts/oof/<run>/confusion_matrix_oof.csv`;
- copied config: `artifacts/oof/<run>/config.yaml`;
- report: configured `reports/*.md`.

Standard prediction parquet columns:

- identifiers: `image_id_ext`, `item_id`, `source_dataset`, `local_path`, `content_hash` when available;
- labels: `target`, `pred`;
- scores: `prob_0..prob_19`;
- logits: `logit_0..logit_19` for fold OOF/fold shadow/OOF exports.

## Experiment diagnostics

Scripts:

- `scripts/make_experiment_registry.py` — aggregates OOF/shadow metrics into `artifacts/experiment_registry.csv`.
- `scripts/make_rare_class_board.py` — rare/ambiguous class board.
- `scripts/make_error_taxonomy.py` — confusion/error taxonomy.
- `scripts/optimize_class_bias.py` — optional class-bias tuning on OOF only.

Reports in `reports/` are the main human-readable audit trail.

## Inference layer

Entry point:

```bash
uv run python -m src.inference.predict --config configs/release/rc1.yaml
```

Core module: `src/inference/predict.py`.

Flow:

1. Load release YAML.
2. Load test CSV and test images.
3. Load one or more fold checkpoints.
4. Average logits across checkpoints.
5. Optionally apply TTA and class bias.
6. Softmax to probabilities.
7. Write:
   - `releases/<name>/submission.csv`;
   - `releases/<name>/predictions.parquet` with logits/probs.
8. Optionally call `validate_submission`.

Submission validator: `src/inference/validate_submission.py`.

It rejects:

- missing/extra/duplicate `image_id_ext`;
- wrong row count;
- non-integer `Predicted`;
- classes outside `0..19`.

## External services

MLflow is local-only by default:

```yaml
mlflow:
  tracking_uri: sqlite:///artifacts/logs/mlflow.db
  experiment_name: room_type_image_baseline
  artifact_root: artifacts/logs/mlruns
```

Details: `docs/mlflow.md`.
