# Project Context

## Цель проекта

Проект решает задачу классификации изображений помещений из объявлений недвижимости. На входе одно изображение, на выходе один из 20 классов типа комнаты/сцены.

Целевая схема классов зафиксирована в `configs/data/class_mapping.yaml`: валидные class id — `0..19`. Финальная отправка должна иметь ровно две колонки:

```csv
image_id_ext,Predicted
```

## Данные

Основные входы:

- `data/raw/train_df.csv` — размеченный development pool.
- `data/raw/val_df.csv` — отдельный shadow holdout, не используется для обучения.
- `data/raw/test_df.csv` — test set для submission.
- `data/raw/*_images/*_images/` — изображения.
- `data/processed/data_manifest.parquet` — manifest с путями, статусами и content hash.
- `data/splits/splits_v1.json` — фиксированный split-контракт для обучения и оценки.

Текущий split:

- train/development pool после фильтров: `4562` строки.
- shadow holdout после фильтров: `477` строк.
- folds: `5`, stratified/group-safe по `item_id` и duplicate/content-hash компонентам.
- train vs shadow по `content_hash`: пересечение `0`.

Важное ограничение: `val_df` является shadow holdout. Его нельзя добавлять в train folds, подбирать на нём гиперпараметры или выбирать postprocess. Shadow — только контрольная проверка.

## Метрика и оценка

Главная метрика — Macro F1 по классам `0..19`.

Стандартная оценка серьёзного image-run:

1. OOF на всех development folds.
2. Shadow holdout inference каждым fold-чекпоинтом и ensemble по вероятностям.
3. Per-class F1 и confusion matrix.
4. Export prediction parquet с `target`, `pred`, `prob_*` и, для fold/OOF outputs, `logit_*`.

Shadow holdout не содержит class `18`, поэтому:

- OOF остаётся главным all-class gate.
- Shadow Macro F1 нужно трактовать осторожно.
- Для shadow полезны две метрики: all-label macro и present-label macro.

## Текущие модели и кандидаты

Основной baseline/backbone family:

- `convnext_tiny.in12k_ft_in1k`, image size `224`.

Whitelist backbone:

- `convnext_tiny.in12k_ft_in1k`
- `efficientnet_b0`
- `resnet50`

Текущий RC1:

- candidate: `cv03_balanced_sampler`.
- config: `configs/model/cv03_balanced_sampler.yaml`.
- release config: `configs/release/rc1.yaml`.
- OOF Macro F1: `0.630100`.
- Shadow present-class Macro F1: `0.747438`.

Второй модельный кандидат для diversity:

- `model2_v1` на `efficientnet_b0`.
- config: `configs/model/model2_v1.yaml`.
- artifacts: `artifacts/oof/model2_v1/`.
- report: `reports/model2_eval_v1.md`.
- назначение: дать ошибки другой семьи для возможного ensemble/diagnostics.

## Известные слабые места

Редкие/сложные классы:

- class `5`: кабинет.
- class `11`: гардеробная / кладовая / постирочная.
- class `18`: не могу дать ответ / не ясно.
- также часто путаются универсальная комната / гостиная / спальня / предметы интерьера.

Data quality:

- есть missing/url filename mismatch в raw image folders;
- corrupted images сейчас не зафиксированы;
- usable rows фильтруются через manifest/splits.

Weak labels подготовлены, но не включены в RC1 по умолчанию.

## Основные команды

Установка и проверки:

```bash
make setup
make lint
make test
make smoke-train
```

Построить splits:

```bash
make splits
```

Обучить один fold:

```bash
uv run python src/training/train_image.py --config configs/model/image_baseline_v1.yaml --fold 0
```

Обучить full CV:

```bash
uv run python src/training/train_image.py --config configs/model/image_baseline_v1.yaml --all-folds
```

Сгенерировать RC1 submission:

```bash
make infer
make validate-submission
```

MLflow UI:

```bash
make mlflow-ui
```
