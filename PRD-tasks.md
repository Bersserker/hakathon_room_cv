# PRD-tasks.md — декомпозиция задач с DOR/DOD

**Проект:** Room Type Image Classification  
**Цель backlog:** довести текущую research-реализацию до релизного решения с валидным submission, устойчивой работой с дисбалансом классов, интерпретируемостью и защитными материалами.

---

## 1. Приоритеты

| Приоритет | Смысл |
|---|---|
| P0 | Без этого нельзя выпускать релиз или сдавать финальный submission |
| P1 | Даёт сильный вклад в качество/устойчивость/защиту |
| P2 | Делать после закрытия критического пути |
| P3 | Nice-to-have |

---

## 2. P0 backlog — критический путь

### TASK-001 — Подтвердить схему классов и диапазон `Predicted`

**Priority:** P0  
**Epic:** Data / Submission correctness  
**Проблема:** в описании задачи есть нестыковка `0..18` vs `0..19`; в текущих артефактах проекта фактически 20 классов `0..19`.

**Что сделать:**
- проверить `train_df.result.unique()`, `val_df.result.unique()`;
- проверить `sample_submission.csv`;
- сверить label mapping из `splits_v1.json`;
- зафиксировать `class_mapping.yaml`;
- описать решение в `reports/class_schema_audit.md`.

**DOR:**
- доступны `train_df.csv`, `val_df.csv`, `sample_submission.csv` или их schema/records;
- есть текущий `splits_v1.json`;
- определён владелец финального submission.

**DOD:**
- создан `configs/data/class_mapping.yaml`;
- создан `reports/class_schema_audit.md`;
- validator знает допустимый диапазон классов;
- все эксперименты используют единый `num_classes`;
- решение по классу `19` явно описано в README/final report.

**Артефакты:**
- `configs/data/class_mapping.yaml`;
- `reports/class_schema_audit.md`.

**Acceptance metric:**
- `validate_submission.py` отклоняет классы вне подтверждённого диапазона.

---

### TASK-002 — Реализовать submission validator

**Priority:** P0  
**Epic:** Inference / Release  
**Проблема:** нет проверки финального CSV.

**Что сделать:**
- создать `src/inference/validate_submission.py`;
- проверять колонки `image_id_ext,Predicted`;
- проверять row count;
- проверять все test IDs;
- проверять отсутствие дублей;
- проверять integer classes;
- проверять диапазон классов;
- проверять отсутствие NaN;
- вывести checksum.

**DOR:**
- TASK-001 закрыт;
- известен путь к `test_df.csv`;
- известен путь к `sample_submission.csv`.

**DOD:**
- валидный CSV проходит;
- CSV с пропущенным id падает;
- CSV с дублем падает;
- CSV с invalid class падает;
- добавлены unit tests.

**Артефакты:**
- `src/inference/validate_submission.py`;
- `tests/test_validate_submission.py`.

**Команда проверки:**
```bash
python -m src.inference.validate_submission \
  --submission releases/rc1/submission.csv \
  --test-csv data/raw/test_df.csv \
  --class-mapping configs/data/class_mapping.yaml
```

---

### TASK-003 — Реализовать deterministic inference pipeline

**Priority:** P0  
**Epic:** Inference / Release  
**Проблема:** `src/inference/` пустой, нет генерации submission.

**Что сделать:**
- создать `src/inference/predict.py`;
- загрузка config + checkpoint;
- создание модели через тот же `timm` backbone;
- preprocessing строго как validation transform;
- batch inference;
- optional TTA flag;
- optional ensemble over folds;
- output probabilities/logits optional;
- генерация `submission.csv`;
- повторяемый seed и deterministic flags.

**DOR:**
- есть checkpoint хотя бы одного fold или список fold checkpoints;
- есть class mapping;
- есть test image directory;
- TASK-002 закрыт.

**DOD:**
- команда inference создаёт CSV;
- CSV проходит validator;
- повторный запуск даёт тот же checksum;
- inference не требует notebook;
- README обновлён.

**Артефакты:**
- `src/inference/predict.py`;
- `configs/release/rc1.yaml`;
- `releases/rc1/submission.csv`;
- `releases/rc1/predictions.parquet` или `.csv`.

**Команда проверки:**
```bash
python -m src.inference.predict --config configs/release/rc1.yaml
python -m src.inference.validate_submission --submission releases/rc1/submission.csv
```

---

### TASK-004 — Собрать release bundle `releases/rc1`

**Priority:** P0  
**Epic:** Release engineering  
**Проблема:** нет финального набора артефактов, пригодного для сдачи.

**Что сделать:**
- создать директорию `releases/rc1/`;
- положить config, checkpoints, class mapping, submission, metrics report;
- добавить checksums;
- добавить model card;
- добавить license notes;
- добавить run commands.

**DOR:**
- TASK-001/002/003 закрыты;
- выбран candidate model;
- есть финальные веса.

**DOD:**
- `releases/rc1/README.md` описывает запуск;
- `sha256sums.txt` создан;
- submission validator проходит;
- model card содержит backbone, data split, metrics, limitations;
- bundle можно передать другому участнику без контекста.

**Артефакты:**
- `releases/rc1/README.md`;
- `releases/rc1/model_card.md`;
- `releases/rc1/sha256sums.txt`;
- `releases/rc1/submission.csv`.

---

### TASK-005 — Исправить propagation `ratio` в training pipeline

**Priority:** P0  
**Epic:** Label quality / Training  
**Проблема:** `ratio_ce` сейчас фактически совпадает с CE, потому что `ratio` не попадает в records внутри `splits_v1.json`.

**Что сделать:**
- добавить `ratio` в `REQUIRED_CSV_COLUMNS` или отдельную optional колонку;
- добавить `ratio` в `rows_to_records()`;
- проверить, что `RoomDataset` получает `ratio`;
- добавить test, который гарантирует наличие `ratio` в fold records;
- перегенерировать `splits_v1.json`;
- повторить `cv03_baseline_ce` и `cv03_ratio_weighting`.

**DOR:**
- доступны исходные `train_df.csv`/`val_df.csv` с колонкой `ratio`;
- текущий `splits_v1.json` зафиксирован;
- есть baseline CE metrics.

**DOD:**
- `ratio` есть в train/valid/shadow records;
- `cv03_ratio_weighting` даёт отличающиеся от CE predictions;
- `reports/ratio_weighting_ablation_v1.md` содержит class-wise deltas;
- нет leakage: `ratio` используется только как sample weight, не как inference feature.

**Артефакты:**
- обновлённый `data/splits/splits_v1.json` или `splits_v2.json`;
- `reports/ratio_weighting_ablation_v1.md`;
- `tests/test_splits_include_ratio.py`.

---

### TASK-006 — Пересчитать честный baseline после фикса `ratio`

**Priority:** P0  
**Epic:** Experiment control  
**Проблема:** после TASK-005 старые эксперименты нельзя напрямую считать финальными.

**Что сделать:**
- переобучить CE baseline;
- переобучить ratio CE;
- сохранить OOF/shadow predictions;
- сравнить class-wise;
- проверить, что split version фиксирован.

**DOR:**
- TASK-005 закрыт;
- MLflow работает;
- конфиги экспериментов обновлены.

**DOD:**
- `reports/baseline_ce_v2.md`;
- `reports/ratio_weighting_v2.md`;
- OOF full coverage;
- shadow metrics;
- таблица delta к текущему baseline.

**Acceptance metric:**
- OOF coverage = 100%;
- Macro F1 и per-class F1 есть для всех классов.

---

### TASK-007 — Исправить shadow-holdout reporting

**Priority:** P0  
**Epic:** Evaluation  
**Проблема:** shadow holdout не содержит класс `18`, поэтому all-label shadow Macro F1 не полностью отражает задачу.

**Что сделать:**
- добавить в отчёты `shadow_macro_f1_all_labels`;
- добавить `shadow_macro_f1_present_labels`;
- явно выводить support по классам на shadow;
- добавить warning, если support=0;
- сделать `dev_holdout_v2` из train pool с покрытием всех классов или использовать OOF как основной RC gate.

**DOR:**
- есть текущий `splits_v1.json`;
- есть OOF/shadow predictions.

**DOD:**
- отчёт показывает support=0 для класса `18`;
- RC decision не опирается только на shadow all-label macro;
- final report корректно объясняет ограничение shadow holdout.

**Артефакты:**
- `reports/evaluation_protocol_v2.md`;
- обновлённый report generator.

---

### TASK-008 — Ввести единый experiment registry и RC decision report

**Priority:** P0  
**Epic:** Experiment management  
**Проблема:** есть отдельные отчёты, но нет единой таблицы решений.

**Что сделать:**
- создать `artifacts/experiment_registry.csv`;
- для каждого run фиксировать:
  - config;
  - split version;
  - data version;
  - checkpoint;
  - OOF Macro F1;
  - shadow Macro F1;
  - class 5/11/18 F1;
  - notes;
  - decision.

**DOR:**
- есть минимум 3 эксперимента;
- OOF reports доступны.

**DOD:**
- registry обновляется вручную или скриптом;
- `reports/rc1_decision.md` объясняет выбор финального кандидата;
- rejected experiments имеют причину отказа.

**Артефакты:**
- `artifacts/experiment_registry.csv`;
- `reports/rc1_decision.md`.

---

## 3. P0 backlog — дисбаланс классов

### TASK-009 — Сделать rare-class board

**Priority:** P0  
**Epic:** Imbalance  
**Проблема:** Macro F1 чувствителен к редким классам, но нет отдельной панели контроля rare classes.

**Что сделать:**
- построить таблицу по каждому классу:
  - train support;
  - shadow support;
  - OOF precision;
  - OOF recall;
  - OOF F1;
  - predicted support;
  - delta к baseline;
- отдельно выделить классы `5`, `11`, `18`, `2`, `3`, `17`.

**DOR:**
- есть OOF predictions;
- есть class mapping.

**DOD:**
- `reports/rare_class_board_v1.md` создан;
- board автоматически строится из OOF;
- board используется в RC decision.

**Артефакты:**
- `reports/rare_class_board_v1.md`;
- `scripts/make_rare_class_board.py`.

---

### TASK-010 — Реализовать capped/effective-number class weights

**Priority:** P0  
**Epic:** Imbalance / Loss  
**Проблема:** raw weighted CE ухудшил OOF Macro F1 до 0.556744.

**Что сделать:**
- добавить config options:
  - `class_weight_policy: none | sqrt_inv | effective_num`;
  - `weight_clip_min`;
  - `weight_clip_max`;
  - `effective_beta`;
- нормализовать weights mean=1;
- логировать веса классов в report;
- прогнать ablation:
  - sqrt inverse, clip `[0.5,2.5]`;
  - effective beta `0.99`, `0.995`, `0.999`, clip `[0.5,3.0]`.

**DOR:**
- baseline CE v2 готов;
- rare-class board готов.

**DOD:**
- `reports/class_weight_ablation_v1.md`;
- минимум 3 политики weights сравнены;
- есть решение include/reject;
- не допускается падение OOF Macro F1 ниже CE baseline более чем на 0.3 pp без rare-class uplift.

**Артефакты:**
- обновлённый `src/training/train_image.py`;
- `configs/model/class_weights_*.yaml`;
- `reports/class_weight_ablation_v1.md`.

---

### TASK-011 — Реализовать мягкий class-aware sampler v2

**Priority:** P0  
**Epic:** Imbalance / Sampling  
**Проблема:** full balanced sampler улучшил Macro F1, но ухудшил ряд frequent/ambiguous классов.

**Что сделать:**
- добавить sampler policy:
  - `shuffle`;
  - `balanced`;
  - `class_aware_mixture`;
  - `repeat_factor`;
- реализовать:
  ```text
  q(c) = (1 - lambda) * p_emp(c) + lambda * 1/C
  ```
- ablation lambda: `0.3`, `0.5`, `0.7`;
- добавить repeat-factor cap.

**DOR:**
- TASK-009 закрыт;
- baseline CE v2 готов;
- текущий `cv03_balanced_sampler` принят как reference.

**DOD:**
- `reports/sampler_ablation_v2.md`;
- есть comparison: CE vs full balanced vs class-aware;
- выбран sampler для RC или явно отвергнут;
- class `5` F1 растёт без сильной деградации `2/3/17`.

**Acceptance metric:**
- OOF Macro F1 ≥ CE baseline;
- F1 class 5 ≥ 0.35;
- F1 class 11 ≥ 0.50;
- падение F1 классов `2`, `3`, `17` ≤ 2 pp к CE baseline.

---

### TASK-012 — Добавить class-bias / logit adjustment по OOF

**Priority:** P0  
**Epic:** Imbalance / Postprocessing  
**Проблема:** для Macro F1 может быть выгодно скорректировать underpredicted classes без переобучения модели.

**Что сделать:**
- реализовать `scripts/optimize_class_bias.py`;
- вход: OOF logits/probs + targets;
- output: `bias_c` или `tau`;
- objective: maximize OOF Macro F1;
- regularization on bias magnitude;
- сохранить bias в release config;
- не использовать shadow для оптимизации.

**DOR:**
- есть OOF logits/probs;
- есть class mapping;
- есть baseline candidate.

**DOD:**
- `reports/class_bias_tuning_v1.md`;
- uplift на OOF показан;
- shadow present-class macro проверен;
- bias зафиксирован в config.

**Артефакты:**
- `scripts/optimize_class_bias.py`;
- `configs/postprocess/class_bias_rc1.yaml`;
- `reports/class_bias_tuning_v1.md`.

---

### TASK-013 — Сформировать финальную imbalance recipe

**Priority:** P0  
**Epic:** Imbalance / Decision  
**Проблема:** нужны не отдельные эксперименты, а выбранная стратегия.

**Что сделать:**
- сравнить:
  - CE;
  - ratio CE;
  - capped weights;
  - sampler v2;
  - sampler v2 + ratio;
  - sampler v2 + bias;
- выбрать финальный набор;
- запретить слишком тяжёлые комбинации, если они нестабильны.

**DOR:**
- TASK-010/011/012 готовы;
- rare-class board есть.

**DOD:**
- `reports/imbalance_strategy_v1.md`;
- выбранный config создан;
- decision содержит:
  - что включили;
  - что отвергли;
  - почему;
  - какие классы выиграли/проиграли;
  - риски.

**Артефакты:**
- `reports/imbalance_strategy_v1.md`;
- `configs/model/imbalance_rc1.yaml`.

---

## 4. P1 backlog — качество модели

### TASK-014 — Подключить weak labels для rare-class uplift

**Priority:** P1  
**Epic:** Weak labels / Rare classes  
**Проблема:** weak labels подготовлены, но не используются; при этом они могут помочь классам `5/6/11`.

**Что сделать:**
- построить manifest для weak images;
- проверить доступность изображений;
- посчитать hashes;
- удалить overlap с train/val/test;
- сделать ручной QA sample;
- добавить weak dataset в train only;
- добавить `weak_weight`;
- добавить per-epoch quota.

**DOR:**
- `weak_labels_v1.parquet` готов;
- baseline v2 стабилен;
- есть возможность загрузить/кэшировать heuristic images.

**DOD:**
- `reports/weak_label_manifest_v1.md`;
- `reports/weak_label_uplift_v1.md`;
- показаны delta по `5/6/11`;
- weak labels не используются в validation/test;
- решение include/reject принято.

**Acceptance metric:**
- class 5 F1 +5 pp к выбранному baseline или class 11 F1 +3 pp;
- общий Macro F1 не падает более чем на 0.5 pp.

---

### TASK-015 — Two-stage training: weak pretrain → clean fine-tune

**Priority:** P1  
**Epic:** Training strategy  
**Проблема:** прямое смешивание weak labels может зашумить модель.

**Что сделать:**
- stage A: 1–2 эпохи на weak+real с low weak weight;
- stage B: fine-tune на real labels only;
- сравнить с direct mixing;
- использовать меньший LR на backbone.

**DOR:**
- TASK-014 закрыт;
- trainer умеет weak labels.

**DOD:**
- `reports/two_stage_training_v1.md`;
- сравнение direct mixing vs two-stage;
- выбран режим.

---

### TASK-016 — Провести image size ablation

**Priority:** P1  
**Epic:** Model quality  
**Что сделать:**
- прогнать `224`, `288`, `320`;
- контролировать batch size/accumulation;
- сравнить качество и inference cost.

**DOR:**
- baseline v2 готов;
- GPU budget понятен.

**DOD:**
- `reports/image_size_ablation_v1.md`;
- выбран image size для RC;
- зафиксирован inference cost.

---

### TASK-017 — Проверить второй backbone

**Priority:** P1  
**Epic:** Model diversity  
**Что сделать:**
- EfficientNetV2-S или EfficientNet-B3;
- Swin/ViT-Small при доступном GPU;
- тот же split, те же метрики;
- сравнить diversity ошибок.

**DOR:**
- baseline pipeline стабилен;
- registry готов.

**DOD:**
- `reports/backbone_comparison_v1.md`;
- минимум один второй backbone обучен;
- есть решение: single model или ensemble candidate.

---

### TASK-018 — Реализовать safe augmentations ablation

**Priority:** P1  
**Epic:** Augmentation  
**Что сделать:**
- сравнить baseline torchvision, safe torchvision, albumentations;
- проверить label smoothing;
- запретить аугментации, меняющие смысл класса;
- визуально проверить sample grid.

**DOR:**
- baseline v2 готов;
- configs уже есть.

**DOD:**
- `reports/augmentation_ablation_v1.md`;
- sample grid сохранён;
- выбран augmentation policy.

---

### TASK-019 — Реализовать small ensemble / fold ensemble

**Priority:** P1  
**Epic:** Ensemble  
**Что сделать:**
- average probabilities across 5 folds;
- optional weighted average best two backbones;
- optional TTA;
- проверить latency;
- сохранить ensemble spec.

**DOR:**
- есть минимум 2 кандидата или 5 fold checkpoints;
- inference pipeline готов.

**DOD:**
- `reports/ensemble_eval_v1.md`;
- `configs/release/ensemble_rc1.yaml`;
- uplift ≥ 0.5 pp или ensemble отвергнут.

---

## 5. P1 backlog — интерпретируемость и защита

### TASK-020 — Сделать class-wise error taxonomy

**Priority:** P1  
**Epic:** Evaluation / Explainability  
**Что сделать:**
- взять top confusion pairs;
- собрать sample grids;
- классифицировать причины ошибок:
  - шум разметки;
  - визуальная близость;
  - нехватка контекста;
  - плохой crop;
  - exterior/interior ambiguity;
  - некомнатное изображение.

**DOR:**
- есть OOF predictions;
- есть локальные изображения.

**DOD:**
- `reports/error_taxonomy_v1.md`;
- минимум 10 confusion pairs разобраны;
- есть рекомендации “лечим данными / лечим моделью / не лечим в рамках срока”.

---

### TASK-021 — Добавить Grad-CAM examples

**Priority:** P1  
**Epic:** Interpretability  
**Что сделать:**
- реализовать Grad-CAM для выбранного backbone;
- примеры correct/high confidence;
- примеры wrong/high confidence;
- примеры rare classes;
- сохранить картинки в `reports/figures/gradcam/`.

**DOR:**
- выбран RC checkpoint;
- есть локальные изображения;
- известен target layer для backbone.

**DOD:**
- `reports/gradcam_examples_v1.md`;
- минимум 20 визуализаций;
- Grad-CAM используется в demo или final report.

---

### TASK-022 — Подготовить финальный report

**Priority:** P1  
**Epic:** Documentation  
**Что сделать:**
- описать задачу;
- target/class schema;
- данные;
- split/leakage;
- модель;
- эксперименты;
- дисбаланс;
- ошибки;
- inference;
- ограничения;
- вклад команды.

**DOR:**
- RC выбран;
- метрики финальные;
- release bundle готов.

**DOD:**
- `final_report.md` создан;
- есть ссылки на артефакты;
- report можно загрузить на платформу как итоговый текстовый документ.

---

### TASK-023 — Сделать Gradio demo

**Priority:** P1  
**Epic:** Demo  
**Что сделать:**
- создать `demo/app.py`;
- загрузка изображения;
- top-3 predictions;
- confidence bars;
- optional Grad-CAM;
- указание model version.

**DOR:**
- inference pipeline готов;
- RC checkpoint готов.

**DOD:**
- `python demo/app.py` запускает интерфейс;
- README содержит команду;
- есть скриншот/GIF для backup.

---

## 6. P2 backlog — мультимодальность и дополнительные признаки

### TASK-024 — Пересобрать feature coverage report только для inference-compatible features

**Priority:** P2  
**Epic:** Feature engineering  
**Проблема:** текущий `DATA_05_report.md` содержит признаки, недоступные на test/inference: `ratio`, `source_dataset`, `consensus_band`, `class_samples`.

**Что сделать:**
- разделить признаки на:
  - training-only audit;
  - inference-compatible;
  - forbidden/leakage;
- проверить покрытие train/val/test;
- проверить schema shift;
- удалить target-derived features.

**DOR:**
- доступны train/test metadata;
- известны Perform/smart-crop/detector columns.

**DOD:**
- `reports/feature_coverage_v2.md`;
- список allowed features;
- список forbidden features;
- решение по fusion принято.

---

### TASK-025 — Late fusion image logits + metadata

**Priority:** P2  
**Epic:** Multimodal / Fusion  
**Что сделать:**
- собрать OOF logits;
- собрать allowed metadata;
- обучить stacker на OOF:
  - Logistic Regression;
  - CatBoost, если разрешён;
- оценить на shadow;
- запретить обучение stacker на in-fold predictions.

**DOR:**
- TASK-024 закрыт;
- OOF logits готовы.

**DOD:**
- `reports/fusion_eval_v1.md`;
- uplift ≥ 0.5 pp или fusion отвергнут;
- stacker обучен без leakage.

---

### TASK-026 — Smart-crop dual-view branch

**Priority:** P2  
**Epic:** Model quality / Multiview  
**Что сделать:**
- full image branch;
- crop image branch;
- average logits или stacker;
- проверить, какие классы выигрывают/проигрывают.

**DOR:**
- smart-crop coordinates доступны для train/test;
- crop quality проверен.

**DOD:**
- `reports/smart_crop_dual_view_v1.md`;
- sample grid crops;
- decision include/reject.

---

## 7. P2/P3 backlog — platform hardening

### TASK-027 — Улучшить test coverage

**Priority:** P2  
**Epic:** Engineering quality  
**Что сделать:**
- tests для transforms config;
- tests для split records;
- tests для metrics report;
- tests для inference validator;
- tests для class weights/sampler.

**DOR:**
- P0 pipelines реализованы.

**DOD:**
- `make test` зелёный;
- покрыты P0 баги: class schema, ratio propagation, submission validator.

---

### TASK-028 — Добавить model/data card и license notes

**Priority:** P2  
**Epic:** Documentation / Compliance  
**Что сделать:**
- model card;
- data card;
- backbone license;
- open-source weights source;
- limitations.

**DOR:**
- выбран RC backbone.

**DOD:**
- `releases/rc1/model_card.md`;
- `releases/rc1/data_card.md`;
- `releases/rc1/LICENSES.md`.

---

### TASK-029 — Добавить DVC или lightweight data versioning notes

**Priority:** P3  
**Epic:** Reproducibility  
**Что сделать:**
- необязательно внедрять DVC;
- но описать data versioning policy;
- добавить hash manifest;
- добавить artifact registry.

**DOR:**
- release bundle готов.

**DOD:**
- `docs/data_versioning.md`;
- data hashes описаны.

---

## 8. Контрольные точки

| Checkpoint | Условие прохождения | Артефакты |
|---|---|---|
| CP1 | class schema + validator + inference skeleton | TASK-001/002/003 |
| CP2 | ratio bug fixed + baseline v2 пересчитан | TASK-005/006 |
| CP3 | imbalance strategy выбрана | TASK-009/010/011/012/013 |
| CP4 | RC собран и валидирован | TASK-004/008 |
| CP5 | demo/report/interpretability готовы | TASK-020/021/022/023 |
| CP6 | финальная сдача | `submission.csv`, `final_report.md`, demo, presentation assets |

---

## 9. Definition of Release Candidate

RC считается готовым, если:

1. Есть подтверждённая class schema.
2. Есть deterministic inference.
3. Submission проходит validator.
4. Есть OOF full coverage.
5. Есть per-class metrics.
6. Есть imbalance strategy report.
7. Нет классов с F1=0 на OOF.
8. Есть release bundle.
9. Есть README с командами запуска.
10. Есть final report и demo или backup demo.

---

## 10. Главный риск-лист

| Риск | Severity | Mitigation |
|---|---:|---|
| Неверный диапазон классов | Critical | TASK-001/002 |
| Нет inference/submission | Critical | TASK-003/004 |
| `ratio` не работает | High | TASK-005/006 |
| Shadow не покрывает class 18 | High | TASK-007 |
| Raw class weights портят качество | High | TASK-010 |
| Weak labels шумные и без hash-dedup | High | TASK-014 |
| Fusion использует недоступные на test признаки | High | TASK-024 |
| Нет интерпретации | Medium | TASK-020/021 |
| Нет demo | Medium | TASK-023 |
| Лучший run не воспроизводим | High | TASK-004/008 |
