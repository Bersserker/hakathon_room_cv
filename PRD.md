# PRD.md — доработки проекта классификации типов комнат

**Проект:** Room Type Image Classification  
**Артефакт:** анализ текущей реализации + продуктово-технические требования к улучшению решения  
**Дата анализа:** 2026-04-30  
**Основа анализа:** zip-репозиторий `hakathon_room_cv.zip`, отчёты `reports/*.md`, конфиги `configs/model/*.yaml`, код `src/`, критерии сдачи и ML Design checklist из приложенных материалов.

---

## 1. Executive summary

Текущая реализация уже имеет сильный исследовательский фундамент: есть воспроизводимая структура репозитория, Docker/Makefile, конфиги экспериментов, MLflow, data integrity report, leakage report, group split по `item_id`/`content_hash`, аудит `ratio`, weak-label audit и несколько ablation-экспериментов по дисбалансу классов.

Лучший текущий кандидат по доступным отчётам — `cv03_balanced_sampler`:

| Эксперимент | Loss | Sampler | OOF Macro F1 | OOF Accuracy | Shadow Macro F1 | Shadow Accuracy | Вывод |
|---|---:|---:|---:|---:|---:|---:|---|
| `cv03_baseline_ce` | CE | shuffle | 0.625963 | 0.648838 | 0.665635 | 0.727463 | сильный baseline |
| `cv03_weighted_ce` | weighted CE | shuffle | 0.556744 | 0.579790 | 0.675242 | 0.729560 | inverse weights слишком агрессивны для OOF |
| `cv03_balanced_sampler` | CE | balanced | **0.630100** | 0.644893 | **0.710066** | **0.746331** | текущий лучший кандидат |
| `cv03_ratio_weighting` | ratio CE | shuffle | 0.625963 | 0.648838 | 0.665635 | 0.727463 | фактически повторяет CE из-за бага с `ratio` |

**Главный вывод:** решение уже может претендовать на высокий балл по качеству модели на локальной валидации, так как OOF Macro F1 выше 0.60. Но как релизный проект оно ещё не закрывает критический путь: нет отдельного inference pipeline, submission validator, финальных checkpoint/release bundle, demo app, интерпретации, полноценного final report и подтверждённого решения проблемы с `ratio`.

---

## 2. Текущее состояние реализации

### 2.1. Что уже хорошо

#### Данные и leakage control

1. **Корректная идея split.**  
   `data02_build_splits.py` строит `StratifiedGroupKFold` с группировкой через connected components по `item_id` и `content_hash`. Это защищает от утечки похожих изображений одного объявления между train/validation.

2. **Есть отдельный shadow holdout.**  
   `val_df` зафиксирован как `separate_shadow_holdout`, что снижает риск переобучения на CV.

3. **Есть data integrity report.**  
   `reports/data_integrity.md` фиксирует:
   - всего изображений в manifest: 53 065;
   - OK: 52 830;
   - missing: 235;
   - corrupted: 0.

4. **Проверены пересечения train/val.**  
   `reports/leakage_report.md` показывает:
   - пересечение по `item_id`: 0;
   - пересечение по `image_id_ext`: 0;
   - пересечение по `image`: 0;
   - пересечение hash train vs shadow: 0;
   - hash overlap across folds: 0.

5. **Есть аудит шумности разметки.**  
   `reports/label_quality_ratio.md` показывает, что:
   - всего размеченных изображений: 5 062;
   - полное согласие `ratio=1.0`: 3 194 / 5 062, 63.1%;
   - спорных `ratio<1.0`: 1 868 / 5 062, 36.9%;
   - low-consensus `ratio<=2/3`: 1 862 / 5 062, 36.8%;
   - рекомендована мягкая политика `sample_weight = clip(ratio, 0.75, 1.0)`.

6. **Есть weak-label pipeline.**  
   `weak_labels_v1.py` и `reports/weak_labels_audit.md` готовят слабую разметку для классов:
   - 5 — кабинет;
   - 6 — детская;
   - 11 — гардеробная / кладовая / постирочная.

#### Моделирование

1. **Использован современный pretrained backbone.**  
   Основной backbone — `convnext_tiny.in12k_ft_in1k` через `timm`. Это адекватный выбор для быстрого и сильного image-only baseline.

2. **Есть несколько конфигов экспериментов.**  
   Реализованы отдельные YAML:
   - `cv03_baseline_ce.yaml`;
   - `cv03_weighted_ce.yaml`;
   - `cv03_balanced_sampler.yaml`;
   - `cv03_ratio_weighting.yaml`;
   - `image_safe_aug_v1.yaml`;
   - `image_albumentations_v1.yaml`.

3. **Есть OOF-артефакты.**  
   В `artifacts/oof/*` лежат OOF/shadow predictions и confusion matrix.

4. **Macro F1 считается корректно как основная метрика.**  
   Код использует `f1_score(..., average="macro", labels=list(range(num_classes)), zero_division=0)`.

5. **Есть MLflow-интеграция.**  
   Логируются параметры, метрики, теги и checkpoints.

#### Инженерия

1. **Структура репозитория хорошая.**  
   Есть `src/`, `scripts/`, `configs/`, `reports/`, `tests/`, `docs/`, `notebooks/`.

2. **Есть Makefile.**  
   Команды `setup`, `lint`, `test`, `smoke-train`, `train`, `splits`, `manifest`, `weak-labels-v1`, `mlflow-ui`.

3. **Есть Docker и GPU Docker.**  
   Присутствуют `Dockerfile`, `Dockerfile.gpu`, `docker-compose.yml`, `docker-compose.gpu.yml`.

4. **Есть базовые тесты.**  
   Проверяется config loader и weak-label logic.

---

## 3. Критические проблемы и риски

### P0-1. Несогласованность числа классов

В брифе пользователя перечислены классы `0..18`, но также упоминается диапазон `0..19`. В текущем репозитории фактический target — **20 классов `0..19`**:

| class_id | label в текущих артефактах |
|---:|---|
| 0 | кухня / столовая |
| 1 | кухня-гостиная |
| 2 | универсальная комната |
| 3 | гостиная |
| 4 | спальня |
| 5 | кабинет |
| 6 | детская |
| 7 | ванная комната |
| 8 | туалет |
| 9 | совмещенный санузел |
| 10 | коридор / прихожая |
| 11 | гардеробная / кладовая / постирочная |
| 12 | балкон / лоджия |
| 13 | вид из окна / с балкона |
| 14 | дом снаружи / двор |
| 15 | подъезд / лестничная площадка |
| 16 | другое |
| 17 | предметы интерьера / быт.техника |
| 18 | не могу дать ответ / не ясно |
| 19 | комната без мебели |

**Риск:** если официальный hidden test ждёт `0..18`, предсказания `19` будут некорректными. Если официальный sample submission допускает `0..19`, текущая схема верна.

**Требование:** сделать `class_schema_audit.md` и submission validator, который сверяет:
- `train_df.result.unique()`;
- `val_df.result.unique()`;
- `sample_submission.csv`;
- официальный формат `Predicted`;
- список допустимых классов.

---

### P0-2. Нет production/release inference pipeline

В `src/inference/` лежит только `.gitkeep`. Нет:
- `src/inference/predict.py`;
- `src/inference/validate_submission.py`;
- deterministic inference;
- TTA/ensemble inference;
- генерации `submission.csv`;
- проверки колонок `image_id_ext,Predicted`;
- release bundle с checkpoint/config/hash.

**Риск:** даже при хорошем OOF невозможно сдать корректный финальный CSV и показать стабильный локальный запуск.

**Требование:** реализовать отдельный inference pipeline, не завязанный на notebook.

---

### P0-3. `ratio_weighting` сейчас не работает как задумано

По отчётам `cv03_ratio_weighting` полностью совпадает с `cv03_baseline_ce`:

| Метрика | CE | ratio weighting |
|---|---:|---:|
| OOF Macro F1 | 0.625963 | 0.625963 |
| OOF Accuracy | 0.648838 | 0.648838 |
| Shadow Macro F1 | 0.665635 | 0.665635 |
| Shadow Accuracy | 0.727463 | 0.727463 |

Причина по коду: `build_fold_frames()` строит train/valid DataFrame из `splits_v1.json`, а `rows_to_records()` в `data02_build_splits.py` не сохраняет `ratio`. В `RoomDataset.__getitem__()` если `ratio` отсутствует, sample weight становится `1.0`. Поэтому `ratio_ce` превращается в обычный CE.

**Требование:** добавить `ratio` в split records или join по `image_id_ext`/`item_id` при построении loader. После фикса повторить ablation.

---

### P0-4. Shadow holdout не покрывает все классы

В `splits_v1.json` shadow holdout содержит 477 строк, но класс `18 — не могу дать ответ / не ясно` отсутствует в shadow support.

**Риск:** shadow Macro F1 по всем 20 классам не является полноценной независимой оценкой для задачи. Он всё ещё полезен как external check, но не должен быть единственным RC gate.

**Требование:** добавить `dev_holdout_v2` или использовать OOF как основную оценку по всем классам, а shadow holdout репортить двумя способами:
- macro по всем классам, включая отсутствующие;
- macro по классам, присутствующим в shadow `y_true`.

---

### P0-5. Нет demo/prototype

В зависимостях есть `gradio`, но нет `demo/app.py`, Streamlit/Gradio app или CLI demo.

**Риск:** по критериям сдачи прототип интерфейса опционален, но он усиливает защиту и демонстрирует практическую применимость.

**Требование:** минимальный Gradio demo:
- загрузка изображения;
- top-3 классов с вероятностями;
- визуальная подсказка Grad-CAM для выбранного класса;
- отображение версии модели/config/checkpoint.

---

### P0-6. Нет интерпретации модели

Нет Grad-CAM, attention maps, sample grids ошибок, class-wise error taxonomy.

**Риск:** критерий “обоснованность подхода и интерпретируемость” будет закрыт поверхностно.

**Требование:** сделать `reports/error_taxonomy_v1.md` и `reports/gradcam_examples_v1.md`.

---

### P1-1. Weak labels подготовлены, но не подключены в обучение

`weak_labels_v1.parquet` создан, но в training pipeline нет интеграции слабых меток.

**Риск:** редкие классы `5` и `11` остаются недоусиленными.

**Дополнительный риск:** в weak-label audit `missing_hash_rows=50340`, то есть hash-дедупликация слабых данных почти не работает, потому что manifest для heuristic images не построен. Overlap с train по hash сейчас фактически не проверяется.

**Требование:** построить weak-label image manifest, скачать/кэшировать weak images или использовать их с очень низким весом и ручным аудитом.

---

### P1-2. Weighted CE деградирует OOF

`cv03_weighted_ce` дал OOF Macro F1 `0.556744`, ниже CE baseline `0.625963`. Это означает, что простые inverse-frequency weights слишком агрессивны.

**Требование:** заменить raw inverse weights на:
- effective number of samples;
- capped weights;
- square-root inverse frequency;
- logit adjustment;
- class-aware sampler с контролируемым смешиванием.

---

### P1-3. Дополнительные признаки пока не production-compatible

`reports/DATA_05_report.md` использует признаки вроде `source_dataset`, `ratio`, `consensus_band`, `class_samples`. Эти признаки недоступны на test/inference и/или являются target-derived.

**Требование:** отделить:
- training-only признаки для аудита качества разметки;
- inference-compatible признаки, доступные на train/val/test.

Для финального fusion использовать только признаки, доступные при inference:
- image logits;
- Perform features, если они есть в train/val/test или их можно получить для train;
- smart-crop/text/person flags, если покрытие одинаковое и нет leakage;
- image metadata: width, height, aspect ratio, status.

---

## 4. Подробный разбор дисбаланса классов

### 4.1. Фактический support в train pool и shadow

| class_id | label | train support | shadow support | total labeled |
|---:|---|---:|---:|---:|
| 0 | кухня / столовая | 248 | 34 | 282 |
| 1 | кухня-гостиная | 199 | 15 | 214 |
| 2 | универсальная комната | 247 | 23 | 270 |
| 3 | гостиная | 249 | 30 | 279 |
| 4 | спальня | 251 | 24 | 275 |
| 5 | кабинет | **74** | 21 | 95 |
| 6 | детская | 215 | 31 | 246 |
| 7 | ванная комната | 255 | 23 | 278 |
| 8 | туалет | 255 | 28 | 283 |
| 9 | совмещенный санузел | 254 | 30 | 284 |
| 10 | коридор / прихожая | 250 | 28 | 278 |
| 11 | гардеробная / кладовая / постирочная | **59** | 23 | 82 |
| 12 | балкон / лоджия | 249 | 22 | 271 |
| 13 | вид из окна / с балкона | 231 | 21 | 252 |
| 14 | дом снаружи / двор | 249 | 21 | 270 |
| 15 | подъезд / лестничная площадка | 253 | 31 | 284 |
| 16 | другое | 270 | 28 | 298 |
| 17 | предметы интерьера / быт.техника | 250 | 23 | 273 |
| 18 | не могу дать ответ / не ясно | 251 | **0** | 251 |
| 19 | комната без мебели | 253 | 21 | 274 |

**Основные minority-классы:** `5` и `11`.  
**Вторичная зона риска:** классы с высокой шумностью разметки: `18`, `2`, `1`, `17`, `5`, `3`, `11`.

### 4.2. Что уже сработало

Balanced sampler улучшил OOF Macro F1:

| class_id | label | CE F1 | balanced sampler F1 | delta |
|---:|---|---:|---:|---:|
| 5 | кабинет | 0.2062 | **0.3584** | **+0.1522** |
| 11 | гардеробная / кладовая / постирочная | 0.5055 | 0.4964 | -0.0091 |
| 18 | не могу дать ответ / не ясно | 0.2375 | **0.2923** | **+0.0548** |
| 2 | универсальная комната | 0.3858 | 0.3263 | -0.0595 |
| 3 | гостиная | 0.5598 | 0.4913 | -0.0685 |
| 17 | предметы интерьера / быт.техника | 0.5741 | 0.5112 | -0.0629 |

**Вывод:** balanced sampler полезен для класса `5` и немного для `18`, но ухудшает несколько визуально похожих/шумных классов. Его нельзя считать финальным решением без более мягкой версии.

---

## 5. Рекомендуемое решение по дисбалансу классов

### 5.1. Принцип

Не включать одновременно “всё тяжёлое”: aggressive oversampling + raw class weights + focal loss + weak labels. Это почти гарантированно даст нестабильность, перекомпенсацию rare-классов и падение frequent/ambiguous-классов.

Правильная последовательность:

1. **CE baseline**.
2. **Мягкий class-aware sampler** вместо full balanced sampler.
3. **Capped class weights** или **effective-number weights**.
4. **Фикс `ratio` и sample weighting**.
5. **Weak labels для `5/6/11` с малым весом и квотами**.
6. **Class bias / logit adjustment по OOF**.
7. **Focal loss** только если предыдущие методы не дают uplift.

---

### 5.2. Loss design

#### Базовая формула loss

```text
loss_i = CE(logits_i, y_i)
         * class_weight[y_i]
         * ratio_weight_i
         * weak_weight_i
```

После умножения веса нормализуются внутри batch:

```text
loss = sum(loss_i) / sum(sample_weight_i)
```

Это важно, чтобы масштаб градиента не прыгал между batch.

#### Class weights v2

Вместо raw inverse-frequency:

```text
raw_inverse: w_c = N / (C * n_c)
```

использовать один из безопасных вариантов:

**Вариант A — sqrt inverse frequency:**

```text
w_c = sqrt(N / (C * n_c))
w_c = clip(w_c, 0.5, 2.5)
normalize mean(w_c)=1
```

**Вариант B — effective number of samples:**

```text
w_c = (1 - beta) / (1 - beta ^ n_c)
beta in {0.99, 0.995, 0.999}
w_c = clip(w_c, 0.5, 3.0)
normalize mean(w_c)=1
```

**Acceptance:** weighted loss v2 принимается только если:
- OOF Macro F1 не ниже CE baseline более чем на 0.3 pp;
- F1 классов `5` и `11` растёт;
- нет падения более 2 pp у классов `0`, `4`, `7`, `8`, `9`, `12`, `14`, `15`, `19`.

---

### 5.3. Sampler v2

Текущий balanced sampler равняет классы слишком резко. Нужен мягкий class-aware sampler.

Пусть:
- `p_emp(c) = n_c / N`;
- `p_uniform(c) = 1 / C`.

Сэмплируем классы из смеси:

```text
q(c) = (1 - lambda) * p_emp(c) + lambda * p_uniform(c)
lambda in {0.3, 0.5, 0.7}
```

Для каждого класса выбираем случайный sample этого класса. Это даёт controlled oversampling minority без полного разрушения natural distribution.

**Дополнительно:** cap repeat factor:

```text
repeat_factor_c = sqrt(target_freq / freq_c)
repeat_factor_c = clip(repeat_factor_c, 1.0, 4.0)
```

**Acceptance:** sampler v2 принимается, если:
- OOF Macro F1 ≥ текущего `cv03_balanced_sampler` или хотя бы ≥ CE baseline;
- F1 `5` ≥ 0.35;
- F1 `11` ≥ 0.50;
- F1 `2`, `3`, `17` не падают относительно CE baseline более чем на 2 pp.

---

### 5.4. Weak labels для rare-class uplift

Текущие weak labels:

| class_id | label | weak rows |
|---:|---|---:|
| 5 | кабинет | 20 451 |
| 6 | детская | 14 739 |
| 11 | гардеробная / кладовая / постирочная | 14 793 |

Риск: weak labels очень много относительно real labels; если их просто смешать, они задавят настоящую разметку.

#### Рекомендуемый режим

1. **Сначала построить manifest для weak images.**
   - скачать/проверить изображения;
   - посчитать hash;
   - удалить overlap с train/val/test по hash и image_id;
   - удалить broken/missing;
   - сделать manual QA sample: минимум 100 изображений на источник.

2. **Использовать weak labels только в train folds.**
   - validation/shadow/test не трогать;
   - weak labels не должны попадать в OOF validation.

3. **Ограничить вклад weak labels.**
   - `weak_weight` старт: `0.15`;
   - максимум weak samples per epoch:
     ```text
     class 5: min(2 * real_count_5, 500)
     class 6: min(1 * real_count_6, 500)
     class 11: min(2 * real_count_11, 500)
     ```
   - либо two-stage:
     - stage A: pretrain head/backbone на weak labels 1–2 эпохи с low LR;
     - stage B: fine-tune на clean real labels.

4. **Считать class-wise delta.**
   - главная метрика ветки: F1 классов `5`, `6`, `11`;
   - guardrail: общий Macro F1 и падение соседних классов.

---

### 5.5. Ratio weighting после фикса бага

После добавления `ratio` в training DataFrame:

```text
ratio_weight = clip(ratio, 0.75, 1.0)
```

Для weak labels:

```text
ratio_weight = 1.0
weak_weight = 0.15..0.35
```

Не использовать `ratio` как feature на inference. Это только training sample weight.

**Acceptance:** ratio weighting принимается, если:
- OOF Macro F1 +0.3 pp к CE baseline или повышает стабильность rare/ambiguous классов;
- нет падения по class `5/11`;
- результат отличается от CE baseline не только в названии конфигурации.

---

### 5.6. Logit adjustment / class-bias tuning

Для multiclass argmax threshold tuning делается через class bias:

```text
pred = argmax(logits_c + bias_c)
```

Bias подбирается на OOF predictions для максимизации Macro F1. Ограничения:
- `bias_c` регуляризуется;
- нельзя подбирать на shadow holdout;
- итоговый bias фиксируется в config.

Альтернатива: prior correction:

```text
adjusted_logit_c = logit_c - tau * log(train_prior_c)
tau in {0.25, 0.5, 0.75, 1.0}
```

**Acceptance:** bias/prior correction принимается, если uplift Macro F1 ≥ 0.5 pp на OOF и не ломает shadow present-class macro.

---

## 6. Error analysis: ключевые пары ошибок

По confusion matrix `cv03_balanced_sampler` основные путаемые пары:

| true | pred | count | Интерпретация |
|---|---|---:|---|
| гостиная | универсальная комната | 45 | визуальная близость общих комнат |
| комната без мебели | не могу дать ответ / не ясно | 38 | ambiguous/empty-room vs unclear |
| туалет | совмещенный санузел | 36 | близкие санитарные классы |
| совмещенный санузел | туалет | 35 | близкие санитарные классы |
| ванная комната | совмещенный санузел | 34 | нужен контекст ванны/унитаза |
| универсальная комната | спальня | 34 | мебель/кровать как слабый сигнал |
| универсальная комната | гостиная | 34 | визуальная близость |
| кухня-гостиная | кухня / столовая | 31 | open-space vs kitchen-only |
| подъезд / лестничная площадка | коридор / прихожая | 30 | interior hallway ambiguity |
| коридор / прихожая | не могу дать ответ / не ясно | 30 | poor quality/ambiguous scenes |

**Следующий шаг:** собрать по 20–30 примеров на каждую пару и разметить причину:
- шум таргета;
- недостаток контекста;
- очень похожие классы;
- плохой crop;
- каталожное/нереалистичное фото;
- изображение не комнаты.

---

## 7. Целевое решение v2

### 7.1. Архитектура

**MVP release v2:**

```text
image -> preprocessing -> ConvNeXt/EfficientNet/ViT backbone -> logits
      -> optional logit adjustment / class bias
      -> Predicted class
```

**Candidate v3:**

```text
full image branch logits
+ smart-crop branch logits
+ inference-compatible metadata
-> OOF stacker / weighted average
-> calibrated logits
-> Predicted class
```

### 7.2. Модели-кандидаты

1. **ConvNeXt-Tiny** — текущий baseline, быстрый и сильный.
2. **EfficientNetV2-S / EfficientNet-B3** — более компактный, полезен как декоррелированный кандидат.
3. **Swin-Tiny / DeiT-Small / ViT-Small** — кандидат на diversity, если GPU позволяет.
4. **CLIP zero-shot / CLIP linear probe** — не как основной путь, а как sanity check и источник embeddings для fusion.

### 7.3. Training setup

- fixed `splits_v1` для сравнимых экспериментов;
- optional `dev_holdout_v2` с покрытием всех классов;
- AdamW;
- cosine scheduler + warmup;
- mixed precision;
- early stopping;
- label smoothing `0.03..0.07`;
- image size experiments: `224`, `288`, `320`;
- safe augmentations only;
- MLflow logging;
- OOF artifacts for every serious run.

### 7.4. Inference setup

- deterministic inference;
- batch inference;
- optional TTA:
  - center crop;
  - resize/center;
  - horizontal flip only if validation gain exists;
- output CSV exactly:
  ```csv
  image_id_ext,Predicted
  ...
  ```
- validator checks:
  - columns;
  - row count equals `test_df`;
  - no duplicate `image_id_ext`;
  - all test IDs covered exactly once;
  - class range matches confirmed schema;
  - no NaN;
  - integer predictions.

---

## 8. Product requirements

### 8.1. User stories

1. **Как участник хакатона**, я хочу запустить training по config, чтобы воспроизвести OOF Macro F1 и сравнить эксперименты.
2. **Как участник хакатона**, я хочу сгенерировать валидный submission CSV одной командой.
3. **Как эксперт**, я хочу видеть per-class F1, confusion matrix и примеры ошибок, чтобы понимать, где модель сильна и слаба.
4. **Как заказчик**, я хочу увидеть demo, чтобы проверить practical applicability.
5. **Как ML engineer**, я хочу иметь model card, config, checkpoint hash и deterministic inference, чтобы решение можно было воспроизвести локально.

### 8.2. Functional requirements

| ID | Требование | Priority |
|---|---|---|
| FR-01 | Подтвердить class schema и допустимый диапазон `Predicted` | P0 |
| FR-02 | Обучение image model из YAML config | P0 |
| FR-03 | Отдельный inference pipeline | P0 |
| FR-04 | Submission validator | P0 |
| FR-05 | OOF + shadow metrics report | P0 |
| FR-06 | Per-class metrics и top confusion pairs | P0 |
| FR-07 | Работающий class imbalance strategy v2 | P0 |
| FR-08 | Исправленная ratio weighting ветка | P0 |
| FR-09 | Release bundle `releases/rc1/` | P0 |
| FR-10 | Gradio/Streamlit demo | P1 |
| FR-11 | Grad-CAM / interpretability report | P1 |
| FR-12 | Weak-label rare-class branch | P1 |
| FR-13 | Late fusion с inference-compatible features | P2 |

### 8.3. Non-functional requirements

| ID | Требование | Целевое состояние |
|---|---|---|
| NFR-01 | Reproducibility | одна команда для train, одна для infer, frozen config |
| NFR-02 | Determinism | одинаковый submission при повторном запуске |
| NFR-03 | Runtime | inference test set batch mode, без ручных notebooks |
| NFR-04 | Local run | open-source модели, локальный запуск |
| NFR-05 | Documentation | README + final_report + model_card + runbook |
| NFR-06 | Observability | MLflow, artifact registry, metrics reports |
| NFR-07 | Safety against leakage | no sampling before split, no target-derived features at inference |
| NFR-08 | Extensibility | легко добавить backbone/fusion branch через config |

---

## 9. Гипотезы улучшения качества

| ID | Гипотеза | Ожидаемый эффект | Риск | Как проверить |
|---|---|---:|---|---|
| H1 | Исправить `ratio` и использовать clipped sample weights | +0.3–1.0 pp Macro F1 или стабильнее noisy classes | эффект может быть малым | CE vs ratio CE на fixed folds |
| H2 | Мягкий class-aware sampler вместо full balanced | +0.5–1.5 pp и меньше деградации common classes | нужно подобрать lambda | ablation lambda 0.3/0.5/0.7 |
| H3 | Effective-number/capped weights | +0.3–1.0 pp rare classes | перетюнинг weights | compare against CE/balanced |
| H4 | Weak labels для `5/6/11` с low weight/quota | +2–10 pp по class 5/11 | шум weak labels | class-wise uplift report |
| H5 | Class bias / logit adjustment по OOF | +0.5–1.5 pp Macro F1 | overfit на OOF | nested/CV bias validation |
| H6 | Larger image size 288/320 | +0.5–2 pp | GPU cost | same backbone, fixed recipe |
| H7 | Второй backbone для ensemble | +0.5–2 pp | latency/complexity | OOF diversity + weighted average |
| H8 | Smart-crop dual-view | +0.5–1.5 pp для предметных/санузлов/балконов | crop может потерять контекст | full vs crop vs full+crop |
| H9 | Late fusion с Perform/smart-crop/text/person только если train/test aligned | +0.5–2 pp | leakage/schema mismatch | coverage + adversarial validation |
| H10 | Grad-CAM guided error analysis | не прямой uplift, но улучшает защиту и debugging | время | sample grids + decisions |

---

## 10. Acceptance criteria для финального RC

### Model quality

Минимум:
- OOF Macro F1 ≥ 0.63.
- Shadow present-class Macro F1 не хуже текущего best.
- Нет классов с OOF F1 = 0.
- Для классов `5` и `11`: F1 не ниже baseline CE и желательно выше.
- Для класса `18`: отдельная оценка через OOF, так как shadow support = 0.

Цель:
- OOF Macro F1 ≥ 0.65.
- Shadow present-class Macro F1 ≥ 0.72.
- F1 class 5 ≥ 0.40.
- F1 class 11 ≥ 0.52.
- F1 class 18 ≥ 0.32.

### Engineering

- `make test` зелёный.
- `make smoke-train` зелёный.
- `python -m src.inference.predict --config ...` создаёт submission.
- `python -m src.inference.validate_submission ...` проходит без ошибок.
- `releases/rc1/` содержит:
  - config;
  - checkpoints;
  - class mapping;
  - metrics report;
  - model card;
  - submission;
  - checksums;
  - license notes.

### Documentation and demo

- README содержит train/infer/demo команды.
- `final_report.md` содержит pipeline, метрики, ошибки, ограничения, вклад команды.
- Demo запускается локально.
- Grad-CAM examples готовы для защиты.
- Есть слайды/тезисы для вопроса “что сделали с дисбалансом классов?”.

---

## 11. Оценка текущего решения по критериям

Оценка ориентировочная: она основана на репозитории и локальных отчётах, а не на hidden leaderboard.

| Критерий | Максимум | Текущее состояние | Оценка риска |
|---|---:|---|---|
| Соответствие брифу | 10 | train/data/metrics есть, но нет полноценного inference/demo/submission/release | 5–7 |
| Качество модели | 20 | OOF Macro F1 best = 0.630100, формально выше 0.60 | 20 локально, hidden неизвестен |
| Работа с данными и экспериментами | 10 | сильные leakage/ratio/weak-label reports, но ratio bug и weak labels не используются | 7–8 |
| Техническая реализация | 10 | skeleton хороший, но нет inference и release bundle | 5–6 |
| Обоснованность и интерпретируемость | 5 | выбор модели понятен, но Grad-CAM/error taxonomy нет | 2–3 |
| Документация и масштабируемость | 5 | README/отчёты есть, но нет final report/runbook | 3 |
| **Итого** | **60** | сильная research-часть, незакрытый release-контур | **35–48** |

После закрытия P0 задач решение может выглядеть как полноценный production-grade хакатонный проект с локальной модельной частью на 20/20 и значительно меньшими рисками на защите.

---

## 12. Рекомендованный критический путь

1. Зафиксировать class schema и submission validator.
2. Реализовать inference pipeline.
3. Исправить `ratio` propagation.
4. Пересчитать CE / ratio CE / sampler v2 / capped weights.
5. Сделать rare-class board и error taxonomy.
6. Выбрать RC: single best или small ensemble.
7. Собрать `releases/rc1`.
8. Сделать demo и Grad-CAM.
9. Обновить README/final_report.
10. Подготовить финальную презентационную историю:
    - “Мы защищались от leakage”.
    - “Мы оптимизировали Macro F1, поэтому отдельно работали с rare classes”.
    - “Мы не просто получили число, а сделали воспроизводимый inference и submission”.
