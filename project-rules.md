## Правила code review

* В `main` напрямую никто не пушит.
* Любой PR должен быть привязан к задаче/ID и содержать:

  * что изменено;
  * как запускалось;
  * run ID или ссылку на артефакт;
  * список затронутых config/data contracts.
* Изменения в:

  * data/split logic — обязательный review от У2;
  * infer/release/Docker — обязательный review от У5;
  * training/model logic — обязательный review от У1 или У3.
* Решение только в ноутбуке не считается завершенной задачей, если нет отдельного script/config.
* Любой PR, который меняет train/infer, должен проходить smoke test.


## Правила experiment tracking

* Все запуски выше smoke-level логируются в MLflow.
* В run обязательно записываем:

  * `commit_sha`
  * `dataset_version`
  * `split_version`
  * `seed`
  * `backbone`
  * `image_size`
  * `loss`
  * `sampler`
  * `ratio_policy`
  * `weak_label_flag`
  * `feature_flags`
  * `tta_flag`
* Сравниваем только runs на одном `split_version`.
* Любая цифра в чате без run ID не участвует в решении.
* Для кандидатов в релиз обязательно сохраняем:

  * OOF predictions;
  * финальный config;
  * checkpoint;
  * inference command;
  * per-class report.
* RC-кандидаты повторно прогоняются минимум на 2 seed только если это помещается в бюджет.

## Naming convention

* **Run:**
  `rt_<split>_<ds>_<backbone>_<img>_<loss>_<sampler>_<featflag>_s<seed>_v<exp>`
  Пример: `rt_splitv1_dsmain_convnexttiny_320_wce_none_img_s42_v03`

* **Model checkpoint:**
  `roomclf_<backbone>_fold<k>_<img>_<exp>.ckpt`

* **Dataset:**
  `ds_main_v1`, `ds_weak_v1`, `feat_aux_v1`, `split_item_v1`

* **Submission:**
  `sub_<date>_<rc>_<modeltag>_<tta>.csv`
  Пример: `sub_2026-04-22_rc1_convnext_stack_tta.csv`