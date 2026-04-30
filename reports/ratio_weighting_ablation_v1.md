# Ratio Weighting Ablation V1

## Status

`ratio` propagation bug is fixed:

- `src/datasets/data02_build_splits.py` requires and validates `ratio`.
- `rows_to_records()` writes `ratio` into fold and shadow records.
- `data/splits/splits_v1.json` now contains `ratio` for train/valid/shadow records.
- `RoomDataset` receives `ratio` and `loss: ratio_ce` uses `clip(ratio, 0.75, 1.0)`.
- Weighted sample loss is normalized as `sum(loss_i * weight_i) / sum(weight_i)`.

## Old run is invalid for decision-making

Existing `cv03_ratio_weighting` artifacts were produced before the fix, so they match CE baseline exactly:

| run | OOF Macro F1 | OOF Accuracy | Shadow Macro F1 all | Shadow Accuracy |
|---|---:|---:|---:|---:|
| `cv03_baseline_ce` | 0.625963 | 0.648838 | 0.665635 | 0.727463 |
| old `cv03_ratio_weighting` | 0.625963 | 0.648838 | 0.665635 | 0.727463 |

## Required next run

```bash
uv run python src/training/train_image.py --config configs/model/cv03_baseline_ce.yaml --all-folds
uv run python src/training/train_image.py --config configs/model/cv03_ratio_weighting.yaml --all-folds
uv run python scripts/make_experiment_registry.py
```

Acceptance: new ratio predictions must differ from CE and must not reduce class `5`/`11` F1.
