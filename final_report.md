# Final Report — Room Type Image Classification

## Target schema

Confirmed schema is 20 classes `0..19`. The final submission column is `Predicted`, integer in `[0, 19]`. See `configs/data/class_mapping.yaml` and `reports/class_schema_audit.md`.

## Data and leakage control

Splits use `StratifiedGroupKFold` with connected groups over `item_id` and `content_hash`. The original `val_df` is kept as a separate shadow holdout. Leakage checks are documented in `reports/leakage_report.md`.

Important limitation: shadow holdout has zero support for class `18`, so OOF Macro F1 is the primary all-class gate. Shadow is reported both as all-label macro and present-label macro. See `reports/evaluation_protocol_v2.md`.

## Model

Current RC1 candidate is `cv03_balanced_sampler`:

- Backbone: `convnext_tiny.in12k_ft_in1k` via `timm`.
- Input size: `224` with validation resize/center-crop preprocessing.
- Objective: multiclass CE.
- Sampler: balanced sampler.
- Inference: deterministic fold/checkpoint ensemble skeleton in `src/inference/predict.py`.

## Metrics

| run | OOF Macro F1 | Shadow Macro F1 all | Shadow Macro F1 present | Decision |
|---|---:|---:|---:|---|
| `cv03_balanced_sampler` | 0.630100 | 0.710066 | 0.747438 | RC1 candidate |
| `cv03_baseline_ce` | 0.625963 | 0.665635 | 0.700668 | baseline |
| `cv03_weighted_ce` | 0.556744 | 0.675242 | 0.710781 | rejected |

Full registry: `artifacts/experiment_registry.csv`.

## Imbalance handling

The current best run improves rare class `5` substantially, but hurts some ambiguous classes. The code now supports safer next-step ablations:

- clipped sqrt inverse weights;
- effective-number weights;
- class-aware mixture sampler;
- repeat-factor sampler;
- fixed ratio weighting with per-batch weight normalization.

See `reports/imbalance_strategy_v1.md`.

## Inference and submission

```bash
uv run python -m src.inference.predict --config configs/release/rc1.yaml
uv run python -m src.inference.validate_submission --submission releases/rc1/submission.csv --test-csv data/raw/test_df.csv --class-mapping configs/data/class_mapping.yaml
```

The validator checks exact columns, row count, ID coverage, duplicates, NaN, integer predictions, valid class range, and prints SHA256.

## Demo

```bash
uv run python demo/app.py --config configs/release/rc1.yaml
```

## Error analysis

See:

- `reports/rare_class_board_v1.md`;
- `reports/error_taxonomy_v1.md`;
- `reports/class_bias_tuning_v1.md`.

Class-bias tuning improves OOF but slightly decreases shadow present-class Macro F1, so it is not enabled by default in RC1.

## Limitations

- Ratio weighting must be retrained after the propagation fix.
- Weak labels are prepared but not included in RC1 because weak-image manifest/hash QA is incomplete.
- Grad-CAM visual examples are not yet generated.
