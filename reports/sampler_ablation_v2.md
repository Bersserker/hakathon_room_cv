# Sampler Ablation V2

## Implemented policies

`src/training/train_image.py` now supports:

- `sampler: shuffle`.
- `sampler: balanced`.
- `sampler: class_aware_mixture` with `sampler_mixture_lambda`.
- `sampler: repeat_factor` with `repeat_factor_cap`.

Prepared config:

- `configs/model/sampler_class_aware_l05.yaml`.

## Current reference

`cv03_balanced_sampler` is the current best completed run:

- OOF Macro F1: `0.630100`.
- Class `5` F1: `0.3584` vs CE `0.2062`.
- Class `11` F1: `0.4964` vs CE `0.5055`.
- Classes `2`, `3`, `17` drop vs CE, so a softer sampler remains desirable.

## Next commands

```bash
uv run python src/training/train_image.py --config configs/model/sampler_class_aware_l05.yaml --all-folds
uv run python scripts/make_rare_class_board.py --oof artifacts/oof/sampler_class_aware_l05/oof_predictions.parquet
uv run python scripts/make_experiment_registry.py
```

Acceptance: OOF Macro F1 must be at least CE baseline; class `5` F1 target `>= 0.35`; classes `2`, `3`, `17` should not drop more than 2 pp vs CE.
