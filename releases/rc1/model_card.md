# Model Card — RC1

## Model details

- Task: room/property image classification.
- Classes: 20 classes `0..19`.
- Backbone: `convnext_tiny.in12k_ft_in1k` from `timm`.
- Input size: 224.
- Candidate run: `release_cv03_balanced_sampler_trainval_90_10`.
- Checkpoint: `artifacts/checkpoints/release_cv03_balanced_sampler_trainval_90_10.ckpt`.

## Intended use

Generate `submission.csv` for the hackathon test set and provide a local demo for single-image inspection.

## Metrics

- Release validation Macro F1: `0.671832`.
- Release validation split: group-safe 90/10 split over combined `train_df + val_df`.
- Reference CV recipe: `cv03_balanced_sampler`.

## Data split

The release checkpoint is trained on the combined labeled data with a group-safe 90/10 validation split for early stopping. The development CV split remains documented in `data/splits/splits_v1.json` and related reports.

## Limitations

- Some visually close classes remain difficult: `2/3`, `7/8/9`, `18/19`.
- Weak labels are not included in this release checkpoint.
- Existing `releases/rc1_single/submission.csv` is not regenerated automatically when the release config changes.

## Reproducibility

```bash
uv run python -m src.inference.predict --config configs/release/rc1.yaml
uv run python -m src.inference.validate_submission --submission releases/rc1/submission.csv --test-csv data/raw/test_df.csv --class-mapping configs/data/class_mapping.yaml
```
