# Model Card — RC1

## Model details

- Task: room/property image classification.
- Classes: 20 classes `0..19`.
- Backbone: `convnext_tiny.in12k_ft_in1k` from `timm`.
- Input size: 224.
- Candidate run: `cv03_balanced_sampler`.
- Checkpoints: 5 fold checkpoints in `artifacts/checkpoints/`.

## Intended use

Generate `submission.csv` for the hackathon test set and provide a local demo for single-image inspection.

## Metrics

- OOF Macro F1: `0.630100`.
- OOF Accuracy: `0.644893`.
- Shadow Macro F1 all labels: `0.710066`.
- Shadow Macro F1 present labels: `0.747438`.
- Shadow Accuracy: `0.746331`.

## Data split

`data/splits/splits_v1.json` uses grouped stratified folds over `item_id` and `content_hash`. `val_df` is a separate shadow holdout.

## Limitations

- Shadow holdout has no class `18` support.
- Some visually close classes remain difficult: `2/3`, `7/8/9`, `18/19`.
- Weak labels are not included in RC1.
- Ratio weighting was fixed in code but needs retraining for final inclusion.

## Reproducibility

```bash
uv run python -m src.inference.predict --config configs/release/rc1.yaml
uv run python -m src.inference.validate_submission --submission releases/rc1/submission.csv --test-csv data/raw/test_df.csv --class-mapping configs/data/class_mapping.yaml
```
