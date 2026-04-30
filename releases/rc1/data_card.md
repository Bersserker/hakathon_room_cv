# Data Card — RC1

## Sources

- `data/raw/train_df.csv` — labeled training rows with `ratio`.
- `data/raw/val_df.csv` — labeled shadow holdout with `ratio`.
- `data/raw/test_df.csv` — test rows for submission.
- `data/processed/data_manifest.parquet` — local image status, dimensions, and content hash.

## Quality checks

- Missing images are excluded by manifest status during split generation.
- Content-hash grouping prevents duplicate-image leakage across folds.
- `ratio` is used only as training sample weight, never as inference feature.

## Known limitations

- Shadow holdout lacks class `18`.
- Weak-label image manifest/dedup is not complete for RC1.
