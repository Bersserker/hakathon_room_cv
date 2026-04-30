# Agent Guide

Project-specific instructions for coding agents working in this repository.

## Read first

Before changing code or experiment configs, read:

1. `docs/CONTEXT.md`
2. `docs/ARCHITECTURE.md`
3. The specific config/report/code files touched by the task.

## Core rules

- Keep changes surgical. Do not refactor unrelated code.
- Do not train on `val_df` / shadow holdout.
- Do not use `data/raw/test_df.csv` labels; test is for submission only.
- Use `data/splits/splits_v1.json` for development training/evaluation unless the task explicitly asks to rebuild splits.
- Preserve the class schema: valid labels are integer ids `0..19`.
- Prefer CLI/scripts over notebooks for reproducible changes.
- Do not delete existing artifacts/checkpoints/reports unless explicitly asked.
- Do not run destructive git commands (`reset --hard`, `clean`, force push, branch delete) unless explicitly asked.

## Development commands

Install/update environment:

```bash
uv sync --extra dev
```

Run checks:

```bash
make lint
make test
```

Smoke train:

```bash
make smoke-train
```

Full train for a config:

```bash
uv run python src/training/train_image.py --config configs/model/<name>.yaml --all-folds
```

Inference + validation:

```bash
make infer
make validate-submission
```

## When adding a model experiment

1. Copy the closest config from `configs/model/`.
2. Set a unique:
   - `artifacts.oof_dir`
   - `artifacts.report_path`
   - `experiment.version`
   - `experiment.feature_flags`
3. Keep `data.splits_json: data/splits/splits_v1.json`.
4. Keep backbone inside `model.whitelist` or update the whitelist only if the task allows it.
5. Run debug smoke first:

```bash
uv run python src/training/train_image.py --config configs/model/<name>.yaml --fold 0 --debug
```

6. Then run full CV only when compute is available:

```bash
uv run python src/training/train_image.py --config configs/model/<name>.yaml --all-folds
```

7. Verify expected artifacts:

```text
artifacts/oof/<run>/oof_predictions.parquet
artifacts/oof/<run>/shadow_holdout_predictions.parquet
artifacts/oof/<run>/confusion_matrix_oof.csv
artifacts/oof/<run>/config.yaml
reports/<run>.md
```

## Evaluation guardrails

For serious runs, check:

- OOF rows equal `4562`.
- Shadow rows equal `477`.
- OOF has all folds `0..4`.
- OOF/shadow do not overlap by `image_id_ext`.
- Report contains OOF Macro F1, per-class F1, confusion matrix and shadow metrics.
- Prediction exports include `target`, `pred`, `prob_0..prob_19`; keep `logit_0..logit_19` where the standard export supports it.

Remember: shadow holdout has zero support for class `18`, so do not select models solely by shadow all-label Macro F1.

## When changing training code

- Preserve config backward compatibility where possible.
- Add/update tests for config parsing, sampler/loss behavior, or prediction schema when the changed logic is testable without full training.
- Run at least:

```bash
make test
```

- If train loop behavior changes, run a debug smoke train.
- Keep MLflow params/tags in sync with new experiment fields.
- Make sure `artifacts/oof/<run>/config.yaml` still captures the resolved config.

## When changing inference/release code

- Keep submission schema exactly:

```csv
image_id_ext,Predicted
```

- Predictions must be integer class ids `0..19`.
- Run:

```bash
uv run python -m src.inference.predict --config configs/release/rc1.yaml
uv run python -m src.inference.validate_submission \
  --submission releases/rc1/submission.csv \
  --test-csv data/raw/test_df.csv \
  --class-mapping configs/data/class_mapping.yaml
```

- If enabling class bias, document why in `reports/` and ensure it was tuned on OOF, not shadow.

## When editing docs/reports

- Keep paths and commands copy-pasteable from repo root.
- Prefer concise factual notes over speculation.
- If a report references a rerun command, make sure it uses the same config that produced the report.

## Common pitfalls

- Accidentally using `configs/model/image_baseline_v1.yaml` in a rerun block for a non-baseline report.
- Treating `val_df` as a validation set for training; it is shadow holdout.
- Comparing shadow Macro F1 without noting class `18` is absent.
- Forgetting to update `artifacts.oof_dir` and overwriting another experiment.
- Adding a new backbone outside whitelist without updating config and documenting why.
