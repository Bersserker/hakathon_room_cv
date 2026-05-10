# Release training report

- experiment: `release_cv03_balanced_sampler_trainval_90_10`
- backbone: `convnext_tiny.in12k_ft_in1k`
- checkpoint: `artifacts/checkpoints/release_cv03_balanced_sampler_trainval_90_10.ckpt`
- manifest_used: `True`
- combined_rows: `5062`
- train_rows: `4555`
- valid_rows: `507`
- val_fraction: `0.1`
- best_epoch: `4`
- best_val_macro_f1: `0.671832`
- best_score: `0.671832`
- monitor_metric: `val_macro_f1`

## Use on server

```bash
CONFIG_PATH=configs/release/rc1_single.yaml \
uv run uvicorn serving.app.main:app --host 0.0.0.0 --port 8000
```
