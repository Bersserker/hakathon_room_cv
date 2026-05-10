# Release training report

- experiment: `cv05_rtx5070_xlarge_512_focal_es`
- backbone: `convnext_xlarge.fb_in22k_ft_in1k_384`
- checkpoint: `artifacts\checkpoints\roomclf_convnextxlargefbin22kftin1k384_release_512_cv05_rtx5070_xlarge_512_focal_es.ckpt`
- manifest_used: `True`
- combined_rows: `5039`
- train_rows: `4535`
- valid_rows: `504`
- val_fraction: `0.1`
- best_epoch: `6`
- best_val_macro_f1: `0.766045`
- best_score: `0.766045`
- monitor_metric: `val_macro_f1`

## Use on server

```bash
CONFIG_PATH=configs/release/rc1_single.yaml \
uv run uvicorn serving.app.main:app --host 0.0.0.0 --port 8000
```
