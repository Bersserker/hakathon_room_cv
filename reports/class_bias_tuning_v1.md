# Class Bias Tuning V1

- source_oof: `artifacts/oof/cv03_balanced_sampler/oof_predictions.parquet`
- output_yaml: `configs/postprocess/class_bias_rc1.yaml`
- l2: `0.001`
- oof_macro_f1_before: `0.630100`
- oof_macro_f1_after: `0.646703`
- oof_macro_f1_delta: `+0.016603`
- shadow_present_macro_f1_before: `0.7474379209662779`
- shadow_present_macro_f1_after: `0.7389585425098274`

## Bias vector
| class_id | bias |
| --- | --- |
| 0 | -0.8000 |
| 1 | +0.2500 |
| 2 | +0.1000 |
| 3 | +0.0500 |
| 4 | -0.3500 |
| 5 | +0.0000 |
| 6 | +0.0000 |
| 7 | +0.6500 |
| 8 | -0.0500 |
| 9 | -0.4000 |
| 10 | -0.2500 |
| 11 | +0.0000 |
| 12 | -0.9500 |
| 13 | +0.7500 |
| 14 | +0.0000 |
| 15 | -0.0500 |
| 16 | +0.5000 |
| 17 | -0.2500 |
| 18 | +0.0000 |
| 19 | +0.3500 |

Note: bias is optimized only on OOF predictions. Shadow is reported as a check and is not used for fitting.
