# Evaluation Protocol V2

## Shadow holdout reporting rule
- Report two shadow metrics for every serious run:
  - `shadow_macro_f1_all_labels`: macro over all classes `0..19`.
  - `shadow_macro_f1_present_labels`: macro only over classes present in shadow `y_true`.
- Use OOF as the primary release-candidate gate for full 20-class coverage.
- Do not select postprocessing/bias on shadow; shadow is a check only.

## Current shadow support
| class_id | shadow_support | warning_support_zero |
| --- | --- | --- |
| 0 | 34 | no |
| 1 | 15 | no |
| 2 | 23 | no |
| 3 | 30 | no |
| 4 | 24 | no |
| 5 | 21 | no |
| 6 | 31 | no |
| 7 | 23 | no |
| 8 | 28 | no |
| 9 | 30 | no |
| 10 | 28 | no |
| 11 | 23 | no |
| 12 | 22 | no |
| 13 | 21 | no |
| 14 | 21 | no |
| 15 | 31 | no |
| 16 | 28 | no |
| 17 | 23 | no |
| 18 | 0 | yes |
| 19 | 21 | no |

## Current reference: cv03_balanced_sampler
- shadow_macro_f1_all_labels: `0.710066`
- shadow_macro_f1_present_labels: `0.747438`
- warning: class `18` has zero shadow support.

## Implementation
- `src/training/train_image.py` now writes both shadow macro variants in future training reports.
- `scripts/make_experiment_registry.py` stores both metrics in `artifacts/experiment_registry.csv`.
