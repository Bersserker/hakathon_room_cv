# Imbalance Strategy V1

## Selected RC1 strategy

Use `cv03_balanced_sampler` as the current RC1 model because it is the strongest completed run:

- OOF Macro F1: `0.630100`.
- Shadow present-class Macro F1: `0.747438`.
- Class `5` F1 improves from `0.2062` to `0.3584` vs CE baseline.
- Class `18` F1 improves from `0.2375` to `0.2923` vs CE baseline.

## Implemented for next ablations

The training pipeline now supports safer imbalance controls:

1. `class_weight_policy: sqrt_inv` with clipping.
2. `class_weight_policy: effective_num` with `effective_beta` and clipping.
3. `sampler: class_aware_mixture` with `sampler_mixture_lambda`.
4. `sampler: repeat_factor` with repeat cap.
5. `loss: ratio_ce` now receives real `ratio` from split records and normalizes by sum of sample weights.

Configs prepared:

- `configs/model/class_weights_sqrt_inv.yaml`.
- `configs/model/class_weights_effective_num_0995.yaml`.
- `configs/model/sampler_class_aware_l05.yaml`.
- `configs/model/imbalance_rc1.yaml`.

## Rejected or not enabled by default

- Raw inverse-frequency weighted CE: rejected because OOF Macro F1 drops to `0.556744`.
- Old `cv03_ratio_weighting`: rejected as invalid because old split records did not include `ratio`.
- Class bias: optional only. It improves OOF but slightly decreases shadow present-class Macro F1.
- Weak labels: not included in RC1 because hash-dedup/manifest QA is not complete.

## Guardrails

Acceptance for future retraining:

- OOF Macro F1 must be at least CE baseline.
- Class `5` F1 target: `>= 0.35`.
- Class `11` F1 target: `>= 0.50`.
- Classes `2`, `3`, `17` must not drop by more than 2 pp vs CE baseline.
