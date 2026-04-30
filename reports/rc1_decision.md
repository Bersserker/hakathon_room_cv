# RC1 Decision Report

## Decision

Current release candidate: `cv03_balanced_sampler`.

Reason: it has the best available full OOF Macro F1 among existing completed runs and the strongest shadow-holdout metrics before the new ratio/sampler-v2 retraining cycle.

## Evidence

See `artifacts/experiment_registry.csv`.

| run | OOF Macro F1 | Shadow Macro F1 all | Shadow Macro F1 present | class 5 F1 | class 11 F1 | class 18 F1 | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `cv03_balanced_sampler` | 0.630100 | 0.710066 | 0.747438 | 0.3584 | 0.4964 | 0.2923 | RC1 candidate |
| `cv03_baseline_ce` | 0.625963 | 0.665635 | 0.700668 | 0.2062 | 0.5055 | 0.2375 | baseline |
| `cv03_ratio_weighting` | 0.625963 | 0.665635 | 0.700668 | 0.2062 | 0.5055 | 0.2375 | invalid old run: ratio was not propagated |
| `cv03_weighted_ce` | 0.556744 | 0.675242 | 0.710781 | 0.3231 | 0.3671 | 0.1716 | rejected: OOF degradation |
| `model2_v1` | 0.510551 | 0.540945 | 0.569416 | 0.1679 | 0.1591 | 0.2118 | rejected |

## Postprocessing check

`configs/postprocess/class_bias_rc1.yaml` improves OOF Macro F1 from `0.630100` to `0.646703`, but decreases shadow present-class Macro F1 from `0.747438` to `0.738959`. Therefore class bias is prepared as an optional postprocessing artifact, but is not enabled in `configs/release/rc1.yaml` by default.

## Known limitations

- `ratio` propagation is fixed in code and in `data/splits/splits_v1.json`, but `cv03_ratio_weighting` must be retrained before it can be considered.
- Shadow holdout has zero support for class `18`; OOF remains the primary all-class gate.
- Weak labels are not included in RC1.
