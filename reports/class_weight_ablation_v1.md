# Class Weight Ablation V1

## Implemented policies

`src/training/train_image.py` now supports:

- `class_weight_policy: none`.
- `class_weight_policy: raw_inverse` for backward compatibility.
- `class_weight_policy: sqrt_inv` with clipping and mean normalization.
- `class_weight_policy: effective_num` with `effective_beta`, clipping, and mean normalization.

Prepared configs:

- `configs/model/class_weights_sqrt_inv.yaml`.
- `configs/model/class_weights_effective_num_0995.yaml`.

## Current evidence

The old raw inverse-frequency weighted CE run is rejected:

- OOF Macro F1: `0.556744` vs CE `0.625963`.
- Class `11` F1 drops from `0.5055` to `0.3671`.
- Class `18` F1 drops from `0.2375` to `0.1716`.

## Next commands

```bash
uv run python src/training/train_image.py --config configs/model/class_weights_sqrt_inv.yaml --all-folds
uv run python src/training/train_image.py --config configs/model/class_weights_effective_num_0995.yaml --all-folds
uv run python scripts/make_experiment_registry.py
```

Acceptance: OOF Macro F1 must not drop more than 0.3 pp vs CE unless rare-class uplift justifies it.
