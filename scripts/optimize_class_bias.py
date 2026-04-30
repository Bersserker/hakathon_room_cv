#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize class-bias vector on OOF logits.")
    parser.add_argument(
        "--oof",
        type=Path,
        default=Path("artifacts/oof/cv03_balanced_sampler/oof_predictions.parquet"),
    )
    parser.add_argument(
        "--shadow",
        type=Path,
        default=Path("artifacts/oof/cv03_balanced_sampler/shadow_holdout_predictions.parquet"),
    )
    parser.add_argument(
        "--class-mapping", type=Path, default=Path("configs/data/class_mapping.yaml")
    )
    parser.add_argument(
        "--output-yaml", type=Path, default=Path("configs/postprocess/class_bias_rc1.yaml")
    )
    parser.add_argument("--report", type=Path, default=Path("reports/class_bias_tuning_v1.md"))
    parser.add_argument("--l2", type=float, default=0.001)
    return parser.parse_args()


def load_class_ids(path: Path) -> list[int]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [int(value) for value in payload["prediction"]["valid_class_ids"]]


def load_scores(path: Path, class_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(path)
    logit_cols = [f"logit_{class_id}" for class_id in class_ids]
    prob_cols = [f"prob_{class_id}" for class_id in class_ids]
    if set(logit_cols).issubset(frame.columns):
        scores = frame[logit_cols].to_numpy(dtype=np.float64)
    elif set(prob_cols).issubset(frame.columns):
        probs = frame[prob_cols].to_numpy(dtype=np.float64)
        scores = np.log(np.clip(probs, 1e-12, 1.0))
    else:
        raise ValueError(f"{path} has neither logits nor probabilities for all classes")
    return scores, frame["target"].to_numpy(dtype=int)


def macro_f1(
    scores: np.ndarray, targets: np.ndarray, bias: np.ndarray, class_ids: list[int]
) -> float:
    preds = (scores + bias.reshape(1, -1)).argmax(axis=1)
    return float(f1_score(targets, preds, average="macro", labels=class_ids, zero_division=0))


def objective(
    scores: np.ndarray,
    targets: np.ndarray,
    bias: np.ndarray,
    class_ids: list[int],
    l2: float,
) -> float:
    return macro_f1(scores, targets, bias, class_ids) - float(l2) * float(np.square(bias).sum())


def optimize_bias(
    scores: np.ndarray, targets: np.ndarray, class_ids: list[int], l2: float
) -> np.ndarray:
    bias = np.zeros(len(class_ids), dtype=np.float64)
    best = objective(scores, targets, bias, class_ids, l2)
    for step in [1.0, 0.5, 0.25, 0.1, 0.05]:
        improved = True
        while improved:
            improved = False
            for class_index in range(len(class_ids)):
                for direction in (-1.0, 1.0):
                    candidate = bias.copy()
                    candidate[class_index] += direction * step
                    score = objective(scores, targets, candidate, class_ids, l2)
                    if score > best:
                        bias = candidate
                        best = score
                        improved = True
    return bias


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
        ]
    )


def main() -> None:
    args = parse_args()
    class_ids = load_class_ids(args.class_mapping)
    scores, targets = load_scores(args.oof, class_ids)
    zero_bias = np.zeros(len(class_ids), dtype=np.float64)
    before = macro_f1(scores, targets, zero_bias, class_ids)
    bias = optimize_bias(scores, targets, class_ids, l2=args.l2)
    after = macro_f1(scores, targets, bias, class_ids)

    shadow_before = None
    shadow_after = None
    if args.shadow.exists():
        shadow_scores, shadow_targets = load_scores(args.shadow, class_ids)
        present = sorted(int(value) for value in np.unique(shadow_targets))
        shadow_before = macro_f1(shadow_scores, shadow_targets, zero_bias, present)
        shadow_after = macro_f1(shadow_scores, shadow_targets, bias, present)

    payload = {
        "schema_version": "class_bias_v1",
        "source_oof": str(args.oof),
        "objective": "oof_macro_f1_minus_l2_bias",
        "l2": float(args.l2),
        "class_ids": class_ids,
        "class_bias": [float(value) for value in bias],
        "metrics": {
            "oof_macro_f1_before": before,
            "oof_macro_f1_after": after,
            "oof_macro_f1_delta": after - before,
            "shadow_present_macro_f1_before": shadow_before,
            "shadow_present_macro_f1_after": shadow_after,
            "shadow_present_macro_f1_delta": None
            if shadow_before is None or shadow_after is None
            else shadow_after - shadow_before,
        },
    }
    write_yaml(args.output_yaml, payload)

    rows = [[class_id, f"{bias[index]:+.4f}"] for index, class_id in enumerate(class_ids)]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "\n".join(
            [
                "# Class Bias Tuning V1",
                "",
                f"- source_oof: `{args.oof}`",
                f"- output_yaml: `{args.output_yaml}`",
                f"- l2: `{args.l2}`",
                f"- oof_macro_f1_before: `{before:.6f}`",
                f"- oof_macro_f1_after: `{after:.6f}`",
                f"- oof_macro_f1_delta: `{after - before:+.6f}`",
                f"- shadow_present_macro_f1_before: `{shadow_before}`",
                f"- shadow_present_macro_f1_after: `{shadow_after}`",
                "",
                "## Bias vector",
                markdown_table(["class_id", "bias"], rows),
                "",
                "Note: bias is optimized only on OOF predictions. Shadow is reported as a check and is not used for fitting.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.output_yaml}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
