#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.metrics import precision_recall_fscore_support

FOCUS_CLASSES = {2, 3, 5, 11, 17, 18}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rare/ambiguous class quality board.")
    parser.add_argument(
        "--oof",
        type=Path,
        default=Path("artifacts/oof/cv03_balanced_sampler/oof_predictions.parquet"),
    )
    parser.add_argument(
        "--baseline-oof",
        type=Path,
        default=Path("artifacts/oof/cv03_baseline_ce/oof_predictions.parquet"),
    )
    parser.add_argument("--splits", type=Path, default=Path("data/splits/splits_v1.json"))
    parser.add_argument(
        "--class-mapping", type=Path, default=Path("configs/data/class_mapping.yaml")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/rare_class_board_v1.md"))
    return parser.parse_args()


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
        ]
    )


def load_class_mapping(path: Path) -> tuple[list[int], dict[int, str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    ids = [int(value) for value in payload["prediction"]["valid_class_ids"]]
    labels = {int(key): str(value) for key, value in payload["id_to_label"].items()}
    return ids, labels


def support_from_splits(path: Path, class_ids: list[int]) -> tuple[pd.Series, pd.Series]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    train_records = [row for fold in payload["folds"] for row in fold["records"]]
    shadow_records = payload["shadow_holdout"]["records"]
    train_support = pd.Series([int(row["result"]) for row in train_records]).value_counts()
    shadow_support = pd.Series([int(row["result"]) for row in shadow_records]).value_counts()
    return (
        train_support.reindex(class_ids, fill_value=0),
        shadow_support.reindex(class_ids, fill_value=0),
    )


def per_class_metrics(frame: pd.DataFrame, class_ids: list[int]) -> dict[str, Any]:
    precision, recall, f1, _support = precision_recall_fscore_support(
        frame["target"],
        frame["pred"],
        labels=class_ids,
        zero_division=0,
    )
    predicted_support = frame["pred"].value_counts().reindex(class_ids, fill_value=0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_support": predicted_support,
    }


def main() -> None:
    args = parse_args()
    class_ids, class_labels = load_class_mapping(args.class_mapping)
    oof = pd.read_parquet(args.oof)
    train_support, shadow_support = support_from_splits(args.splits, class_ids)
    metrics = per_class_metrics(oof, class_ids)

    baseline_f1 = None
    if args.baseline_oof.exists():
        baseline = pd.read_parquet(args.baseline_oof)
        baseline_f1 = per_class_metrics(baseline, class_ids)["f1"]

    rows = []
    for index, class_id in enumerate(class_ids):
        f1 = float(metrics["f1"][index])
        base = None if baseline_f1 is None else float(baseline_f1[index])
        rows.append(
            [
                class_id,
                class_labels[class_id],
                int(train_support.loc[class_id]),
                int(shadow_support.loc[class_id]),
                f"{metrics['precision'][index]:.4f}",
                f"{metrics['recall'][index]:.4f}",
                f"{f1:.4f}",
                int(metrics["predicted_support"].loc[class_id]),
                "-" if base is None else f"{f1 - base:+.4f}",
                "yes" if class_id in FOCUS_CLASSES else "no",
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(
            [
                "# Rare-Class Board V1",
                "",
                f"- oof_predictions: `{args.oof}`",
                f"- baseline_oof: `{args.baseline_oof if args.baseline_oof.exists() else 'not_available'}`",
                "- focus_classes: `2, 3, 5, 11, 17, 18`",
                "",
                markdown_table(
                    [
                        "class_id",
                        "label",
                        "train_support",
                        "shadow_support",
                        "oof_precision",
                        "oof_recall",
                        "oof_f1",
                        "predicted_support",
                        "delta_f1_vs_baseline",
                        "focus",
                    ],
                    rows,
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
