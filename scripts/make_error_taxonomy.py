#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_REASONS = {
    (3, 2): "visual similarity: living/common room",
    (19, 18): "ambiguous empty-room vs unclear image",
    (8, 9): "close sanitary classes",
    (9, 8): "close sanitary classes",
    (7, 9): "bathroom vs combined bathroom context",
    (2, 4): "generic room vs bedroom furniture cue",
    (2, 3): "visual similarity: common rooms",
    (1, 0): "open-space kitchen-living vs kitchen-only",
    (15, 10): "building hallway vs apartment hallway",
    (10, 18): "poor quality / ambiguous hallway",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build top-confusion error taxonomy.")
    parser.add_argument(
        "--oof",
        type=Path,
        default=Path("artifacts/oof/cv03_balanced_sampler/oof_predictions.parquet"),
    )
    parser.add_argument(
        "--class-mapping", type=Path, default=Path("configs/data/class_mapping.yaml")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/error_taxonomy_v1.md"))
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def load_labels(path: Path) -> dict[int, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {int(key): str(value) for key, value in payload["id_to_label"].items()}


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
    labels = load_labels(args.class_mapping)
    frame = pd.read_parquet(args.oof)
    errors = frame.loc[frame["target"] != frame["pred"]].copy()
    pairs = (
        errors.groupby(["target", "pred"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(args.top_k)
    )

    rows = []
    samples = []
    for row in pairs.itertuples(index=False):
        true_id = int(row.target)
        pred_id = int(row.pred)
        pair_errors = errors.loc[(errors["target"] == true_id) & (errors["pred"] == pred_id)]
        sample_ids = pair_errors["image_id_ext"].head(10).tolist()
        reason = DEFAULT_REASONS.get((true_id, pred_id), "needs manual review")
        rows.append([true_id, labels[true_id], pred_id, labels[pred_id], int(row.count), reason])
        samples.append(
            f"### true `{true_id}` → pred `{pred_id}`\n\n"
            f"- reason: {reason}\n"
            f"- sample_image_id_ext: `{sample_ids}`\n"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(
            [
                "# Error Taxonomy V1",
                "",
                f"- source_oof: `{args.oof}`",
                f"- top_pairs: `{args.top_k}`",
                "",
                markdown_table(
                    ["true", "true_label", "pred", "pred_label", "count", "taxonomy"], rows
                ),
                "",
                "## Samples for manual review",
                "",
                *samples,
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
