#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _bootstrap import bootstrap_repo_root

bootstrap_repo_root()

from src.diagnostics.adversarial_cv import CVResult, run_adversarial_cv
from src.diagnostics.adversarial_data import (
    FORBIDDEN_FEATURE_COLUMNS,
    class_balance_sources,
    label_shift_table,
    load_domain_frames_from_splits,
)
from src.diagnostics.adversarial_features import build_metadata_matrix, build_visual_matrix
from src.diagnostics.adversarial_report import build_adversarial_markdown_report, utc_now
from src.diagnostics.embeddings import ClipEmbeddingExtractor, ensure_embeddings


DEFAULT_SPLITS_JSON = Path("data/splits/splits_v1.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/diagnostics/adversarial_validation")
DEFAULT_REPORT_MD = Path("reports/adversarial_validation.md")
DEFAULT_SEED = 26042026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adversarial validation: train/dev pool vs shadow holdout."
    )
    parser.add_argument("--splits-json", type=Path, default=DEFAULT_SPLITS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--clip-model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-processor-name", default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument(
        "--allow-missing-image-paths",
        action="store_true",
        help="Only for metadata debugging; default fails if split records point to missing images.",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    return value


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _with_result_keys(df: pd.DataFrame, result: CVResult) -> pd.DataFrame:
    output = df.copy()
    output.insert(0, "feature_set", result.feature_set)
    output.insert(0, "mode", result.mode)
    return output


def write_artifacts(
    *,
    output_dir: Path,
    report_md: Path,
    label_shift: pd.DataFrame,
    balance_audit: dict[str, Any],
    results: list[CVResult],
    command: str,
    generated_at: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    fold_metrics = pd.concat(
        [_with_result_keys(result.fold_metrics, result) for result in results],
        ignore_index=True,
        sort=False,
    )
    predictions = pd.concat(
        [result.predictions for result in results],
        ignore_index=True,
        sort=False,
    )
    feature_importance = pd.concat(
        [
            result.feature_importance
            for result in results
            if result.feature_set == "metadata" and not result.feature_importance.empty
        ],
        ignore_index=True,
        sort=False,
    )
    per_class = pd.concat(
        [result.per_class_summary for result in results],
        ignore_index=True,
        sort=False,
    )
    top_examples = pd.concat(
        [result.top_examples for result in results],
        ignore_index=True,
        sort=False,
    )

    _write_csv(output_dir / "label_shift.csv", label_shift)
    _write_csv(output_dir / "fold_metrics.csv", fold_metrics)
    _write_csv(output_dir / "predictions.csv", predictions)
    _write_csv(output_dir / "metadata_feature_importance.csv", feature_importance)
    _write_csv(output_dir / "per_class_summary.csv", per_class)
    _write_csv(output_dir / "top_examples.csv", top_examples)

    summary_payload = {
        "generated_at_utc": generated_at,
        "policy": {
            "domain_label_0": "train_dev_pool",
            "domain_label_1": "shadow_holdout",
            "main_gate": "class_balanced_visual_roc_auc",
            "excluded_features": sorted(FORBIDDEN_FEATURE_COLUMNS),
            "warning": "Low AUC does not prove absence of all shift; shadow holdout is diagnostic only.",
        },
        "balance_audit": balance_audit,
        "results": [result.summary for result in results],
        "artifacts": {
            "label_shift_csv": str(output_dir / "label_shift.csv"),
            "fold_metrics_csv": str(output_dir / "fold_metrics.csv"),
            "predictions_csv": str(output_dir / "predictions.csv"),
            "metadata_feature_importance_csv": str(output_dir / "metadata_feature_importance.csv"),
            "per_class_summary_csv": str(output_dir / "per_class_summary.csv"),
            "top_examples_csv": str(output_dir / "top_examples.csv"),
            "report_md": str(report_md),
        },
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(_json_safe(summary_payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    report = build_adversarial_markdown_report(
        label_shift=label_shift,
        results=results,
        balance_audit=balance_audit,
        command=command,
        generated_at=generated_at,
    )
    report_md.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    embedding_cache = args.embedding_cache or (args.output_dir / "clip_embeddings.parquet")
    generated_at = utc_now()

    frames = load_domain_frames_from_splits(
        args.splits_json,
        require_image_paths=not args.allow_missing_image_paths,
    )
    raw_df = frames.combined
    balanced_df, balance_audit = class_balance_sources(raw_df, seed=args.seed)
    label_shift = label_shift_table(raw_df)

    mode_frames = {
        "raw": raw_df,
        "class_balanced": balanced_df,
    }
    results: list[CVResult] = []

    for mode, frame in mode_frames.items():
        metadata_x, metadata_features = build_metadata_matrix(frame)
        results.append(
            run_adversarial_cv(
                frame,
                metadata_x,
                metadata_features,
                mode=mode,
                feature_set="metadata",
                n_splits=args.n_folds,
                seed=args.seed,
            )
        )

    if not args.skip_visual:
        extractor = ClipEmbeddingExtractor(
            args.clip_model_name,
            processor_name_or_path=args.clip_processor_name,
        )
        embeddings = ensure_embeddings(
            raw_df,
            extractor=extractor,
            cache_path=embedding_cache,
            batch_size=args.batch_size,
        )
        for mode, frame in mode_frames.items():
            visual_x, visual_features = build_visual_matrix(frame, embeddings)
            results.append(
                run_adversarial_cv(
                    frame,
                    visual_x,
                    visual_features,
                    mode=mode,
                    feature_set="visual",
                    n_splits=args.n_folds,
                    seed=args.seed,
                )
            )

    command = "python scripts/run_adversarial_validation.py " + " ".join(
        [
            f"--splits-json {args.splits_json}",
            f"--output-dir {args.output_dir}",
            f"--report-md {args.report_md}",
            f"--embedding-cache {embedding_cache}",
            f"--clip-model-name {args.clip_model_name}",
            f"--n-folds {args.n_folds}",
            f"--seed {args.seed}",
        ]
    )
    if args.clip_processor_name is not None:
        command += f" --clip-processor-name {args.clip_processor_name}"
    if args.skip_visual:
        command += " --skip-visual"

    write_artifacts(
        output_dir=args.output_dir,
        report_md=args.report_md,
        label_shift=label_shift,
        balance_audit=balance_audit,
        results=results,
        command=command,
        generated_at=generated_at,
    )
    print(f"Wrote adversarial validation report: {args.report_md}")
    print(f"Wrote machine-readable artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
