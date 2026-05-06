from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from src.diagnostics.adversarial_cv import CVResult
from src.diagnostics.adversarial_data import FORBIDDEN_FEATURE_COLUMNS


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def markdown_table(df: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    existing = [column for column in columns if column in df.columns]
    if df.empty or not existing:
        return "_No rows._"
    sample = df[existing].head(max_rows).copy()
    for column in sample.columns:
        if pd.api.types.is_float_dtype(sample[column]):
            sample[column] = sample[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            sample[column] = sample[column].fillna("").astype(str)
    header = "| " + " | ".join(existing) + " |"
    sep = "| " + " | ".join(["---"] * len(existing)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in sample.astype(str).to_numpy()]
    suffix = ""
    if len(df) > max_rows:
        suffix = f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return "\n".join([header, sep, *rows]) + suffix


def _summary_frame(results: Iterable[CVResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {
            "mode": result.mode,
            "feature_set": result.feature_set,
            "rows": result.summary["rows"],
            "features": result.summary["features"],
            "roc_auc_mean": result.summary["roc_auc_mean"],
            "roc_auc_std": result.summary["roc_auc_std"],
            "roc_auc_ci95_half_width": result.summary["roc_auc_ci95_half_width"],
            "pr_auc_mean": result.summary["pr_auc_mean"],
            "balanced_accuracy_mean": result.summary["balanced_accuracy_mean"],
            "brier_score_mean": result.summary["brier_score_mean"],
            "log_loss_mean": result.summary["log_loss_mean"],
            "interpretation": result.summary["roc_auc_interpretation"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _find_result(results: Iterable[CVResult], *, mode: str, feature_set: str) -> CVResult | None:
    for result in results:
        if result.mode == mode and result.feature_set == feature_set:
            return result
    return None


def build_adversarial_markdown_report(
    *,
    label_shift: pd.DataFrame,
    results: list[CVResult],
    balance_audit: dict,
    command: str,
    generated_at: str | None = None,
    excluded_features: Iterable[str] = FORBIDDEN_FEATURE_COLUMNS,
) -> str:
    generated_at = generated_at or utc_now()
    summary = _summary_frame(results)
    main = _find_result(results, mode="class_balanced", feature_set="visual")

    if main is None:
        main_text = (
            "Visual class-balanced result is unavailable; do not draw the main covariate-shift "
            "conclusion until visual embeddings are evaluated."
        )
    else:
        main_text = (
            f"Class-balanced visual ROC-AUC = {main.summary['roc_auc_mean']:.4f} "
            f"± {main.summary['roc_auc_std']:.4f} across folds "
            f"({main.summary['roc_auc_interpretation']})."
        )

    top_label_shift = label_shift.copy()
    if not top_label_shift.empty:
        top_label_shift["abs_pct_diff"] = top_label_shift["pct_diff_holdout_minus_train"].abs()
        top_label_shift = top_label_shift.sort_values(
            "abs_pct_diff", ascending=False, kind="stable"
        )

    metadata_importance = pd.concat(
        [result.feature_importance for result in results if result.feature_set == "metadata"],
        ignore_index=True,
        sort=False,
    ) if results else pd.DataFrame()
    top_examples = pd.concat(
        [result.top_examples for result in results],
        ignore_index=True,
        sort=False,
    ) if results else pd.DataFrame()
    per_class = pd.concat(
        [result.per_class_summary for result in results],
        ignore_index=True,
        sort=False,
    ) if results else pd.DataFrame()

    missing_holdout = balance_audit.get("missing_from_holdout_classes", [])
    class_18_note = (
        "Class 18 is absent from shadow holdout and is treated as label shift, not covariate shift."
        if 18 in missing_holdout
        else "No special class-18 exclusion was needed beyond the shared-class filter."
    )

    lines = [
        "# Adversarial validation: train/dev pool vs shadow holdout",
        "",
        "## Executive summary",
        f"- generated_at_utc: `{generated_at}`",
        "- main covariate-shift gate: **class-balanced visual classifier**.",
        f"- main result: {main_text}",
        "- raw mode is a warning signal because it includes known label distribution differences.",
        "- low adversarial AUC does **not** prove absence of all shift; it only means these features and this classifier did not find strong separability.",
        "- shadow holdout remains diagnostic only and must not be used for room-classifier tuning/model selection.",
        "",
        "## Interpretation thresholds",
        "- ROC-AUC near 0.50–0.60: negligible / no clear separability.",
        "- 0.60–0.70: weak shift.",
        "- 0.70–0.75: moderate shift.",
        "- 0.75–0.85: notable shift.",
        "- >0.85: severe shift.",
        "",
        "## Excluded features",
        "The adversarial classifiers exclude target labels, label names, item ids, image ids, URLs, hashes, local paths, source names, and model outputs.",
        "",
        ", ".join(f"`{feature}`" for feature in sorted(excluded_features)),
        "",
        "## Label shift report",
        markdown_table(
            top_label_shift,
            [
                "class_id",
                "label",
                "train_count",
                "holdout_count",
                "train_pct",
                "holdout_pct",
                "pct_diff_holdout_minus_train",
                "present_in_train",
                "present_in_holdout",
            ],
            max_rows=30,
        ),
        "",
        "## Class-balanced mode audit",
        f"- shared_classes: `{balance_audit.get('shared_classes', [])}`",
        f"- excluded_classes: `{balance_audit.get('excluded_classes', [])}`",
        f"- missing_from_holdout_classes: `{missing_holdout}`",
        f"- missing_from_train_classes: `{balance_audit.get('missing_from_train_classes', [])}`",
        f"- rows_before: `{balance_audit.get('rows_before')}`",
        f"- rows_after: `{balance_audit.get('rows_after')}`",
        f"- class_18_note: {class_18_note}",
        "",
        "## Raw mode and class-balanced mode metrics",
        markdown_table(
            summary,
            [
                "mode",
                "feature_set",
                "rows",
                "features",
                "roc_auc_mean",
                "roc_auc_std",
                "roc_auc_ci95_half_width",
                "pr_auc_mean",
                "balanced_accuracy_mean",
                "brier_score_mean",
                "log_loss_mean",
                "interpretation",
            ],
            max_rows=20,
        ),
        "",
        "## Metadata shift",
        "Metadata-only classifiers use annotation consensus ratio, dimensions, aspect ratio, megapixels, and orientation/shape indicators only.",
        markdown_table(
            metadata_importance,
            [
                "mode",
                "feature_set",
                "rank",
                "feature",
                "mean_abs_coefficient",
                "max_abs_coefficient",
            ],
            max_rows=20,
        ),
        "",
        "## Visual shift",
        "Visual classifiers use frozen image embeddings; no room-classifier logits/probabilities are included.",
        markdown_table(
            summary.loc[summary["feature_set"] == "visual"] if not summary.empty else summary,
            [
                "mode",
                "feature_set",
                "roc_auc_mean",
                "roc_auc_std",
                "pr_auc_mean",
                "balanced_accuracy_mean",
                "interpretation",
            ],
            max_rows=10,
        ),
        "",
        "## Per-class adversarial summaries",
        markdown_table(
            per_class,
            [
                "mode",
                "feature_set",
                "class_id",
                "label",
                "train_count",
                "holdout_count",
                "roc_auc",
                "mean_p_for_train_rows",
                "mean_p_for_holdout_rows",
            ],
            max_rows=40,
        ),
        "",
        "## Top examples for inspection",
        markdown_table(
            top_examples,
            [
                "mode",
                "feature_set",
                "example_type",
                "rank",
                "image_id_ext",
                "item_id",
                "result",
                "label",
                "domain",
                "p_shadow_holdout",
                "fold",
                "local_path",
            ],
            max_rows=60,
        ),
        "",
        "## Re-run",
        "```bash",
        command,
        "```",
        "",
    ]
    return "\n".join(lines)
