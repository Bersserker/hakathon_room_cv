from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.diagnostics.adversarial_data import HOLDOUT_DOMAIN_LABEL, TRAIN_DOMAIN_LABEL


METRIC_NAMES = [
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "brier_score",
    "log_loss",
    "mean_p_holdout",
    "mean_p_for_train_rows",
    "mean_p_for_holdout_rows",
]


@dataclass(frozen=True)
class CVResult:
    mode: str
    feature_set: str
    fold_metrics: pd.DataFrame
    summary: dict[str, Any]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    per_class_summary: pd.DataFrame
    top_examples: pd.DataFrame


def make_group_stratified_folds(
    df: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
    domain_col: str = "domain_label",
    group_col: str = "item_id",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create domain-stratified folds that keep item/listing groups intact."""
    missing = sorted({domain_col, group_col}.difference(df.columns))
    if missing:
        raise ValueError(f"CV dataframe missing required columns: {missing}")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    group_domain_counts = df.groupby(group_col)[domain_col].nunique()
    leaking_groups = group_domain_counts.loc[group_domain_counts > 1]
    if not leaking_groups.empty:
        sample = leaking_groups.head(10).index.astype(str).tolist()
        raise ValueError(f"Item groups span both domains, cannot build safe folds: {sample}")

    group_domains = df.groupby(group_col)[domain_col].first().astype(int)
    groups_per_domain = group_domains.value_counts().to_dict()
    for label in [TRAIN_DOMAIN_LABEL, HOLDOUT_DOMAIN_LABEL]:
        if groups_per_domain.get(label, 0) < n_splits:
            raise ValueError(
                f"Need at least {n_splits} item groups for domain_label={label}; "
                f"got {groups_per_domain.get(label, 0)}."
            )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    y = df[domain_col].astype(int).to_numpy()
    groups = df[group_col].astype(str).to_numpy()
    dummy_x = np.zeros((len(df), 1), dtype=np.float32)
    for train_index, eval_index in splitter.split(dummy_x, y, groups=groups):
        eval_domains = set(y[eval_index].tolist())
        train_domains = set(y[train_index].tolist())
        if eval_domains != {TRAIN_DOMAIN_LABEL, HOLDOUT_DOMAIN_LABEL}:
            raise ValueError("A CV evaluation fold does not contain both domains.")
        if train_domains != {TRAIN_DOMAIN_LABEL, HOLDOUT_DOMAIN_LABEL}:
            raise ValueError("A CV training fold does not contain both domains.")
        folds.append((train_index, eval_index))
    return folds


def new_classifier(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _safe_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(float), 1e-7, 1.0 - 1e-7)


def _fold_metric_row(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    fold: int,
    train_size: int,
    eval_size: int,
) -> dict[str, Any]:
    y_pred = (y_score >= 0.5).astype(int)
    row: dict[str, Any] = {
        "fold": int(fold),
        "train_rows": int(train_size),
        "eval_rows": int(eval_size),
        "eval_holdout_ratio": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, _safe_probability(y_score))),
        "log_loss": float(log_loss(y_true, _safe_probability(y_score), labels=[0, 1])),
        "mean_p_holdout": float(np.mean(y_score)),
    }
    train_mask = y_true == TRAIN_DOMAIN_LABEL
    holdout_mask = y_true == HOLDOUT_DOMAIN_LABEL
    row["mean_p_for_train_rows"] = float(np.mean(y_score[train_mask]))
    row["mean_p_for_holdout_rows"] = float(np.mean(y_score[holdout_mask]))
    return row


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_folds": int(len(fold_metrics))}
    for metric in METRIC_NAMES:
        values = pd.to_numeric(fold_metrics[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            mean = std = ci95 = float("nan")
        else:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci95 = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std
        summary[f"{metric}_ci95_half_width"] = ci95
    summary["roc_auc_interpretation"] = interpret_auc(summary["roc_auc_mean"])
    return summary


def interpret_auc(auc: float) -> str:
    if not np.isfinite(auc):
        return "unavailable"
    if auc < 0.6:
        return "negligible/no clear separability"
    if auc < 0.7:
        return "weak shift"
    if auc < 0.75:
        return "moderate shift"
    if auc < 0.85:
        return "notable shift"
    return "severe shift"


def _feature_importance_from_models(
    coefficients: list[np.ndarray],
    feature_names: list[str],
    *,
    mode: str,
    feature_set: str,
) -> pd.DataFrame:
    if not coefficients:
        return pd.DataFrame(
            columns=[
                "mode",
                "feature_set",
                "rank",
                "feature",
                "mean_abs_coefficient",
                "max_abs_coefficient",
            ]
        )
    stacked = np.vstack(coefficients)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_coefficient": np.mean(np.abs(stacked), axis=0),
            "max_abs_coefficient": np.max(np.abs(stacked), axis=0),
        }
    ).sort_values("mean_abs_coefficient", ascending=False, kind="stable")
    importance.insert(0, "rank", np.arange(1, len(importance) + 1, dtype=int))
    importance.insert(0, "feature_set", feature_set)
    importance.insert(0, "mode", mode)
    return importance.reset_index(drop=True)


def per_class_adversarial_summary(
    predictions: pd.DataFrame,
    *,
    mode: str,
    feature_set: str,
    label_col: str = "result",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for class_id, part in predictions.groupby(label_col, sort=True):
        y_true = part["domain_label"].astype(int).to_numpy()
        y_score = part["p_shadow_holdout"].astype(float).to_numpy()
        train_mask = y_true == TRAIN_DOMAIN_LABEL
        holdout_mask = y_true == HOLDOUT_DOMAIN_LABEL
        if len(set(y_true.tolist())) == 2:
            class_auc = float(roc_auc_score(y_true, y_score))
        else:
            class_auc = float("nan")
        rows.append(
            {
                "mode": mode,
                "feature_set": feature_set,
                "class_id": int(class_id),
                "label": str(part["label"].dropna().astype(str).iloc[0])
                if "label" in part.columns and part["label"].notna().any()
                else "",
                "train_count": int(train_mask.sum()),
                "holdout_count": int(holdout_mask.sum()),
                "roc_auc": class_auc,
                "mean_p_for_train_rows": float(np.mean(y_score[train_mask]))
                if train_mask.any()
                else float("nan"),
                "mean_p_for_holdout_rows": float(np.mean(y_score[holdout_mask]))
                if holdout_mask.any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def top_adversarial_examples(
    predictions: pd.DataFrame,
    *,
    mode: str,
    feature_set: str,
    n: int = 20,
) -> pd.DataFrame:
    base_columns = [
        "image_id_ext",
        "item_id",
        "result",
        "label",
        "domain",
        "p_shadow_holdout",
        "fold",
        "local_path",
    ]
    existing = [column for column in base_columns if column in predictions.columns]
    parts: list[pd.DataFrame] = []

    specs = [
        (
            "holdout_like_train_examples",
            predictions.loc[predictions["domain_label"] == TRAIN_DOMAIN_LABEL].sort_values(
                "p_shadow_holdout", ascending=False, kind="stable"
            ),
        ),
        (
            "train_like_holdout_examples",
            predictions.loc[predictions["domain_label"] == HOLDOUT_DOMAIN_LABEL].sort_values(
                "p_shadow_holdout", ascending=True, kind="stable"
            ),
        ),
        (
            "confidently_separated_holdout_examples",
            predictions.loc[predictions["domain_label"] == HOLDOUT_DOMAIN_LABEL].sort_values(
                "p_shadow_holdout", ascending=False, kind="stable"
            ),
        ),
    ]
    for example_type, part in specs:
        sample = part.head(n)[existing].copy()
        sample.insert(0, "rank", np.arange(1, len(sample) + 1, dtype=int))
        sample.insert(0, "example_type", example_type)
        sample.insert(0, "feature_set", feature_set)
        sample.insert(0, "mode", mode)
        parts.append(sample)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def run_adversarial_cv(
    df: pd.DataFrame,
    X: np.ndarray,
    feature_names: list[str],
    *,
    mode: str,
    feature_set: str,
    n_splits: int,
    seed: int,
    domain_col: str = "domain_label",
    group_col: str = "item_id",
) -> CVResult:
    if len(df) != X.shape[0]:
        raise ValueError(f"Feature row count mismatch: df={len(df)} X={X.shape[0]}")
    if X.ndim != 2:
        raise ValueError("X must be a 2D feature matrix.")
    if X.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match X columns.")

    folds = make_group_stratified_folds(
        df,
        n_splits=n_splits,
        seed=seed,
        domain_col=domain_col,
        group_col=group_col,
    )
    y = df[domain_col].astype(int).to_numpy()

    fold_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    coefficients: list[np.ndarray] = []
    prediction_columns = [
        "row_id",
        "image_id_ext",
        "item_id",
        "result",
        "label",
        "domain",
        "domain_label",
        "local_path",
    ]
    prediction_columns = [column for column in prediction_columns if column in df.columns]

    for fold, (train_index, eval_index) in enumerate(folds):
        classifier = new_classifier(seed + fold)
        classifier.fit(X[train_index], y[train_index])
        y_score = classifier.predict_proba(X[eval_index])[:, 1]
        fold_rows.append(
            _fold_metric_row(
                y_true=y[eval_index],
                y_score=y_score,
                fold=fold,
                train_size=len(train_index),
                eval_size=len(eval_index),
            )
        )

        clf = classifier.named_steps["classifier"]
        coefficients.append(clf.coef_[0].astype(float))

        pred = df.iloc[eval_index][prediction_columns].copy()
        pred["fold"] = fold
        pred["p_shadow_holdout"] = y_score
        pred["predicted_domain_label"] = (y_score >= 0.5).astype(int)
        pred["mode"] = mode
        pred["feature_set"] = feature_set
        prediction_parts.append(pred)

    fold_metrics = pd.DataFrame(fold_rows)
    summary = summarize_fold_metrics(fold_metrics)
    summary.update(
        {
            "mode": mode,
            "feature_set": feature_set,
            "rows": int(len(df)),
            "features": int(X.shape[1]),
            "n_train_domain_rows": int((y == TRAIN_DOMAIN_LABEL).sum()),
            "n_holdout_domain_rows": int((y == HOLDOUT_DOMAIN_LABEL).sum()),
        }
    )
    predictions = pd.concat(prediction_parts, ignore_index=True, sort=False)
    feature_importance = _feature_importance_from_models(
        coefficients,
        feature_names,
        mode=mode,
        feature_set=feature_set,
    )
    class_summary = per_class_adversarial_summary(
        predictions,
        mode=mode,
        feature_set=feature_set,
    )
    examples = top_adversarial_examples(
        predictions,
        mode=mode,
        feature_set=feature_set,
    )
    return CVResult(
        mode=mode,
        feature_set=feature_set,
        fold_metrics=fold_metrics,
        summary=summary,
        predictions=predictions,
        feature_importance=feature_importance,
        per_class_summary=class_summary,
        top_examples=examples,
    )
