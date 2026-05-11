from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

REQUIRED_PREDICTION_COLUMNS = {"target", "pred"}


def class_ids(num_classes: int) -> list[int]:
    return list(range(int(num_classes)))


def sorted_class_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [column for column in df.columns if column.startswith(prefix)]
    return sorted(cols, key=lambda column: int(column.split("_")[1]))


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows._"
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
        ]
    )


def validate_prediction_frame(
    frame: pd.DataFrame,
    class_ids: list[int],
    *,
    source: str | Path = "prediction frame",
    require_scores: bool = False,
) -> None:
    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} missing prediction columns: {missing}")

    valid_set = set(class_ids)
    for column in ("target", "pred"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            bad = frame.loc[values.isna(), column].head(10).tolist()
            raise ValueError(f"{source} column {column} must be numeric, bad_sample={bad}")
        invalid = sorted(set(values.astype(int)).difference(valid_set))
        if invalid:
            raise ValueError(f"{source} column {column} has classes outside schema: {invalid}")

    expected_logit_cols = [f"logit_{class_id}" for class_id in class_ids]
    expected_prob_cols = [f"prob_{class_id}" for class_id in class_ids]
    has_logits = set(expected_logit_cols).issubset(frame.columns)
    has_probs = set(expected_prob_cols).issubset(frame.columns)
    if require_scores and not (has_logits or has_probs):
        raise ValueError(f"{source} has neither logits nor probabilities for all classes")


def metrics_from_frame(frame: pd.DataFrame, class_ids: list[int]) -> dict[str, Any]:
    validate_prediction_frame(frame, class_ids, require_scores=False)
    labels = frame["target"].to_numpy(dtype=int)
    preds = frame["pred"].to_numpy(dtype=int)
    return {
        "rows": int(len(frame)),
        "macro_f1": float(
            f1_score(labels, preds, average="macro", labels=class_ids, zero_division=0)
        ),
        "accuracy": float(accuracy_score(labels, preds)),
        "per_class_f1": f1_score(labels, preds, average=None, labels=class_ids, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds, labels=class_ids),
    }


def present_label_macro_f1(frame: pd.DataFrame) -> float:
    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"prediction frame missing prediction columns: {missing}")
    present = sorted(int(value) for value in pd.to_numeric(frame["target"]).dropna().unique())
    if not present:
        return 0.0
    return float(
        f1_score(
            pd.to_numeric(frame["target"]).to_numpy(dtype=int),
            pd.to_numeric(frame["pred"]).to_numpy(dtype=int),
            average="macro",
            labels=present,
            zero_division=0,
        )
    )


def per_class_metrics(frame: pd.DataFrame, class_ids: list[int]) -> dict[str, Any]:
    validate_prediction_frame(frame, class_ids, require_scores=False)
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


def scores_and_targets(frame: pd.DataFrame, class_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    validate_prediction_frame(frame, class_ids, require_scores=True)
    logit_cols = [f"logit_{class_id}" for class_id in class_ids]
    prob_cols = [f"prob_{class_id}" for class_id in class_ids]
    if set(logit_cols).issubset(frame.columns):
        scores = frame[logit_cols].to_numpy(dtype=np.float64)
    else:
        probs = frame[prob_cols].to_numpy(dtype=np.float64)
        scores = np.log(np.clip(probs, 1e-12, 1.0))
    return scores, frame["target"].to_numpy(dtype=int)


def macro_f1_from_scores(
    scores: np.ndarray,
    targets: np.ndarray,
    bias: np.ndarray,
    class_ids: list[int],
) -> float:
    preds = (scores + bias.reshape(1, -1)).argmax(axis=1)
    return float(f1_score(targets, preds, average="macro", labels=class_ids, zero_division=0))


def load_prediction_frame(
    path: str | Path, class_ids: list[int], *, require_scores: bool = False
) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_parquet(path)
    validate_prediction_frame(frame, class_ids, source=path, require_scores=require_scores)
    return frame
