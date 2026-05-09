from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

IMAGE_ID_COLUMN = "image_id_ext"
LABEL_COLUMN = "result"
PREDICTION_COLUMN = "Predicted"
SUBMISSION_COLUMNS = [IMAGE_ID_COLUMN, PREDICTION_COLUMN]
DEFAULT_IMAGE_SUFFIX = ".jpg"


@dataclass(frozen=True)
class ClassSchema:
    valid_class_ids: list[int]
    id_to_label: dict[int, str]

    @property
    def num_classes(self) -> int:
        return len(self.valid_class_ids)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ClassSchema:
        path = Path(path)
        with path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a mapping")

        prediction = payload.get("prediction", {}) if isinstance(payload.get("prediction"), dict) else {}
        if "valid_class_ids" in prediction:
            valid_class_ids = sorted(int(value) for value in prediction["valid_class_ids"])
        elif "id_to_label" in payload:
            valid_class_ids = sorted(int(value) for value in payload["id_to_label"].keys())
        elif "num_classes" in payload:
            valid_class_ids = list(range(int(payload["num_classes"])))
        else:
            raise ValueError(f"{path} does not define valid class ids")

        id_to_label = {
            int(key): str(value) for key, value in (payload.get("id_to_label") or {}).items()
        }
        for class_id in valid_class_ids:
            id_to_label.setdefault(class_id, str(class_id))
        return cls(valid_class_ids=valid_class_ids, id_to_label=id_to_label)


def normalize_image_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def image_id_with_extension(value: Any, suffix: str = DEFAULT_IMAGE_SUFFIX) -> str:
    text = normalize_image_id(value)
    if not text:
        return ""
    return text if Path(text).suffix else f"{text}{suffix}"


def require_columns(df: pd.DataFrame, required: Iterable[str], source: str | Path) -> None:
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def validate_ratio_column(df: pd.DataFrame, column: str, source: str | Path) -> pd.Series:
    ratio = pd.to_numeric(df[column], errors="raise")
    if ((ratio <= 0.0) | (ratio > 1.0)).any():
        raise ValueError(f"{source} has {column} values outside (0, 1].")
    return ratio


def validate_class_ids(values: Iterable[Any], valid_class_ids: Iterable[int], source: str) -> None:
    valid_set = set(int(value) for value in valid_class_ids)
    observed = {int(value) for value in pd.Series(list(values)).dropna().astype(int).tolist()}
    invalid = sorted(observed.difference(valid_set))
    if invalid:
        raise ValueError(f"{source} contains classes outside schema: {invalid}")


def records_to_frame(records: list[dict[str, Any]], image_col: str = IMAGE_ID_COLUMN) -> pd.DataFrame:
    frame = pd.DataFrame(records).copy()
    if image_col in frame.columns:
        frame[image_col] = frame[image_col].map(image_id_with_extension)
    return frame.reset_index(drop=True)


def validate_split_contract(splits: dict[str, Any], expected_version: str | None = "splits_v1") -> None:
    if expected_version is not None and splits.get("version") != expected_version:
        raise ValueError(f"Unsupported split version: {splits.get('version')!r}")
    if not isinstance(splits.get("folds"), list) or not splits["folds"]:
        raise ValueError("Splits must contain a non-empty 'folds' list")
    if not isinstance(splits.get("shadow_holdout"), dict):
        raise ValueError("Splits must contain 'shadow_holdout'")

    fold_ids = [int(fold_payload["fold"]) for fold_payload in splits["folds"]]
    if len(fold_ids) != len(set(fold_ids)):
        raise ValueError(f"Duplicate fold ids in split file: {fold_ids}")

    train_records = [row for fold in splits["folds"] for row in fold.get("records", [])]
    shadow_records = splits["shadow_holdout"].get("records", [])
    train_ids = {image_id_with_extension(row.get(IMAGE_ID_COLUMN)) for row in train_records}
    shadow_ids = {image_id_with_extension(row.get(IMAGE_ID_COLUMN)) for row in shadow_records}
    overlap = sorted(train_ids.intersection(shadow_ids))
    if overlap:
        raise ValueError(f"Shadow holdout leaks into train folds by image_id_ext: {overlap[:10]}")

    summary = splits.get("summary") or {}
    expected_train_rows = summary.get("train_pool_rows_after_filters")
    if expected_train_rows is not None and int(expected_train_rows) != len(train_records):
        raise ValueError(
            "Split summary drift: "
            f"train_pool_rows_after_filters={expected_train_rows} records={len(train_records)}"
        )
    expected_shadow_rows = summary.get("shadow_holdout_rows_after_filters")
    if expected_shadow_rows is not None and int(expected_shadow_rows) != len(shadow_records):
        raise ValueError(
            "Split summary drift: "
            f"shadow_holdout_rows_after_filters={expected_shadow_rows} records={len(shadow_records)}"
        )


def load_splits(path: str | Path, expected_version: str | None = "splits_v1") -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        splits = json.load(file)
    if not isinstance(splits, dict):
        raise ValueError(f"{path} must contain a split mapping")
    validate_split_contract(splits, expected_version=expected_version)
    return splits


def build_fold_frames(splits: dict[str, Any], fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = splits["folds"]
    valid_records: list[dict[str, Any]] | None = None
    train_records: list[dict[str, Any]] = []
    for fold_payload in folds:
        if int(fold_payload["fold"]) == int(fold):
            valid_records = fold_payload.get("records", [])
        else:
            train_records.extend(fold_payload.get("records", []))
    if valid_records is None:
        raise ValueError(f"Fold {fold} not found in split file")
    return records_to_frame(train_records), records_to_frame(valid_records)


def class_names_from_splits(splits: dict[str, Any], num_classes: int) -> list[str]:
    names = [str(index) for index in range(num_classes)]
    for fold_payload in splits["folds"]:
        for row in fold_payload.get("records", []):
            if LABEL_COLUMN in row and "label" in row:
                names[int(row[LABEL_COLUMN])] = str(row["label"])
    for row in splits.get("shadow_holdout", {}).get("records", []):
        if LABEL_COLUMN in row and "label" in row:
            names[int(row[LABEL_COLUMN])] = str(row["label"])
    return names
