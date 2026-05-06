from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TRAIN_DOMAIN = "train_dev_pool"
HOLDOUT_DOMAIN = "shadow_holdout"
TRAIN_DOMAIN_LABEL = 0
HOLDOUT_DOMAIN_LABEL = 1
DOMAIN_LABELS = {TRAIN_DOMAIN: TRAIN_DOMAIN_LABEL, HOLDOUT_DOMAIN: HOLDOUT_DOMAIN_LABEL}

REQUIRED_RECORD_COLUMNS = {
    "item_id",
    "image_id_ext",
    "result",
    "label",
    "ratio",
    "local_path",
    "width",
    "height",
    "status",
    "content_hash",
}

FORBIDDEN_FEATURE_COLUMNS = {
    "item_id",
    "image_id_ext",
    "image",
    "url",
    "content_hash",
    "hash",
    "hash_sha256",
    "checksum",
    "local_path",
    "path",
    "source_dataset",
    "domain",
    "domain_label",
    "result",
    "target",
    "class_id",
    "label",
    "label_name",
    "pred",
    "predicted",
}


@dataclass(frozen=True)
class DomainFrames:
    train: pd.DataFrame
    holdout: pd.DataFrame
    combined: pd.DataFrame


def _python_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _require_columns(df: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def _records_to_frame(records: list[dict[str, Any]], source: str) -> pd.DataFrame:
    df = pd.DataFrame(records)
    _require_columns(df, REQUIRED_RECORD_COLUMNS, source)
    return df.copy()


def _validate_image_paths(df: pd.DataFrame, source: str) -> None:
    missing_paths: list[str] = []
    for value in df["local_path"].tolist():
        if value is None or pd.isna(value) or not Path(str(value)).exists():
            missing_paths.append(str(value))
            if len(missing_paths) >= 10:
                break
    if missing_paths:
        raise FileNotFoundError(
            f"{source} has missing local_path files; first missing values: {missing_paths}"
        )


def _normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["image_id_ext"] = result["image_id_ext"].astype(str)
    result["item_id"] = result["item_id"].astype(str)
    result["result"] = pd.to_numeric(result["result"], errors="raise").astype(int)
    result["ratio"] = pd.to_numeric(result["ratio"], errors="raise")
    result["width"] = pd.to_numeric(result["width"], errors="raise")
    result["height"] = pd.to_numeric(result["height"], errors="raise")

    if ((result["ratio"] <= 0.0) | (result["ratio"] > 1.0)).any():
        raise ValueError("ratio must be inside (0, 1].")
    if ((result["width"] <= 0) | (result["height"] <= 0)).any():
        raise ValueError("width and height must be positive.")
    return result


def load_domain_frames_from_splits(
    splits_json: Path,
    *,
    require_image_paths: bool = True,
) -> DomainFrames:
    """Load train/dev pool and shadow holdout from the fixed split contract."""
    if not splits_json.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_json}")

    payload = json.loads(splits_json.read_text(encoding="utf-8"))
    folds = payload.get("folds")
    shadow = payload.get("shadow_holdout", {}).get("records")
    if not isinstance(folds, list) or shadow is None:
        raise ValueError(f"{splits_json} is not a splits_v1-like JSON file.")

    train_records: list[dict[str, Any]] = []
    for fold in folds:
        records = fold.get("records", [])
        if not isinstance(records, list):
            raise ValueError("Each fold must contain a list under `records`.")
        train_records.extend(records)

    train = _records_to_frame(train_records, "train/dev pool split records")
    holdout = _records_to_frame(shadow, "shadow holdout split records")

    train = _normalize_types(train)
    holdout = _normalize_types(holdout)

    if train["image_id_ext"].duplicated().any():
        dupes = train.loc[train["image_id_ext"].duplicated(), "image_id_ext"].head(10).tolist()
        raise ValueError(f"Duplicate image_id_ext inside train/dev pool split records: {dupes}")
    if holdout["image_id_ext"].duplicated().any():
        dupes = holdout.loc[holdout["image_id_ext"].duplicated(), "image_id_ext"].head(10).tolist()
        raise ValueError(f"Duplicate image_id_ext inside shadow holdout split records: {dupes}")

    if require_image_paths:
        _validate_image_paths(train, "train/dev pool")
        _validate_image_paths(holdout, "shadow holdout")

    train = train.sort_values(["item_id", "image_id_ext"], kind="stable").reset_index(drop=True)
    holdout = holdout.sort_values(["item_id", "image_id_ext"], kind="stable").reset_index(
        drop=True
    )
    combined = assemble_domain_dataset(train, holdout)
    return DomainFrames(train=train, holdout=holdout, combined=combined)


def assemble_domain_dataset(train: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    """Attach binary domain labels without adding target/id fields to classifier features."""
    _require_columns(train, REQUIRED_RECORD_COLUMNS, "train/dev pool")
    _require_columns(holdout, REQUIRED_RECORD_COLUMNS, "shadow holdout")

    left = train.copy()
    left["domain"] = TRAIN_DOMAIN
    right = holdout.copy()
    right["domain"] = HOLDOUT_DOMAIN
    df = pd.concat([left, right], ignore_index=True, sort=False)
    df["domain_label"] = df["domain"].map(DOMAIN_LABELS).astype(int)
    df["row_id"] = np.arange(len(df), dtype=int)
    return df.reset_index(drop=True)


def label_shift_table(
    df: pd.DataFrame,
    *,
    label_col: str = "result",
    label_name_col: str = "label",
    domain_col: str = "domain",
) -> pd.DataFrame:
    """Compare class distribution between train/dev pool and shadow holdout."""
    _require_columns(df, {label_col, label_name_col, domain_col}, "domain dataframe")
    domains = [TRAIN_DOMAIN, HOLDOUT_DOMAIN]
    rows: list[dict[str, Any]] = []
    labels = sorted(df[label_col].dropna().astype(int).unique().tolist())

    for class_id in labels:
        by_class = df.loc[df[label_col].astype(int) == class_id]
        label_name = by_class[label_name_col].dropna().astype(str).iloc[0]
        counts = {
            domain: int(((df[domain_col] == domain) & (df[label_col].astype(int) == class_id)).sum())
            for domain in domains
        }
        totals = {domain: int((df[domain_col] == domain).sum()) for domain in domains}
        train_pct = counts[TRAIN_DOMAIN] / totals[TRAIN_DOMAIN] if totals[TRAIN_DOMAIN] else 0.0
        holdout_pct = (
            counts[HOLDOUT_DOMAIN] / totals[HOLDOUT_DOMAIN] if totals[HOLDOUT_DOMAIN] else 0.0
        )
        rows.append(
            {
                "class_id": int(class_id),
                "label": label_name,
                "train_count": counts[TRAIN_DOMAIN],
                "holdout_count": counts[HOLDOUT_DOMAIN],
                "train_pct": train_pct,
                "holdout_pct": holdout_pct,
                "pct_diff_holdout_minus_train": holdout_pct - train_pct,
                "present_in_train": counts[TRAIN_DOMAIN] > 0,
                "present_in_holdout": counts[HOLDOUT_DOMAIN] > 0,
            }
        )

    return pd.DataFrame(rows).sort_values("class_id", kind="stable").reset_index(drop=True)


def class_balance_sources(
    df: pd.DataFrame,
    *,
    seed: int,
    label_col: str = "result",
    domain_col: str = "domain_label",
    train_label: int = TRAIN_DOMAIN_LABEL,
    holdout_label: int = HOLDOUT_DOMAIN_LABEL,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep shared classes only and sample equal train/holdout counts per class.

    The intended production case has more train rows than holdout rows. If a synthetic or
    future dataset has fewer train rows for a shared class, both sides are sampled to the
    smaller count so the covariate test remains class-balanced.
    """
    _require_columns(df, {label_col, domain_col}, "domain dataframe")
    rng = np.random.default_rng(seed)

    train_classes = set(df.loc[df[domain_col] == train_label, label_col].astype(int).tolist())
    holdout_classes = set(df.loc[df[domain_col] == holdout_label, label_col].astype(int).tolist())
    shared_classes = sorted(train_classes.intersection(holdout_classes))
    excluded_classes = sorted(train_classes.symmetric_difference(holdout_classes))

    selected_parts: list[pd.DataFrame] = []
    per_class_counts: dict[str, dict[str, int]] = {}
    downsampled_holdout_classes: list[int] = []

    for class_id in shared_classes:
        train_part = df.loc[(df[domain_col] == train_label) & (df[label_col].astype(int) == class_id)]
        holdout_part = df.loc[
            (df[domain_col] == holdout_label) & (df[label_col].astype(int) == class_id)
        ]
        target_count = min(len(train_part), len(holdout_part))
        if target_count <= 0:
            continue
        if len(train_part) > target_count:
            train_index = rng.choice(train_part.index.to_numpy(), size=target_count, replace=False)
            train_part = train_part.loc[np.sort(train_index)]
        if len(holdout_part) > target_count:
            holdout_index = rng.choice(
                holdout_part.index.to_numpy(), size=target_count, replace=False
            )
            holdout_part = holdout_part.loc[np.sort(holdout_index)]
            downsampled_holdout_classes.append(int(class_id))
        selected_parts.extend([train_part, holdout_part])
        per_class_counts[str(class_id)] = {
            "train_count": int(target_count),
            "holdout_count": int(target_count),
        }

    if not selected_parts:
        raise ValueError("No shared classes with positive counts for class-balanced mode.")

    balanced = pd.concat(selected_parts, ignore_index=False).sort_index(kind="stable")
    balanced = balanced.reset_index(drop=True)
    audit = {
        "shared_classes": shared_classes,
        "excluded_classes": excluded_classes,
        "missing_from_holdout_classes": sorted(train_classes.difference(holdout_classes)),
        "missing_from_train_classes": sorted(holdout_classes.difference(train_classes)),
        "downsampled_holdout_classes": sorted(downsampled_holdout_classes),
        "per_class_counts": per_class_counts,
        "rows_before": int(len(df)),
        "rows_after": int(len(balanced)),
    }
    return balanced, audit


def compact_records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    existing = [column for column in columns if column in df.columns]
    return [
        {key: _python_value(value) for key, value in row.items()}
        for row in df[existing].to_dict(orient="records")
    ]
