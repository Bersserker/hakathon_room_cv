from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.diagnostics.adversarial_data import FORBIDDEN_FEATURE_COLUMNS


METADATA_FEATURES = [
    "ratio",
    "width",
    "height",
    "aspect_ratio",
    "megapixels",
    "is_landscape",
    "is_portrait",
    "is_squareish",
    "is_wide",
    "is_tall",
]


def assert_no_forbidden_features(feature_names: Iterable[str]) -> None:
    forbidden = sorted(set(feature_names).intersection(FORBIDDEN_FEATURE_COLUMNS))
    if forbidden:
        raise ValueError(f"Forbidden leakage features requested: {forbidden}")


def build_metadata_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build safe numeric metadata features for adversarial validation.

    Deliberately excludes ids, URLs, hashes, paths, target labels and class names.
    """
    required = {"ratio", "width", "height"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"metadata dataframe missing required columns: {missing}")

    features = pd.DataFrame(index=df.index)
    features["ratio"] = pd.to_numeric(df["ratio"], errors="raise")
    features["width"] = pd.to_numeric(df["width"], errors="raise")
    features["height"] = pd.to_numeric(df["height"], errors="raise")

    if ((features["width"] <= 0) | (features["height"] <= 0)).any():
        raise ValueError("width and height must be positive for metadata features.")

    features["aspect_ratio"] = features["width"] / features["height"]
    features["megapixels"] = (features["width"] * features["height"]) / 1_000_000.0
    features["is_landscape"] = (features["aspect_ratio"] > 1.05).astype(float)
    features["is_portrait"] = (features["aspect_ratio"] < 0.95).astype(float)
    features["is_squareish"] = (
        (features["aspect_ratio"] >= 0.95) & (features["aspect_ratio"] <= 1.05)
    ).astype(float)
    features["is_wide"] = (features["aspect_ratio"] >= 1.5).astype(float)
    features["is_tall"] = (features["aspect_ratio"] <= (2.0 / 3.0)).astype(float)

    features = features[METADATA_FEATURES].astype(float)
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("metadata features contain non-finite values.")
    assert_no_forbidden_features(features.columns)
    return features


def build_metadata_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    frame = build_metadata_feature_frame(df)
    return frame.to_numpy(dtype=np.float64), frame.columns.tolist()


def embedding_columns(embeddings: pd.DataFrame) -> list[str]:
    columns = [column for column in embeddings.columns if column.startswith("emb_")]
    if not columns:
        raise ValueError("Embedding table has no `emb_*` columns.")
    return sorted(columns)


def build_visual_matrix(df: pd.DataFrame, embeddings: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if "image_id_ext" not in df.columns:
        raise ValueError("visual dataframe missing required column: image_id_ext")
    if "image_id_ext" not in embeddings.columns:
        raise ValueError("embedding table missing required column: image_id_ext")

    emb_cols = embedding_columns(embeddings)
    assert_no_forbidden_features(emb_cols)

    embedding_table = embeddings.copy()
    embedding_table["image_id_ext"] = embedding_table["image_id_ext"].astype(str)
    lookup = embedding_table.drop_duplicates("image_id_ext", keep="last").set_index("image_id_ext")
    image_ids = df["image_id_ext"].astype(str).tolist()
    missing = sorted(set(image_ids).difference(lookup.index.tolist()))
    if missing:
        raise ValueError(f"Missing embeddings for image_id_ext values: {missing[:10]}")

    aligned = lookup.loc[image_ids, emb_cols]
    matrix = aligned.to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("visual embedding matrix contains non-finite values.")
    return matrix, emb_cols
