from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_SORT_COLUMNS = ("item_id", "image_id_ext", "image")


def require_columns(df: pd.DataFrame, required: set[str], source_path: Path) -> None:
    """Проверяет наличие обязательных колонок в таблице.

    Пояснение: останавливает запуск, если схема входных данных не совпадает с ожиданием.
    """
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{source_path} missing required columns: {missing}")


def load_labeled_csv(
    path: Path,
    required_columns: set[str],
    source_dataset: str,
    split_role: str | None = None,
    ratio_column: str | None = None,
) -> pd.DataFrame:
    """Читает размеченный CSV и добавляет служебные поля.

    Пояснение: валидирует схему, помечает источник данных и при необходимости проверяет `ratio`.
    """
    df = pd.read_csv(path)
    require_columns(df, required_columns, path)

    df = df.copy()
    df["source_dataset"] = source_dataset
    if split_role is not None:
        df["split_role"] = split_role

    if ratio_column is not None:
        df[ratio_column] = pd.to_numeric(df[ratio_column], errors="raise")
        if ((df[ratio_column] <= 0.0) | (df[ratio_column] > 1.0)).any():
            raise ValueError(f"{path} has {ratio_column} values outside (0, 1].")

    sort_columns = [column for column in DEFAULT_SORT_COLUMNS if column in df.columns]
    if sort_columns:
        return df.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return df.reset_index(drop=True)
