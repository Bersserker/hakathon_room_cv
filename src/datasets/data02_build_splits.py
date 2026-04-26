#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.labeled_data import load_labeled_csv, require_columns


REQUIRED_CSV_COLUMNS = {"item_id", "image", "image_id_ext", "result", "label"}
DEFAULT_TRAIN_CSV = Path("data/raw/train_df.csv")
DEFAULT_VAL_CSV = Path("data/raw/val_df.csv")
DEFAULT_MANIFEST = Path("data/processed/data_manifest.parquet")
DEFAULT_OUTPUT_JSON = Path("data/splits/splits_v1.json")
DEFAULT_REPORT_MD = Path("reports/leakage_report.md")
SEED = 26042026


def parse_args() -> argparse.Namespace:
    """Собирает аргументы командной строки для запуска DATA-02 pipeline.

    Пояснение: читает параметры запуска скрипта, если их передали вручную.
    """
    parser = argparse.ArgumentParser(
        description="Build DATA-02 leakage report and group splits."
    )
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--n-folds", type=int, default=5)

    return parser.parse_args()


def python_value(value: Any) -> Any:
    """Приводит pandas/numpy значения к обычным Python-типам и заменяет NaN на None.

    Пояснение: делает значения удобными для JSON и отчётов.
    """
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def iso_utc_from_mtime(paths: list[Path]) -> str:
    """Возвращает UTC timestamp по самому свежему времени изменения входных файлов.

    Пояснение: ставит в артефакты метку, когда входные данные последний раз менялись.
    """
    latest_mtime = max(path.stat().st_mtime for path in paths if path.exists())
    return (
        datetime.fromtimestamp(latest_mtime, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def add_manifest_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет пустые manifest-поля, если manifest пока отсутствует.

    Пояснение: сохраняет одинаковую схему данных даже без parquet-файла.
    """
    df = df.copy()
    for column in ["local_path", "width", "height", "status", "content_hash"]:
        df[column] = None
    return df


def load_manifest(path: Path) -> tuple[pd.DataFrame, str | None, int]:
    """Читает manifest, нормализует его колонки и убирает дубли по image_id_ext.

    Пояснение: подготавливает parquet так, чтобы его можно было безопасно смержить с CSV.
    """
    manifest = pd.read_parquet(path)
    require_columns(manifest, {"image_id_ext", "status"}, path)

    hash_source_column = None
    if "hash" in manifest.columns:
        hash_source_column = "hash"
    elif "checksum" in manifest.columns:
        hash_source_column = "checksum"

    manifest = manifest.copy()
    manifest["status"] = (
        manifest["status"].astype("string").str.lower().fillna("missing_in_manifest")
    )
    if "local_path" not in manifest.columns:
        manifest["local_path"] = None
    if "width" not in manifest.columns:
        manifest["width"] = None
    if "height" not in manifest.columns:
        manifest["height"] = None
    if hash_source_column is None:
        manifest["content_hash"] = None
    else:
        manifest["content_hash"] = manifest[hash_source_column]

    manifest = manifest[
        ["image_id_ext", "local_path", "width", "height", "status", "content_hash"]
    ]
    manifest = manifest.sort_values(["image_id_ext"], kind="stable").reset_index(
        drop=True
    )
    duplicate_rows = int(
        manifest.duplicated(subset=["image_id_ext"], keep="first").sum()
    )
    manifest = manifest.drop_duplicates(
        subset=["image_id_ext"], keep="first"
    ).reset_index(drop=True)
    return manifest, hash_source_column, duplicate_rows


def merge_manifest(df: pd.DataFrame, manifest: pd.DataFrame | None) -> pd.DataFrame:
    """Мержит manifest в основную таблицу по image_id_ext.

    Пояснение: приклеивает к строкам картинки их статус, размер, путь и хэш.
    """
    if manifest is None:
        return add_manifest_placeholders(df)

    merged = df.merge(manifest, on="image_id_ext", how="left")
    merged["status"] = merged["status"].fillna("missing_in_manifest")
    return merged


def drop_image_id_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Удаляет повторные строки с одинаковым image_id_ext, сохраняя первую.

    Пояснение: выбрасывает дубли одной и той же картинки до построения split.
    """
    duplicate_mask = df.duplicated(subset=["image_id_ext"], keep="first")
    return df.loc[~duplicate_mask].reset_index(drop=True), int(duplicate_mask.sum())


def filter_usable_rows(
    df: pd.DataFrame, manifest_used: bool
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Фильтрует строки для работы pipeline с учётом manifest-статусов.

    Пояснение: если manifest есть, оставляет только картинки со статусом `ok`.
    """
    if not manifest_used:
        return df.copy(), {}

    status_counts = {
        str(key): int(value)
        for key, value in df["status"].value_counts(dropna=False).sort_index().items()
    }
    usable = df.loc[df["status"] == "ok"].copy().reset_index(drop=True)
    return usable, status_counts


def overlap_summary(
    left: pd.DataFrame, right: pd.DataFrame, column: str
) -> dict[str, Any]:
    """Считает пересечение двух таблиц по одному ключу.

    Пояснение: отвечает на вопрос, есть ли одинаковые item_id, image_id_ext или URL в train и val.
    """
    left_values = {python_value(value) for value in left[column].dropna().tolist()}
    right_values = {python_value(value) for value in right[column].dropna().tolist()}
    intersection = sorted(left_values.intersection(right_values))
    return {
        "left_unique": int(len(left_values)),
        "right_unique": int(len(right_values)),
        "intersection_count": int(len(intersection)),
        "sample": intersection[:10],
    }


def hash_duplicate_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Считает дубли по content_hash внутри одной таблицы.

    Пояснение: показывает, сколько картинок имеют одинаковое содержимое по хэшу.
    """
    hash_df = df.loc[df["content_hash"].notna()].copy()
    if hash_df.empty:
        return {
            "rows_with_hash": 0,
            "unique_hashes": 0,
            "duplicate_hash_groups": 0,
            "rows_inside_duplicate_hash_groups": 0,
        }

    counts = hash_df.groupby("content_hash").size()
    duplicate_groups = counts.loc[counts > 1]
    return {
        "rows_with_hash": int(len(hash_df)),
        "unique_hashes": int(hash_df["content_hash"].nunique()),
        "duplicate_hash_groups": int(len(duplicate_groups)),
        "rows_inside_duplicate_hash_groups": int(duplicate_groups.sum()),
    }


def cross_hash_overlap_summary(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> dict[str, Any]:
    """Считает пересечение train и shadow holdout по content_hash.

    Пояснение: ищет одинаковые картинки между train и val, даже если у них разные image_id_ext.
    """
    train_hashes = {
        python_value(value) for value in train_df["content_hash"].dropna().tolist()
    }
    val_hashes = {
        python_value(value) for value in val_df["content_hash"].dropna().tolist()
    }
    overlap = sorted(train_hashes.intersection(val_hashes))
    return {
        "train_unique_hashes": int(len(train_hashes)),
        "shadow_holdout_unique_hashes": int(len(val_hashes)),
        "intersection_count": int(len(overlap)),
        "sample": overlap[:10],
    }


def cross_fold_hash_leakage(assignments: pd.DataFrame) -> dict[str, Any]:
    """Проверяет, что одинаковые хэши не разъехались по разным folds.

    Пояснение: ищет hash-based leakage внутри train k-fold split.
    """
    hash_df = assignments.loc[assignments["content_hash"].notna()].copy()
    if hash_df.empty:
        return {"hash_groups_spanning_multiple_folds": 0, "sample": []}

    folds_per_hash = hash_df.groupby("content_hash")["fold"].nunique()
    leaking_hashes = sorted(folds_per_hash.loc[folds_per_hash > 1].index.tolist())
    return {
        "hash_groups_spanning_multiple_folds": int(len(leaking_hashes)),
        "sample": leaking_hashes[:10],
    }


def build_fold_assignments(train_df: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Строит fold-назначения через StratifiedGroupKFold по item_id и result.

    Пояснение: делит train на folds так, чтобы один item_id не попадал в разные части.
    """
    assignments = (
        train_df.sort_values(["item_id", "image_id_ext"], kind="stable")
        .reset_index(drop=True)
        .copy()
    )
    assignments["fold"] = -1

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for fold_index, (_, valid_index) in enumerate(
        splitter.split(
            assignments[["image_id_ext"]],
            assignments["result"],
            groups=assignments["item_id"],
        )
    ):
        assignments.loc[valid_index, "fold"] = fold_index

    if (assignments["fold"] < 0).any():
        raise RuntimeError("Some train rows were not assigned to a fold.")

    item_id_fold_counts = assignments.groupby("item_id")["fold"].nunique()
    if not bool((item_id_fold_counts == 1).all()):
        raise RuntimeError(
            "Group leakage detected: some item_id values span multiple folds."
        )

    return assignments


def class_distribution_table(assignments: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Строит таблицу распределения классов по folds.

    Пояснение: показывает, сколько примеров каждого класса попало в каждый fold.
    """
    counts = (
        assignments.groupby(["result", "fold"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    counts = counts.reindex(columns=list(range(n_folds)), fill_value=0)
    label_map = assignments.groupby("result")["label"].first().sort_index()

    table = counts.copy()
    table.insert(0, "label", label_map)
    table["total"] = counts.sum(axis=1)
    table.index.name = "result"
    return table.reset_index()


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Преобразует заголовки и строки в markdown-таблицу.

    Пояснение: собирает текст таблицы для итогового отчёта.
    """
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def rows_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Преобразует DataFrame в список стабильных JSON-записей нужной схемы.

    Пояснение: готовит строки fold или holdout к записи в `splits_v1.json`.
    """
    record_columns = [
        "image_id_ext",
        "item_id",
        "result",
        "label",
        "image",
        "source_dataset",
        "local_path",
        "width",
        "height",
        "status",
        "content_hash",
    ]

    records = []
    for row in df[record_columns].to_dict(orient="records"):
        records.append({key: python_value(value) for key, value in row.items()})
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Записывает итоговый JSON-артефакт на диск.

    Пояснение: сохраняет split в файл для downstream-потребителей.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def write_report(
    path: Path,
    payload: dict[str, Any],
    class_distribution: pd.DataFrame,
    rerun_command: str,
) -> None:
    """Собирает и записывает markdown-отчёт по leakage, дублям и splits.

    Пояснение: делает человекочитаемый итоговый отчёт с реальными числами.
    """
    overlap_rows = []
    for key, summary in payload["summary"]["cross_dataset_overlap"].items():
        overlap_rows.append(
            [
                key,
                summary["left_unique"],
                summary["right_unique"],
                summary["intersection_count"],
                (
                    ", ".join(str(value) for value in summary["sample"])
                    if summary["sample"]
                    else "-"
                ),
            ]
        )

    duplicate_rows = [
        [
            "train_df",
            payload["summary"]["duplicates_removed"]["train_df_image_id_ext_rows"],
            payload["summary"]["in_dataset_duplicate_rows"]["train_df_image_url_rows"],
            payload["summary"]["train_pool_rows_after_filters"],
        ],
        [
            "val_df",
            payload["summary"]["duplicates_removed"]["val_df_image_id_ext_rows"],
            payload["summary"]["in_dataset_duplicate_rows"]["val_df_image_url_rows"],
            payload["summary"]["shadow_holdout_rows_after_filters"],
        ],
    ]

    fold_rows = []
    for fold in payload["folds"]:
        fold_rows.append(
            [
                fold["fold"],
                fold["validation_rows"],
                fold["validation_item_groups"],
                fold["training_rows"],
                fold["training_item_groups"],
            ]
        )

    class_rows = []
    for row in class_distribution.to_dict(orient="records"):
        class_rows.append(
            [
                row["result"],
                row["label"],
                *[row[fold] for fold in range(payload["policy"]["n_folds"])],
                row["total"],
            ]
        )

    pending_checks = payload["pending_checks"] or ["none"]

    manifest_section_lines = [
        "## Manifest integration",
        f"- used_manifest: `{payload['inputs']['manifest']['used']}`",
        f"- manifest_path: `{payload['inputs']['manifest']['path']}`",
        f"- manifest_exists: `{payload['inputs']['manifest']['exists']}`",
        f"- manifest_hash_source_column: `{payload['inputs']['manifest']['hash_source_column']}`",
    ]

    if payload["inputs"]["manifest"]["used"]:
        manifest_section_lines.extend(
            [
                f"- manifest_duplicate_image_id_ext_rows: `{payload['summary']['duplicates_removed']['manifest_image_id_ext_rows']}`",
                f"- train_status_counts: `{payload['summary']['status_counts']['train_df']}`",
                f"- val_status_counts: `{payload['summary']['status_counts']['val_df']}`",
                f"- hash_overlap_train_vs_shadow_holdout: `{payload['summary']['hash_checks']['train_vs_shadow_holdout']['intersection_count']}`",
                f"- hash_overlap_across_folds: `{payload['summary']['hash_checks']['across_folds']['hash_groups_spanning_multiple_folds']}`",
            ]
        )
    else:
        manifest_section_lines.append(
            "- hash/status checks: `pending until data/processed/data_manifest.parquet exists`"
        )

    content = "\n".join(
        [
            "# DATA-02 Leakage Report",
            "",
            "## Inputs and policy",
            f"- version: `{payload['version']}`",
            f"- timestamp_utc: `{payload['timestamp_utc']}`",
            f"- train_csv: `{payload['inputs']['train_csv']}`",
            f"- val_csv: `{payload['inputs']['val_csv']}`",
            f"- n_folds: `{payload['policy']['n_folds']}`",
            f"- group_key: `{payload['policy']['group_key']}`",
            f"- splitter: `{payload['policy']['splitter']}`",
            f"- duplicate_policy_image_id_ext: `{payload['policy']['duplicate_policy']['image_id_ext']}`",
            f"- duplicate_policy_hash: `{payload['policy']['duplicate_policy']['content_hash']}`",
            f"- val_df_status: `{payload['policy']['shadow_holdout']['status']}`",
            f"- val_df_reason: {payload['policy']['shadow_holdout']['reason']}",
            "",
            "## Train/val overlap checks",
            markdown_table(
                ["key", "train_unique", "val_unique", "intersection_count", "sample"],
                overlap_rows,
            ),
            "",
            "## Duplicate checks",
            markdown_table(
                [
                    "dataset",
                    "removed_image_id_ext_rows",
                    "duplicate_image_url_rows",
                    "rows_after_filters",
                ],
                duplicate_rows,
            ),
            "",
            *manifest_section_lines,
            "",
            "## Shadow holdout",
            f"- `val_df` fixed as `{payload['policy']['shadow_holdout']['status']}`.",
            f"- rows_in_shadow_holdout_after_filters: `{payload['summary']['shadow_holdout_rows_after_filters']}`",
            f"- item_groups_in_shadow_holdout_after_filters: `{payload['summary']['shadow_holdout_item_groups_after_filters']}`",
            "",
            "## Fold summary",
            markdown_table(
                [
                    "fold",
                    "validation_rows",
                    "validation_item_groups",
                    "training_rows",
                    "training_item_groups",
                ],
                fold_rows,
            ),
            "",
            "## Class distribution by fold",
            markdown_table(
                [
                    "result",
                    "label",
                    *[f"fold_{fold}" for fold in range(payload["policy"]["n_folds"])],
                    "total",
                ],
                class_rows,
            ),
            "",
            "## Pending checks",
            *[f"- {item}" for item in pending_checks],
            "",
            "## Re-run",
            "```bash",
            rerun_command,
            "```",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    """Запускает полный DATA-02 pipeline от чтения данных до записи артефактов.

    Пояснение: это главная функция, которая делает всю работу скрипта по шагам.
    """
    args = parse_args()

    train_df_raw = load_labeled_csv(
        args.train_csv,
        required_columns=REQUIRED_CSV_COLUMNS,
        source_dataset="train_df",
    )
    val_df_raw = load_labeled_csv(
        args.val_csv,
        required_columns=REQUIRED_CSV_COLUMNS,
        source_dataset="val_df",
    )

    manifest_used = args.manifest.exists()
    manifest_hash_source = None
    manifest_duplicate_rows = 0
    manifest_df = None
    if manifest_used:
        manifest_df, manifest_hash_source, manifest_duplicate_rows = load_manifest(
            args.manifest
        )

    train_df = merge_manifest(train_df_raw, manifest_df)
    val_df = merge_manifest(val_df_raw, manifest_df)

    train_df, train_image_dup_rows = drop_image_id_duplicates(train_df)
    val_df, val_image_dup_rows = drop_image_id_duplicates(val_df)

    train_pool, train_status_counts = filter_usable_rows(train_df, manifest_used)
    shadow_holdout, val_status_counts = filter_usable_rows(val_df, manifest_used)

    if train_pool["item_id"].nunique() < args.n_folds:
        raise ValueError(
            "Not enough unique item_id groups to build requested number of folds."
        )

    assignments = build_fold_assignments(train_pool, n_folds=args.n_folds)
    class_distribution = class_distribution_table(assignments, n_folds=args.n_folds)

    fold_payloads = []
    total_train_rows = int(len(assignments))
    total_train_groups = int(assignments["item_id"].nunique())
    for fold in range(args.n_folds):
        fold_df = (
            assignments.loc[assignments["fold"] == fold].copy().reset_index(drop=True)
        )
        fold_payloads.append(
            {
                "fold": fold,
                "validation_rows": int(len(fold_df)),
                "validation_item_groups": int(fold_df["item_id"].nunique()),
                "training_rows": total_train_rows - int(len(fold_df)),
                "training_item_groups": total_train_groups
                - int(fold_df["item_id"].nunique()),
                "class_distribution": {
                    str(key): int(value)
                    for key, value in fold_df["result"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "records": rows_to_records(fold_df),
            }
        )

    hash_checks: dict[str, Any]
    pending_checks: list[str] = []
    if manifest_used:
        hash_checks = {
            "train_pool": hash_duplicate_summary(train_pool),
            "shadow_holdout": hash_duplicate_summary(shadow_holdout),
            "train_vs_shadow_holdout": cross_hash_overlap_summary(
                train_pool, shadow_holdout
            ),
            "across_folds": cross_fold_hash_leakage(assignments),
        }
        if manifest_hash_source is None:
            pending_checks.append(
                "manifest exists but has neither `hash` nor `checksum`; content-hash leakage checks remain pending"
            )
    else:
        hash_checks = {
            "train_pool": None,
            "shadow_holdout": None,
            "train_vs_shadow_holdout": None,
            "across_folds": None,
        }
        pending_checks.extend(
            [
                "merge `data/processed/data_manifest.parquet` to add `local_path`, `width`, `height`, `status`, `content_hash`",
                "exclude non-`ok` manifest statuses from train pool and shadow holdout",
                "run content-hash leakage checks across train folds and between train pool vs shadow holdout",
            ]
        )

    input_paths = [args.train_csv, args.val_csv]
    if manifest_used:
        input_paths.append(args.manifest)

    payload = {
        "version": "splits_v1",
        "timestamp_utc": iso_utc_from_mtime(input_paths),
        "inputs": {
            "train_csv": str(args.train_csv),
            "val_csv": str(args.val_csv),
            "manifest": {
                "path": str(args.manifest),
                "exists": bool(args.manifest.exists()),
                "used": bool(manifest_used),
                "hash_source_column": manifest_hash_source,
            },
        },
        "policy": {
            "n_folds": args.n_folds,
            "group_key": "item_id",
            "label_key": "result",
            "splitter": "StratifiedGroupKFold",
            # TODO: use global constant
            "seed": SEED,
            "duplicate_policy": {
                "image_id_ext": "drop_duplicates_keep_first_before_split",
                "content_hash": "report_only_when_manifest_available",
            },
            "shadow_holdout": {
                "status": "separate_shadow_holdout",
                "reason": "safe default: keep original val_df outside train k-folds to avoid tuning leakage",
            },
        },
        "summary": {
            "train_df_raw_rows": int(len(train_df_raw)),
            "val_df_raw_rows": int(len(val_df_raw)),
            "train_pool_rows_after_filters": int(len(assignments)),
            "shadow_holdout_rows_after_filters": int(len(shadow_holdout)),
            "train_pool_item_groups_after_filters": int(
                assignments["item_id"].nunique()
            ),
            "shadow_holdout_item_groups_after_filters": int(
                shadow_holdout["item_id"].nunique()
            ),
            "duplicates_removed": {
                "train_df_image_id_ext_rows": int(train_image_dup_rows),
                "val_df_image_id_ext_rows": int(val_image_dup_rows),
                "manifest_image_id_ext_rows": int(manifest_duplicate_rows),
            },
            "in_dataset_duplicate_rows": {
                "train_df_image_url_rows": int(
                    train_df_raw.duplicated(subset=["image"], keep="first").sum()
                ),
                "val_df_image_url_rows": int(
                    val_df_raw.duplicated(subset=["image"], keep="first").sum()
                ),
            },
            "cross_dataset_overlap": {
                "item_id": overlap_summary(train_df_raw, val_df_raw, "item_id"),
                "image_id_ext": overlap_summary(
                    train_df_raw, val_df_raw, "image_id_ext"
                ),
                "image": overlap_summary(train_df_raw, val_df_raw, "image"),
            },
            "status_counts": {
                "train_df": train_status_counts if manifest_used else None,
                "val_df": val_status_counts if manifest_used else None,
            },
            "hash_checks": hash_checks,
        },
        "folds": fold_payloads,
        "shadow_holdout": {
            "status": "separate_shadow_holdout",
            "rows": int(len(shadow_holdout)),
            "item_groups": int(shadow_holdout["item_id"].nunique()),
            "records": rows_to_records(
                shadow_holdout.sort_values(
                    ["item_id", "image_id_ext"], kind="stable"
                ).reset_index(drop=True)
            ),
        },
        "pending_checks": pending_checks,
    }

    rerun_command = (
        "python3 scripts/data02_build_splits.py "
        f"--train-csv {args.train_csv} --val-csv {args.val_csv} --manifest {args.manifest} "
        f"--output-json {args.output_json} --report-md {args.report_md} --n-folds {args.n_folds}"
    )

    write_json(args.output_json, payload)
    write_report(args.report_md, payload, class_distribution, rerun_command)


if __name__ == "__main__":
    main()
