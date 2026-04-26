#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.utils.labeled_data import load_labeled_csv


REQUIRED_LABELED_COLUMNS = {
    "item_id",
    "image",
    "image_id_ext",
    "result",
    "label",
    "ratio",
}
DEFAULT_TRAIN_CSV = Path("data/raw/train_df.csv")
DEFAULT_VAL_CSV = Path("data/raw/val_df.csv")
DEFAULT_SPLITS_JSON = Path("data/splits/splits_v1.json")
DEFAULT_REPORT_MD = Path("reports/label_quality_ratio.md")
DEFAULT_RATIO_BINS_CSV = Path("tables/ratio_bins.csv")
DEFAULT_LOW_CONSENSUS_CSV = Path("tables/low_consensus_samples.csv")
EPSILON = 1e-9
LOW_CONSENSUS_MAX = (2.0 / 3.0) + EPSILON


def parse_args() -> argparse.Namespace:
    """Собирает аргументы командной строки для DATA-03.

    Пояснение: задаёт пути к входным CSV, split JSON и выходным артефактам.
    """
    parser = argparse.ArgumentParser(
        description="Build DATA-03 label quality / ratio audit artifacts."
    )
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--splits-json", type=Path, default=DEFAULT_SPLITS_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--ratio-bins-csv", type=Path, default=DEFAULT_RATIO_BINS_CSV)
    parser.add_argument(
        "--low-consensus-csv", type=Path, default=DEFAULT_LOW_CONSENSUS_CSV
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    """Создаёт родительскую директорию для файла.

    Пояснение: нужен перед записью отчётов и CSV, если папка ещё не существует.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def format_ratio(value: float) -> str:
    """Форматирует ratio до трёх знаков после запятой.

    Пояснение: делает числа в отчёте компактными и единообразными.
    """
    return f"{float(value):.3f}"


def format_pct(value: float) -> str:
    """Форматирует долю в проценты с одним знаком после запятой.

    Пояснение: используется для долей спорных и low-consensus примеров.
    """
    return f"{float(value) * 100:.1f}%"


def as_int(value: Any) -> int:
    """Приводит значение к int, заменяя NaN на ноль.

    Пояснение: упрощает чтение чисел из JSON-метаданных split.
    """
    if pd.isna(value):
        return 0
    return int(value)


def markdown_table(
    df: pd.DataFrame, formatters: dict[str, Callable[[Any], str]] | None = None
) -> str:
    """Преобразует DataFrame в markdown-таблицу.

    Пояснение: нужен для генерации читаемого `.md` отчёта без внешних библиотек.
    """
    if df.empty:
        return "_Нет строк._"

    formatters = formatters or {}
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows: list[str] = [header, separator]

    for _, row in df.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if column in formatters:
                text = formatters[column](value)
            elif pd.isna(value):
                text = ""
            else:
                text = str(value)
            values.append(text.replace("|", "/").replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows)


def load_split_metadata(splits_json: Path, train_rows: int, val_rows: int) -> dict[str, Any]:
    """Загружает и проверяет метаданные безопасного split.

    Пояснение: убеждается, что используем именно `splits_v1` с отдельным shadow holdout.
    """
    data = json.loads(splits_json.read_text())
    if data.get("version") != "splits_v1":
        raise ValueError(
            f"{splits_json} has unsupported version: {data.get('version')!r}"
        )

    policy = data.get("policy", {})
    summary = data.get("summary", {})
    shadow_policy = policy.get("shadow_holdout", {})
    shadow_holdout = data.get("shadow_holdout", {})

    if policy.get("group_key") != "item_id":
        raise ValueError(f"{splits_json} must use group_key='item_id'.")
    if policy.get("label_key") != "result":
        raise ValueError(f"{splits_json} must use label_key='result'.")
    if policy.get("splitter") != "StratifiedGroupKFold":
        raise ValueError(f"{splits_json} must use StratifiedGroupKFold.")
    if shadow_policy.get("status") != "separate_shadow_holdout":
        raise ValueError(
            f"{splits_json} must fix val_df as separate_shadow_holdout."
        )
    if as_int(summary.get("train_df_raw_rows")) != train_rows:
        raise ValueError(
            f"{splits_json} train_df_raw_rows mismatch: "
            f"{summary.get('train_df_raw_rows')} != {train_rows}"
        )
    if as_int(summary.get("val_df_raw_rows")) != val_rows:
        raise ValueError(
            f"{splits_json} val_df_raw_rows mismatch: "
            f"{summary.get('val_df_raw_rows')} != {val_rows}"
        )
    if as_int(shadow_holdout.get("rows")) != val_rows:
        raise ValueError(
            f"{splits_json} shadow_holdout.rows mismatch: "
            f"{shadow_holdout.get('rows')} != {val_rows}"
        )

    return {
        "version": data["version"],
        "timestamp_utc": data.get("timestamp_utc"),
        "n_folds": as_int(policy.get("n_folds")),
        "seed": as_int(policy.get("seed")),
        "group_key": policy.get("group_key"),
        "label_key": policy.get("label_key"),
        "splitter": policy.get("splitter"),
        "shadow_holdout_status": shadow_policy.get("status"),
        "shadow_holdout_reason": shadow_policy.get("reason", ""),
        "train_pool_rows_after_filters": as_int(
            summary.get("train_pool_rows_after_filters")
        ),
        "shadow_holdout_rows_after_filters": as_int(
            summary.get("shadow_holdout_rows_after_filters")
        ),
        "train_pool_item_groups_after_filters": as_int(
            summary.get("train_pool_item_groups_after_filters")
        ),
        "shadow_holdout_item_groups_after_filters": as_int(
            summary.get("shadow_holdout_item_groups_after_filters")
        ),
    }


def add_consensus_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет флаги согласия разметки на основе `ratio`.

    Пояснение: помечает спорные, low-consensus и unanimous примеры для дальнейшего аудита.
    """
    enriched = df.copy()
    enriched["is_disputed"] = enriched["ratio"] < (1.0 - EPSILON)
    enriched["is_low_consensus"] = enriched["ratio"] <= LOW_CONSENSUS_MAX
    enriched["consensus_band"] = "medium"
    enriched.loc[enriched["is_low_consensus"], "consensus_band"] = "low"
    enriched.loc[~enriched["is_disputed"], "consensus_band"] = "unanimous"
    return enriched


def build_ratio_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Строит распределение `ratio` по train, val и общему набору.

    Пояснение: считает число примеров, долю и веса для разных weighting policy.
    """
    frames: list[pd.DataFrame] = []
    scopes = [("all", df)] + [
        (scope, scope_df.copy()) for scope, scope_df in df.groupby("source_dataset")
    ]

    for scope, scope_df in scopes:
        grouped = (
            scope_df.groupby(["ratio", "consensus_band"], as_index=False)
            .size()
            .rename(columns={"size": "samples", "ratio": "ratio_value"})
            .sort_values(["ratio_value"], kind="stable")
            .reset_index(drop=True)
        )
        grouped["scope"] = scope
        grouped["share_of_scope"] = grouped["samples"] / len(scope_df)
        grouped["sample_weight_linear"] = grouped["ratio_value"]
        grouped["sample_weight_limited"] = grouped["ratio_value"].clip(lower=0.75)
        grouped["total_samples_in_scope"] = len(scope_df)
        frames.append(grouped)

    result = pd.concat(frames, ignore_index=True)
    return result[
        [
            "scope",
            "ratio_value",
            "consensus_band",
            "samples",
            "share_of_scope",
            "sample_weight_linear",
            "sample_weight_limited",
            "total_samples_in_scope",
        ]
    ].sort_values(["scope", "ratio_value"], kind="stable").reset_index(drop=True)


def build_class_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Считает статистику качества разметки по классам.

    Пояснение: даёт размер класса, медианный `ratio` и долю спорных/low-consensus примеров.
    """
    grouped = (
        df.groupby(["result", "label"], as_index=False)
        .agg(
            samples=("ratio", "size"),
            median_ratio=("ratio", "median"),
            mean_ratio=("ratio", "mean"),
            disputed_samples=("is_disputed", "sum"),
            low_consensus_samples=("is_low_consensus", "sum"),
            train_pool_samples=("split_role", lambda s: int((s == "train_pool").sum())),
            shadow_holdout_samples=(
                "split_role",
                lambda s: int((s == "shadow_holdout").sum()),
            ),
        )
        .sort_values(["result"], kind="stable")
        .reset_index(drop=True)
    )
    grouped["share_of_labeled"] = grouped["samples"] / len(df)
    grouped["disputed_share"] = grouped["disputed_samples"] / grouped["samples"]
    grouped["low_consensus_share"] = (
        grouped["low_consensus_samples"] / grouped["samples"]
    )
    return grouped


def build_low_consensus_samples(
    df: pd.DataFrame, class_summary: pd.DataFrame
) -> pd.DataFrame:
    """Собирает полный список изображений с низким согласием.

    Пояснение: добавляет контекст по item и классу, чтобы спорные примеры было удобно смотреть руками.
    """
    low = df.loc[df["is_low_consensus"]].copy()

    item_stats = (
        df.groupby("item_id", as_index=False)
        .agg(
            item_total_images=("image_id_ext", "size"),
            item_disputed_images=("is_disputed", "sum"),
            item_low_consensus_images=("is_low_consensus", "sum"),
        )
        .reset_index(drop=True)
    )

    class_stats = class_summary[
        [
            "result",
            "label",
            "samples",
            "median_ratio",
            "disputed_share",
            "low_consensus_share",
        ]
    ].rename(
        columns={
            "samples": "class_samples",
            "median_ratio": "class_median_ratio",
            "disputed_share": "class_disputed_share",
            "low_consensus_share": "class_low_consensus_share",
        }
    )

    low = low.merge(item_stats, on="item_id", how="left")
    low = low.merge(class_stats, on=["result", "label"], how="left")
    low = low[
        [
            "source_dataset",
            "split_role",
            "item_id",
            "image_id_ext",
            "image",
            "result",
            "label",
            "ratio",
            "consensus_band",
            "item_total_images",
            "item_disputed_images",
            "item_low_consensus_images",
            "class_samples",
            "class_median_ratio",
            "class_disputed_share",
            "class_low_consensus_share",
        ]
    ]
    return low.sort_values(
        [
            "ratio",
            "class_low_consensus_share",
            "item_low_consensus_images",
            "class_samples",
            "image_id_ext",
        ],
        ascending=[True, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def select_report_examples(
    low_consensus_samples: pd.DataFrame, class_summary: pd.DataFrame, limit: int = 10
) -> pd.DataFrame:
    """Выбирает короткий список спорных примеров для markdown-отчёта.

    Пояснение: берёт по одному показательному примеру на класс с самым низким согласием.
    """
    if low_consensus_samples.empty:
        return low_consensus_samples

    examples = low_consensus_samples.drop_duplicates(subset=["result"], keep="first")
    examples = examples.merge(
        class_summary[["result", "low_consensus_share", "samples"]],
        on="result",
        how="left",
    )
    examples = examples.sort_values(
        ["low_consensus_share", "samples", "ratio", "image_id_ext"],
        ascending=[False, True, True, True],
        kind="stable",
    ).head(limit)
    return examples[
        [
            "result",
            "label",
            "image_id_ext",
            "item_id",
            "source_dataset",
            "split_role",
            "ratio",
            "low_consensus_share",
        ]
    ].reset_index(drop=True)


def recommend_weighting_policy(
    df: pd.DataFrame, class_summary: pd.DataFrame
) -> dict[str, Any]:
    """Выбирает рекомендацию по weighting policy на основе статистики `ratio`.

    Пояснение: решает, стоит ли не взвешивать, взвешивать линейно или ограниченно через clip.
    """
    total_samples = len(df)
    disputed_samples = int(df["is_disputed"].sum())
    low_consensus_samples = int(df["is_low_consensus"].sum())
    disputed_share = disputed_samples / total_samples
    low_consensus_share = low_consensus_samples / total_samples

    disputed_ratio_counts = (
        df.loc[df["is_disputed"], "ratio"].round(6).value_counts().sort_index()
    )
    dominant_disputed_ratio = float(disputed_ratio_counts.idxmax())
    dominant_disputed_share = (
        float(disputed_ratio_counts.max()) / disputed_samples if disputed_samples else 0.0
    )

    rare_classes = class_summary.loc[class_summary["samples"] < 120].copy()
    rare_classes = rare_classes.sort_values(
        ["low_consensus_share", "samples"], ascending=[False, True], kind="stable"
    )
    rare_hotspots = rare_classes.head(2)
    rare_labels = ", ".join(
        f"{row.label} ({format_pct(row.low_consensus_share)})"
        for row in rare_hotspots.itertuples()
    )

    if disputed_share < 0.10:
        return {
            "policy_name": "не взвешивать",
            "policy_key": "no_weighting",
            "formula": "sample_weight = 1.0",
            "reason_lines": [
                f"Спорных примеров мало: {disputed_samples}/{total_samples} ({format_pct(disputed_share)}).",
                "Дополнительное weighting почти не изменит effective dataset.",
            ],
        }

    if low_consensus_share > 0.30 and dominant_disputed_share > 0.90:
        return {
            "policy_name": "ограниченно",
            "policy_key": "limited_weighting",
            "formula": "sample_weight = clip(ratio, 0.75, 1.0)",
            "reason_lines": [
                f"Невысокое согласие у {low_consensus_samples}/{total_samples} примеров ({format_pct(low_consensus_share)}).",
                f"Почти все спорные примеры сидят в одном уровне `ratio={format_ratio(dominant_disputed_ratio)}` ({format_pct(dominant_disputed_share)} от всех спорных).",
                f"Редкие классы тоже шумные: {rare_labels}.",
                "Сырым линейным weighting легко недовзвесить редкие классы, поэтому безопаснее мягкий clip.",
            ],
        }

    return {
        "policy_name": "линейно",
        "policy_key": "linear_weighting",
        "formula": "sample_weight = ratio",
        "reason_lines": [
            f"Спорных примеров достаточно: {disputed_samples}/{total_samples} ({format_pct(disputed_share)}).",
            "Распределение ratio не схлопывается в один спорный уровень, линейное weighting даст полезный градиент доверия.",
        ],
    }


def build_report(
    df: pd.DataFrame,
    split_meta: dict[str, Any],
    class_summary: pd.DataFrame,
    ratio_bins: pd.DataFrame,
    low_consensus_samples: pd.DataFrame,
    weighting_policy: dict[str, Any],
    train_csv_path: Path,
    val_csv_path: Path,
    splits_json_path: Path,
    report_path: Path,
    ratio_bins_path: Path,
    low_consensus_path: Path,
) -> str:
    """Собирает итоговый markdown-отчёт DATA-03.

    Пояснение: сводит метаданные split, агрегаты по классам и рекомендации в один `.md` артефакт.
    """
    generated_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    total_samples = len(df)
    disputed_samples = int(df["is_disputed"].sum())
    low_samples = int(df["is_low_consensus"].sum())
    unanimous_samples = int((~df["is_disputed"]).sum())
    median_ratio = float(df["ratio"].median())
    mean_ratio = float(df["ratio"].mean())

    ratio_bins_all = ratio_bins.loc[ratio_bins["scope"] == "all", :].copy()
    ratio_bins_all = ratio_bins_all[
        ["ratio_value", "consensus_band", "samples", "share_of_scope"]
    ]

    class_table = class_summary[
        [
            "result",
            "label",
            "samples",
            "median_ratio",
            "mean_ratio",
            "disputed_samples",
            "disputed_share",
            "low_consensus_samples",
            "low_consensus_share",
        ]
    ]

    hotspot_table = class_summary.sort_values(
        ["low_consensus_share", "samples"], ascending=[False, True], kind="stable"
    ).head(8)[
        [
            "result",
            "label",
            "samples",
            "median_ratio",
            "disputed_share",
            "low_consensus_share",
        ]
    ]

    examples_table = select_report_examples(low_consensus_samples, class_summary)

    report_lines = [
        "# DATA-03 Label Quality / Ratio Audit",
        "",
        "## Входы и фиксация split",
        f"- generated_at_utc: `{generated_at}`",
        f"- train_csv: `{train_csv_path.as_posix()}`",
        f"- val_csv: `{val_csv_path.as_posix()}`",
        f"- splits_json: `{splits_json_path.as_posix()}`",
        f"- split_version: `{split_meta['version']}`",
        f"- split_timestamp_utc: `{split_meta['timestamp_utc']}`",
        (
            "- safe_split: "
            f"`{split_meta['shadow_holdout_status']}` via "
            f"`{split_meta['splitter']}` on `{split_meta['group_key']}` "
            f"with `{split_meta['n_folds']}` folds"
        ),
        f"- safe_split_reason: {split_meta['shadow_holdout_reason']}",
        (
            "- split_rows: "
            f"train_pool={split_meta['train_pool_rows_after_filters']}, "
            f"shadow_holdout={split_meta['shadow_holdout_rows_after_filters']}, "
            f"total_labeled={total_samples}"
        ),
        "",
        "## Ключевые выводы",
        f"- Размеченных изображений: `{total_samples}`.",
        f"- Полное согласие (`ratio = 1.0`): `{unanimous_samples}` / `{total_samples}` ({format_pct(unanimous_samples / total_samples)}).",
        f"- Спорных (`ratio < 1.0`): `{disputed_samples}` / `{total_samples}` ({format_pct(disputed_samples / total_samples)}).",
        f"- Низкое согласие (`ratio <= 2/3`): `{low_samples}` / `{total_samples}` ({format_pct(low_samples / total_samples)}).",
        f"- Медианный `ratio` по всему labeled set: `{format_ratio(median_ratio)}`; средний: `{format_ratio(mean_ratio)}`.",
        (
            "- Рекомендация по weighting policy: "
            f"**{weighting_policy['policy_name']}**. "
            f"Стартовая формула: `{weighting_policy['formula']}`."
        ),
        "- Почему:",
    ]
    report_lines.extend(f"  - {reason}" for reason in weighting_policy["reason_lines"])
    report_lines.extend(
        [
            "",
            "## Распределение ratio",
            markdown_table(
                ratio_bins_all,
                formatters={
                    "ratio_value": format_ratio,
                    "share_of_scope": format_pct,
                },
            ),
            "",
            "## По классам",
            markdown_table(
                class_table,
                formatters={
                    "median_ratio": format_ratio,
                    "mean_ratio": format_ratio,
                    "disputed_share": format_pct,
                    "low_consensus_share": format_pct,
                },
            ),
            "",
            "## Классы с самым частым низким согласием",
            markdown_table(
                hotspot_table,
                formatters={
                    "median_ratio": format_ratio,
                    "disputed_share": format_pct,
                    "low_consensus_share": format_pct,
                },
            ),
            "",
            "## Примеры спорных изображений",
            "Полный список лежит в `tables/low_consensus_samples.csv`.",
            markdown_table(
                examples_table,
                formatters={
                    "ratio": format_ratio,
                    "low_consensus_share": format_pct,
                },
            ),
            "",
            "## Артефакты",
            f"- report: `{report_path.as_posix()}`",
            f"- ratio_bins: `{ratio_bins_path.as_posix()}`",
            f"- low_consensus_samples: `{low_consensus_path.as_posix()}`",
            "",
            "## Re-run",
            "```bash",
            "python3 scripts/data03_audit_label_quality_ratio.py",
            "```",
            "",
        ]
    )
    return "\n".join(report_lines)


def main() -> None:
    """Запускает полный pipeline аудита label quality и ratio.

    Пояснение: читает входы, считает статистики и записывает report плюс две CSV-таблицы.
    """
    args = parse_args()

    train_df = load_labeled_csv(
        args.train_csv,
        required_columns=REQUIRED_LABELED_COLUMNS,
        source_dataset="train_df",
        split_role="train_pool",
        ratio_column="ratio",
    )
    val_df = load_labeled_csv(
        args.val_csv,
        required_columns=REQUIRED_LABELED_COLUMNS,
        source_dataset="val_df",
        split_role="shadow_holdout",
        ratio_column="ratio",
    )
    split_meta = load_split_metadata(args.splits_json, len(train_df), len(val_df))

    labeled_df = pd.concat([train_df, val_df], ignore_index=True)
    labeled_df = add_consensus_flags(labeled_df)

    ratio_bins = build_ratio_bins(labeled_df)
    class_summary = build_class_summary(labeled_df)
    low_consensus_samples = build_low_consensus_samples(labeled_df, class_summary)
    weighting_policy = recommend_weighting_policy(labeled_df, class_summary)

    ensure_parent(args.report_md)
    ensure_parent(args.ratio_bins_csv)
    ensure_parent(args.low_consensus_csv)

    ratio_bins.to_csv(args.ratio_bins_csv, index=False, float_format="%.6f")
    low_consensus_samples.to_csv(
        args.low_consensus_csv, index=False, float_format="%.6f"
    )
    report_text = build_report(
        df=labeled_df,
        split_meta=split_meta,
        class_summary=class_summary,
        ratio_bins=ratio_bins,
        low_consensus_samples=low_consensus_samples,
        weighting_policy=weighting_policy,
        train_csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        splits_json_path=args.splits_json,
        report_path=args.report_md,
        ratio_bins_path=args.ratio_bins_csv,
        low_consensus_path=args.low_consensus_csv,
    )
    args.report_md.write_text(report_text)


if __name__ == "__main__":
    main()
