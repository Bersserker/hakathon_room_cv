from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.datasets.make_manifest import MANIFEST_COLUMNS, process_split
from src.utils.labeled_data import require_columns

REQUIRED_LABELED_COLUMNS = {"item_id", "image", "image_id_ext", "result", "label", "ratio"}
REQUIRED_TEST_COLUMNS = {"item_id", "image", "image_id_ext"}
SPLIT_SPECS = {
    "train": ("train_df.csv", "train_images"),
    "val": ("val_df.csv", "val_images"),
    "test": ("test_df.csv", "test_images"),
}
EXACT_LEAK_KEYS = ["image_id_ext_file", "image", "content_hash"]
REPORT_KEYS = ["item_id", *EXACT_LEAK_KEYS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe preprocessed dataset.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/preprocess"))
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("reports/preprocess_leakage_report.md"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Defaults to OUT_DIR/preprocess_summary.json.",
    )
    return parser.parse_args()


def current_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_image_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def normalize_image_file(value: Any) -> str:
    text = normalize_image_id(value)
    return text if Path(text).suffix else f"{text}.jpg"


def sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in ["item_id", "image_id_ext", "image"] if col in df.columns]
    if not columns:
        return df.reset_index(drop=True)
    return df.sort_values(columns, kind="stable").reset_index(drop=True)


def read_labeled_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, REQUIRED_LABELED_COLUMNS, path)
    df = df.copy()
    df["image_id_ext"] = df["image_id_ext"].map(normalize_image_id)
    df["ratio"] = pd.to_numeric(df["ratio"], errors="raise")
    if ((df["ratio"] <= 0.0) | (df["ratio"] > 1.0)).any():
        raise ValueError(f"{path} has ratio values outside (0, 1].")
    return sort_frame(df)


def read_test_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, REQUIRED_TEST_COLUMNS, path)
    df = df.copy()
    df["image_id_ext"] = df["image_id_ext"].map(normalize_image_id)
    return sort_frame(df)


def build_manifest(raw_dir: Path, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split, (_csv_name, folder_name) in SPLIT_SPECS.items():
        rows.extend(process_split(split, frames[split], raw_dir, folder_name))
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def attach_manifest(df: pd.DataFrame, manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    manifest_split = manifest.loc[manifest["split"] == split].copy()
    manifest_split["image_id_ext_file"] = manifest_split["image_id_ext"].map(normalize_image_file)
    manifest_split = manifest_split.rename(columns={"hash_sha256": "content_hash"})[
        ["image_id_ext_file", "local_path", "status", "content_hash", "width", "height"]
    ]

    result = df.copy()
    result["image_id_ext_file"] = result["image_id_ext"].map(normalize_image_file)
    return result.merge(manifest_split, on="image_id_ext_file", how="left")


def value_set(df: pd.DataFrame, column: str) -> set[Any]:
    if column not in df.columns:
        return set()
    values = df[column].dropna()
    if column == "content_hash":
        values = values.loc[values.astype(str) != ""]
    return set(values.astype(str).tolist())


def overlap_summary(left: pd.DataFrame, right: pd.DataFrame, column: str) -> dict[str, Any]:
    left_values = value_set(left, column)
    right_values = value_set(right, column)
    intersection = sorted(left_values.intersection(right_values))
    return {
        "left_unique": int(len(left_values)),
        "right_unique": int(len(right_values)),
        "intersection_count": int(len(intersection)),
        "sample": intersection[:10],
    }


def exact_test_leak_mask(df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for key in EXACT_LEAK_KEYS:
        test_values = value_set(test_df, key)
        if not test_values:
            continue
        mask = mask | df[key].astype(str).isin(test_values)
    return mask


def drop_exact_test_leaks(
    df: pd.DataFrame, test_df: pd.DataFrame, original_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mask = exact_test_leak_mask(df, test_df)
    dropped = df.loc[mask].copy()
    counts_by_key = {}
    for key in EXACT_LEAK_KEYS:
        test_values = value_set(test_df, key)
        counts_by_key[key] = int(df[key].astype(str).isin(test_values).sum()) if test_values else 0

    kept = df.loc[~mask, original_columns].copy().reset_index(drop=True)
    return kept, {
        "dropped_rows": int(len(dropped)),
        "counts_by_key": counts_by_key,
        "sample_image_id_ext": dropped["image_id_ext"].astype(str).head(10).tolist(),
    }


def pairwise_overlaps(frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    result: dict[str, dict[str, Any]] = {}
    for left, right in pairs:
        pair_key = f"{left}_vs_{right}"
        result[pair_key] = {
            column: overlap_summary(frames[left], frames[right], column) for column in REPORT_KEYS
        }
    return result


def fatal_exact_leaks(overlaps: dict[str, dict[str, Any]]) -> list[str]:
    failures = []
    for pair_name, pair_summary in overlaps.items():
        for column in EXACT_LEAK_KEYS:
            count = pair_summary[column]["intersection_count"]
            if count > 0:
                failures.append(f"{pair_name}.{column}={count}")
    return failures


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def write_report(path: Path, summary: dict[str, Any]) -> None:
    status_rows = []
    for split, counts in summary["manifest_status_counts"].items():
        for status, count in counts.items():
            status_rows.append([split, status, count])

    drop_rows = []
    for split, payload in summary["dropped_exact_test_overlaps"].items():
        drop_rows.append(
            [
                split,
                payload["dropped_rows"],
                payload["counts_by_key"].get("image_id_ext_file", 0),
                payload["counts_by_key"].get("image", 0),
                payload["counts_by_key"].get("content_hash", 0),
                ", ".join(payload["sample_image_id_ext"]) or "-",
            ]
        )

    overlap_rows = []
    for pair_name, pair_summary in summary["overlaps_after_filtering"].items():
        for column, payload in pair_summary.items():
            overlap_rows.append(
                [
                    pair_name,
                    column,
                    payload["intersection_count"],
                    ", ".join(payload["sample"]) if payload["sample"] else "-",
                ]
            )

    lines = [
        "# DATA-01 Preprocess and leakage report",
        "",
        f"- generated_at_utc: `{summary['generated_at_utc']}`",
        f"- raw_dir: `{summary['raw_dir']}`",
        f"- out_dir: `{summary['out_dir']}`",
        f"- decision: `{summary['decision']}`",
        "",
        "## Outputs",
        f"- train_csv: `{summary['outputs']['train_csv']}`",
        f"- val_csv: `{summary['outputs']['val_csv']}`",
        f"- test_csv: `{summary['outputs']['test_csv']}`",
        f"- manifest: `{summary['outputs']['manifest']}`",
        f"- summary_json: `{summary['outputs']['summary_json']}`",
        "",
        "## Rows",
        markdown_table(
            ["split", "raw_rows", "saved_rows"],
            [
                [split, summary["raw_rows"][split], summary["saved_rows"][split]]
                for split in ["train", "val", "test"]
            ],
        ),
        "",
        "## Manifest status",
        markdown_table(["split", "status", "rows"], status_rows),
        "",
        "## Dropped exact overlaps with test",
        markdown_table(
            [
                "split",
                "dropped_rows",
                "image_id_ext_matches",
                "image_url_matches",
                "content_hash_matches",
                "sample_image_id_ext",
            ],
            drop_rows,
        ),
        "",
        "## Overlaps after filtering",
        markdown_table(["pair", "key", "intersection_count", "sample"], overlap_rows),
        "",
        "## Leakage policy",
        "- Fatal leakage keys: `image_id_ext`, `image` URL, `content_hash`.",
        "- Exact train/test and val/test overlaps are removed from labeled CSVs before split building.",
        "- `item_id` overlaps with test are reported only: the image model does not consume item_id as a feature.",
        "- Fold-level item/content-hash leakage is checked by DATA-02 split builder.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    summary_json = args.summary_json or (args.out_dir / "preprocess_summary.json")

    train = read_labeled_csv(args.raw_dir / "train_df.csv")
    val = read_labeled_csv(args.raw_dir / "val_df.csv")
    test = read_test_csv(args.raw_dir / "test_df.csv")
    raw_frames = {"train": train, "val": val, "test": test}

    manifest = build_manifest(args.raw_dir, raw_frames)
    attached = {
        split: attach_manifest(frame, manifest, split) for split, frame in raw_frames.items()
    }

    train_clean, train_drops = drop_exact_test_leaks(
        attached["train"], attached["test"], train.columns.tolist()
    )
    val_clean, val_drops = drop_exact_test_leaks(
        attached["val"], attached["test"], val.columns.tolist()
    )
    clean_frames = {"train": train_clean, "val": val_clean, "test": test}
    clean_attached = {
        "train": attach_manifest(train_clean, manifest, "train"),
        "val": attach_manifest(val_clean, manifest, "val"),
        "test": attached["test"],
    }

    overlaps_after = pairwise_overlaps(clean_attached)
    failures = fatal_exact_leaks(overlaps_after)
    if failures:
        raise RuntimeError(f"Exact leakage remains after preprocessing: {failures}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train_csv": args.out_dir / "train_df.csv",
        "val_csv": args.out_dir / "val_df.csv",
        "test_csv": args.out_dir / "test_df.csv",
        "manifest": args.out_dir / "data_manifest.parquet",
        "summary_json": summary_json,
    }

    save_csv(clean_frames["train"], outputs["train_csv"])
    save_csv(clean_frames["val"], outputs["val_csv"])
    save_csv(clean_frames["test"], outputs["test_csv"])
    manifest.to_parquet(outputs["manifest"], index=False, engine="pyarrow")

    summary = {
        "generated_at_utc": current_utc(),
        "raw_dir": str(args.raw_dir),
        "out_dir": str(args.out_dir),
        "decision": "ok_no_exact_leakage_after_filtering",
        "raw_rows": {split: int(len(frame)) for split, frame in raw_frames.items()},
        "saved_rows": {split: int(len(frame)) for split, frame in clean_frames.items()},
        "manifest_status_counts": {
            split: {
                str(key): int(value)
                for key, value in manifest.loc[manifest["split"] == split, "status"]
                .value_counts(dropna=False)
                .sort_index()
                .items()
            }
            for split in SPLIT_SPECS
        },
        "dropped_exact_test_overlaps": {"train": train_drops, "val": val_drops},
        "overlaps_after_filtering": overlaps_after,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.report_md, summary)

    print("Preprocess complete")
    for key, value in outputs.items():
        print(f"{key}: {value}")
    print(f"dropped_train_exact_test_overlaps: {train_drops['dropped_rows']}")
    print(f"dropped_val_exact_test_overlaps: {val_drops['dropped_rows']}")


if __name__ == "__main__":
    main()
