from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


VERSION = "weak_labels_v1"
DEFAULT_WEAK_WEIGHT = 0.5
DEFAULT_HEURISTIC_CSVS = (
    Path("data/raw/heuristics_cabinet.csv"),
    Path("data/raw/heuristics_detskaya.csv"),
    Path("data/raw/heuristics_dressing_room.csv"),
)
DEFAULT_TRAIN_CSV = Path("data/raw/train_df.csv")
DEFAULT_MANIFEST = Path("data/processed/data_manifest.parquet")
DEFAULT_OUTPUT_PARQUET = Path("data/processed/weak_labels_v1.parquet")
DEFAULT_REPORT_MD = Path("reports/weak_labels_audit.md")

SOURCE_POLICY: dict[str, dict[str, Any]] = {
    "heuristics_cabinet": {"class_id": 5, "source": "heuristics_cabinet"},
    "heuristics_detskaya": {"class_id": 6, "source": "heuristics_detskaya"},
    "heuristics_dressing_room": {"class_id": 11, "source": "heuristics_dressing_room"},
}

REQUIRED_HEURISTIC_COLUMNS = {"image_id_ext"}
REQUIRED_TRAIN_COLUMNS = {"image_id_ext"}
REQUIRED_MANIFEST_COLUMNS = {"image_id_ext", "hash_sha256"}
OUTPUT_COLUMNS = [
    "image_id_ext",
    "class_id",
    "weak_weight",
    "source",
    "hash_sha256",
    "is_train_overlap_image_id_ext",
    "is_train_overlap_hash_sha256",
]


@dataclass(frozen=True)
class WeakLabelsResult:
    weak_labels: pd.DataFrame
    audit: dict[str, Any]


def normalize_image_id_ext(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text if Path(text).suffix else f"{text}.jpg"


def require_columns(df: pd.DataFrame, required: set[str], name: str | Path) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def source_name_from_path(path: Path) -> str:
    source = path.stem
    if source not in SOURCE_POLICY:
        known = ", ".join(sorted(SOURCE_POLICY))
        raise ValueError(f"Unknown weak-label heuristic source {source!r}. Known: {known}")
    return source


def load_heuristic_sources(paths: list[Path]) -> dict[str, pd.DataFrame]:
    sources: dict[str, pd.DataFrame] = {}
    for path in paths:
        source = source_name_from_path(path)
        if source in sources:
            raise ValueError(f"Duplicate heuristic source: {source}")
        df = pd.read_csv(path)
        require_columns(df, REQUIRED_HEURISTIC_COLUMNS, path)
        sources[source] = df
    return sources


def load_train_keys(path: Path) -> pd.DataFrame:
    train = pd.read_csv(path)
    require_columns(train, REQUIRED_TRAIN_COLUMNS, path)
    return train


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_parquet(path)
    require_columns(manifest, REQUIRED_MANIFEST_COLUMNS, path)
    manifest = manifest.copy()
    manifest["image_id_ext"] = manifest["image_id_ext"].map(normalize_image_id_ext)
    manifest = manifest[["image_id_ext", "hash_sha256"]]
    manifest = manifest.sort_values(["image_id_ext"], kind="stable")
    return manifest.drop_duplicates(subset=["image_id_ext"], keep="first").reset_index(drop=True)


def build_raw_weak_rows(
    heuristic_sources: dict[str, pd.DataFrame], weak_weight: float = DEFAULT_WEAK_WEIGHT
) -> tuple[pd.DataFrame, dict[str, int]]:
    frames: list[pd.DataFrame] = []
    input_counts: dict[str, int] = {}

    for source in sorted(heuristic_sources):
        df = heuristic_sources[source]
        require_columns(df, REQUIRED_HEURISTIC_COLUMNS, source)
        policy = SOURCE_POLICY[source]
        frame = pd.DataFrame(
            {
                "image_id_ext": df["image_id_ext"].map(normalize_image_id_ext),
                "class_id": int(policy["class_id"]),
                "weak_weight": float(weak_weight),
                "source": str(policy["source"]),
            }
        )
        frames.append(frame)
        input_counts[source] = int(len(frame))

    if not frames:
        raise ValueError("No heuristic sources were provided.")

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.loc[raw["image_id_ext"] != ""].copy().reset_index(drop=True)
    return raw, input_counts


def enrich_with_hashes(weak_rows: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    require_columns(manifest, REQUIRED_MANIFEST_COLUMNS, "manifest")
    manifest_keys = manifest.copy()
    manifest_keys["image_id_ext"] = manifest_keys["image_id_ext"].map(normalize_image_id_ext)
    manifest_keys = manifest_keys[["image_id_ext", "hash_sha256"]]
    manifest_keys = manifest_keys.sort_values(["image_id_ext"], kind="stable")
    manifest_keys = manifest_keys.drop_duplicates(subset=["image_id_ext"], keep="first")
    return weak_rows.merge(manifest_keys, on="image_id_ext", how="left")


def add_train_overlap_flags(weak_rows: pd.DataFrame, train: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    require_columns(train, REQUIRED_TRAIN_COLUMNS, "train")
    train_ids = train["image_id_ext"].map(normalize_image_id_ext)
    train_id_set = set(train_ids.dropna())

    manifest_keys = manifest.copy()
    manifest_keys["image_id_ext"] = manifest_keys["image_id_ext"].map(normalize_image_id_ext)
    train_hashes = (
        pd.DataFrame({"image_id_ext": train_ids})
        .merge(manifest_keys[["image_id_ext", "hash_sha256"]], on="image_id_ext", how="left")[
            "hash_sha256"
        ]
        .dropna()
    )
    train_hash_set = set(train_hashes)

    result = weak_rows.copy()
    result["is_train_overlap_image_id_ext"] = result["image_id_ext"].isin(train_id_set)
    result["is_train_overlap_hash_sha256"] = result["hash_sha256"].notna() & result[
        "hash_sha256"
    ].isin(train_hash_set)
    return result


def duplicate_counts(df: pd.DataFrame) -> dict[str, int]:
    image_id_duplicate_rows = int(df.duplicated(subset=["image_id_ext"], keep="first").sum())
    after_image_id = df.drop_duplicates(subset=["image_id_ext"], keep="first")
    hashable = after_image_id.loc[after_image_id["hash_sha256"].notna()].copy()
    return {
        "image_id_ext_rows": image_id_duplicate_rows,
        "hash_sha256_rows": int(
            hashable.duplicated(subset=["hash_sha256"], keep="first").sum()
        ),
    }


def remove_train_overlaps_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    kept = df.loc[
        ~df["is_train_overlap_image_id_ext"] & ~df["is_train_overlap_hash_sha256"]
    ].copy()
    kept = kept.sort_values(["class_id", "source", "image_id_ext"], kind="stable")
    kept = kept.drop_duplicates(subset=["image_id_ext"], keep="first")

    with_hash = kept.loc[kept["hash_sha256"].notna()].copy()
    without_hash = kept.loc[kept["hash_sha256"].isna()].copy()
    with_hash = with_hash.drop_duplicates(subset=["hash_sha256"], keep="first")
    kept = pd.concat([with_hash, without_hash], ignore_index=True)
    kept = kept.sort_values(["class_id", "source", "image_id_ext"], kind="stable")[
        OUTPUT_COLUMNS
    ].reset_index(drop=True)
    kept["hash_sha256"] = kept["hash_sha256"].astype(object).where(
        kept["hash_sha256"].notna(), None
    )
    return kept


def summarize_final(df: pd.DataFrame) -> dict[str, Any]:
    by_source_class = (
        df.groupby(["source", "class_id"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values(["source", "class_id"], kind="stable")
    )
    by_class = df.groupby("class_id").size().sort_index()
    by_source = df.groupby("source").size().sort_index()
    return {
        "rows": int(len(df)),
        "by_source": {str(k): int(v) for k, v in by_source.items()},
        "by_class": {int(k): int(v) for k, v in by_class.items()},
        "by_source_class": by_source_class.to_dict(orient="records"),
    }


def build_weak_labels(
    heuristic_sources: dict[str, pd.DataFrame],
    train: pd.DataFrame,
    manifest: pd.DataFrame,
    weak_weight: float = DEFAULT_WEAK_WEIGHT,
) -> WeakLabelsResult:
    unknown = sorted(set(heuristic_sources).difference(SOURCE_POLICY))
    if unknown:
        raise ValueError(f"Unknown weak-label heuristic sources: {unknown}")
    for source, df in heuristic_sources.items():
        require_columns(df, REQUIRED_HEURISTIC_COLUMNS, source)
    require_columns(train, REQUIRED_TRAIN_COLUMNS, "train")
    require_columns(manifest, REQUIRED_MANIFEST_COLUMNS, "manifest")

    raw, input_counts = build_raw_weak_rows(heuristic_sources, weak_weight=weak_weight)
    hashed = enrich_with_hashes(raw, manifest)
    flagged = add_train_overlap_flags(hashed, train, manifest)

    pre_dedup_non_train = flagged.loc[
        ~flagged["is_train_overlap_image_id_ext"] & ~flagged["is_train_overlap_hash_sha256"]
    ].copy()
    duplicates = duplicate_counts(pre_dedup_non_train)
    final = remove_train_overlaps_and_duplicates(flagged)

    mapped_class_ids = {
        source: int(SOURCE_POLICY[source]["class_id"]) for source in sorted(heuristic_sources)
    }
    audit = {
        "version": VERSION,
        "policy": {
            "source_to_class_id": mapped_class_ids,
            "weak_weight": float(weak_weight),
            "source_identity": "heuristic CSV filename stem",
            "train_overlap_policy": "drop rows overlapping train by image_id_ext or hash_sha256",
            "duplicate_policy": "drop weak duplicates by image_id_ext, then by hash_sha256 when hash is available; keep deterministic first sorted by class/source/image_id_ext",
        },
        "input_rows_by_source": input_counts,
        "mapped_class_ids_by_source": mapped_class_ids,
        "missing_hash_rows": int(flagged["hash_sha256"].isna().sum()),
        "train_overlap_rows": {
            "image_id_ext": int(flagged["is_train_overlap_image_id_ext"].sum()),
            "hash_sha256": int(flagged["is_train_overlap_hash_sha256"].sum()),
            "either": int(
                (
                    flagged["is_train_overlap_image_id_ext"]
                    | flagged["is_train_overlap_hash_sha256"]
                ).sum()
            ),
        },
        "weak_internal_duplicate_rows": duplicates,
        "final": summarize_final(final),
    }
    return WeakLabelsResult(weak_labels=final, audit=audit)


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


def build_audit_report(audit: dict[str, Any]) -> str:
    source_rows = [[k, v] for k, v in sorted(audit["input_rows_by_source"].items())]
    class_rows = [[k, v] for k, v in sorted(audit["final"]["by_class"].items())]
    final_rows = [
        [row["source"], row["class_id"], row["rows"]]
        for row in audit["final"]["by_source_class"]
    ]
    mapping_rows = [[k, v] for k, v in sorted(audit["mapped_class_ids_by_source"].items())]

    return "\n".join(
        [
            "# Weak Labels Audit: weak_labels_v1",
            "",
            "## Policy and assumptions",
            f"- version: `{audit['version']}`",
            f"- source_identity: `{audit['policy']['source_identity']}`",
            f"- weak_weight: `{audit['policy']['weak_weight']}` for every v1 weak label",
            f"- train_overlap_policy: `{audit['policy']['train_overlap_policy']}`",
            f"- duplicate_policy: `{audit['policy']['duplicate_policy']}`",
            "- no images are downloaded; hashes are reused from `data/processed/data_manifest.parquet`.",
            "",
            "## Source to class mapping",
            markdown_table(["source", "class_id"], mapping_rows),
            "",
            "## Input rows by source",
            markdown_table(["source", "rows"], source_rows),
            "",
            "## Missing hashes",
            f"- missing_hash_rows: `{audit['missing_hash_rows']}`",
            "",
            "## Train overlaps removed",
            f"- by image_id_ext: `{audit['train_overlap_rows']['image_id_ext']}`",
            f"- by hash_sha256: `{audit['train_overlap_rows']['hash_sha256']}`",
            f"- either key: `{audit['train_overlap_rows']['either']}`",
            "",
            "## Weak internal duplicates removed",
            f"- by image_id_ext: `{audit['weak_internal_duplicate_rows']['image_id_ext_rows']}`",
            f"- by hash_sha256: `{audit['weak_internal_duplicate_rows']['hash_sha256_rows']}`",
            "",
            "## Final rows by class",
            markdown_table(["class_id", "rows"], class_rows),
            "",
            "## Final rows by source/class",
            markdown_table(["source", "class_id", "rows"], final_rows),
            "",
            "## Final row count",
            f"- rows: `{audit['final']['rows']}`",
            "",
            "## Re-run",
            "```bash",
            "python3 scripts/build_weak_labels_v1.py",
            "```",
            "",
        ]
    )


def write_artifacts(df: pd.DataFrame, report_text: str, parquet_path: Path, report_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    report_path.write_text(report_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weak_labels_v1 parquet and audit report.")
    parser.add_argument("--heuristic-csv", type=Path, action="append", dest="heuristic_csvs")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--weak-weight", type=float, default=DEFAULT_WEAK_WEIGHT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heuristic_csvs = args.heuristic_csvs or list(DEFAULT_HEURISTIC_CSVS)
    heuristic_sources = load_heuristic_sources(heuristic_csvs)
    train = load_train_keys(args.train_csv)
    manifest = load_manifest(args.manifest)
    result = build_weak_labels(
        heuristic_sources=heuristic_sources,
        train=train,
        manifest=manifest,
        weak_weight=args.weak_weight,
    )
    report_text = build_audit_report(result.audit)
    write_artifacts(result.weak_labels, report_text, args.output_parquet, args.report_md)
    print(f"weak_labels -> {args.output_parquet}")
    print(f"audit_report -> {args.report_md}")


if __name__ == "__main__":
    main()
