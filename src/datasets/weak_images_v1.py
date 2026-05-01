from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image

VERSION = "weak_images_v1"
DEFAULT_WEAK_WEIGHT = 0.35
DEFAULT_HEURISTIC_CSVS = (
    Path("data/raw/heuristics_cabinet.csv"),
    Path("data/raw/heuristics_detskaya.csv"),
    Path("data/raw/heuristics_dressing_room.csv"),
)
DEFAULT_LEGACY_IMAGES_DIR = Path("data/raw/heuristics_images/heuristics_images")
DEFAULT_MANIFEST = Path("data/processed/data_manifest.parquet")
DEFAULT_SPLITS_JSON = Path("data/splits/splits_v1.json")
DEFAULT_CLASS_MAPPING = Path("configs/data/class_mapping.yaml")
DEFAULT_OUTPUT_IMAGE_DIR = Path("data/raw/weak_images/weak_images")
DEFAULT_OUTPUT_CSV = Path("data/processed/weak_downloaded_v1.csv")
DEFAULT_REPORT_MD = Path("reports/weak_images_download_report.md")
DEFAULT_MAX_ADDED_PER_CLASS = {5: 180, 6: 80, 11: 200}
DEFAULT_MIN_WIDTH = 64
DEFAULT_MIN_HEIGHT = 64

SOURCE_POLICY: dict[str, dict[str, Any]] = {
    "heuristics_cabinet": {"class_id": 5, "source": "heuristics_cabinet"},
    "heuristics_detskaya": {"class_id": 6, "source": "heuristics_detskaya"},
    "heuristics_dressing_room": {
        "class_id": 11,
        "source": "heuristics_dressing_room",
    },
}

REQUIRED_HEURISTIC_COLUMNS = {"image_id_ext"}
REQUIRED_MANIFEST_COLUMNS = {"image_id_ext", "hash_sha256"}
MANIFEST_COLUMNS = [
    "image_id_ext",
    "class_id",
    "label",
    "weak_weight",
    "source",
    "source_dataset",
    "source_local_path",
    "selected_local_path",
    "hash_sha256",
    "width",
    "height",
    "candidate_score",
    "n_texts",
    "person_found",
    "is_catalog",
    "crop_area",
    "perform_top_microcat_prob",
    "perform_top_other_classes_prob",
    "leakage_checked",
    "selected_rank",
]


@dataclass(frozen=True)
class WeakImagesResult:
    manifest: pd.DataFrame
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
        raise ValueError(f"Unknown weak-image heuristic source {source!r}. Known: {known}")
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


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_parquet(path)
    require_columns(manifest, REQUIRED_MANIFEST_COLUMNS, path)
    manifest = manifest.copy()
    manifest["image_id_ext"] = manifest["image_id_ext"].map(normalize_image_id_ext)
    return manifest


def load_splits(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_class_mapping(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    mapping = data.get("id_to_label", {}) if isinstance(data, dict) else {}
    return {int(class_id): str(label) for class_id, label in mapping.items()}


def to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def as_number(value: Any, default: float = 0.0) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return default
    return float(number)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> tuple[str | None, int | None, int | None, str | None]:
    if not path.exists():
        return None, None, None, "missing_image"
    try:
        hash_sha256 = sha256_file(path)
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        return hash_sha256, int(width), int(height), None
    except Exception:
        return None, None, None, "corrupted_image"


def split_records(splits: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fold_payload in splits.get("folds", []):
        records.extend(fold_payload.get("records", []))
    records.extend(splits.get("shadow_holdout", {}).get("records", []))
    return records


def leakage_keys(manifest: pd.DataFrame, splits: dict[str, Any]) -> tuple[set[str], set[str]]:
    require_columns(manifest, REQUIRED_MANIFEST_COLUMNS, "manifest")
    image_ids = set(manifest["image_id_ext"].map(normalize_image_id_ext).dropna())
    hashes = set(manifest["hash_sha256"].dropna().astype(str))

    for record in split_records(splits):
        image_id = normalize_image_id_ext(record.get("image_id_ext"))
        if image_id:
            image_ids.add(image_id)
        for key in ("hash_sha256", "content_hash"):
            value = record.get(key)
            if value is not None and not pd.isna(value):
                hashes.add(str(value))
    return image_ids, hashes


def class_id_to_label(class_id: int, labels: dict[int, str]) -> str:
    return labels.get(int(class_id), str(class_id))


def raw_candidate_rows(
    heuristic_sources: dict[str, pd.DataFrame],
    legacy_images_dir: Path,
    class_labels: dict[int, str],
    weak_weight: float,
) -> pd.DataFrame:
    unknown = sorted(set(heuristic_sources).difference(SOURCE_POLICY))
    if unknown:
        raise ValueError(f"Unknown weak-image heuristic sources: {unknown}")

    frames: list[pd.DataFrame] = []
    for source in sorted(heuristic_sources):
        df = heuristic_sources[source].copy()
        require_columns(df, REQUIRED_HEURISTIC_COLUMNS, source)
        policy = SOURCE_POLICY[source]
        class_id = int(policy["class_id"])
        frame = pd.DataFrame({"image_id_ext": df["image_id_ext"].map(normalize_image_id_ext)})
        frame["class_id"] = class_id
        frame["label"] = class_id_to_label(class_id, class_labels)
        frame["weak_weight"] = float(weak_weight)
        frame["source"] = str(policy["source"])
        frame["source_dataset"] = "legacy_heuristics_images"
        frame["source_local_path"] = frame["image_id_ext"].map(
            lambda value: (legacy_images_dir / value).as_posix()
        )
        for col, default in (
            ("n_texts", 0),
            ("person_found", False),
            ("is_catalog", False),
            ("crop_area", 0.0),
            ("perform_top_microcat_prob", 0.0),
            ("perform_top_other_classes_prob", 0.0),
        ):
            frame[col] = df[col] if col in df.columns else default
        frames.append(frame)

    if not frames:
        raise ValueError("No heuristic sources were provided.")
    return pd.concat(frames, ignore_index=True)


def attach_image_metadata(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in candidates.to_dict(orient="records"):
        hash_sha256, width, height, error = inspect_image(Path(str(row["source_local_path"])))
        row["hash_sha256"] = hash_sha256
        row["width"] = width
        row["height"] = height
        row["image_error"] = error
        rows.append(row)
    return pd.DataFrame(rows)


def add_candidate_score(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    microcat = pd.to_numeric(result["perform_top_microcat_prob"], errors="coerce").fillna(0.0)
    other = pd.to_numeric(result["perform_top_other_classes_prob"], errors="coerce").fillna(0.0)
    crop_area = pd.to_numeric(result["crop_area"], errors="coerce").fillna(0.0)
    n_texts = pd.to_numeric(result["n_texts"], errors="coerce").fillna(0.0)
    result["candidate_score"] = microcat - other + (0.1 * crop_area) - (0.01 * n_texts)
    return result


def drop_with_mask(
    df: pd.DataFrame,
    mask: pd.Series,
    reason: str,
    drop_counts: dict[str, int],
) -> pd.DataFrame:
    drop_counts[reason] = int(mask.sum())
    return df.loc[~mask].copy().reset_index(drop=True)


def apply_gates(
    candidates: pd.DataFrame,
    manifest: pd.DataFrame,
    splits: dict[str, Any],
    max_texts: int,
    drop_person: bool,
    drop_catalog: bool,
    min_width: int,
    min_height: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    drop_counts: dict[str, int] = {}
    df = candidates.copy().reset_index(drop=True)

    df = drop_with_mask(df, df["image_id_ext"].eq(""), "empty_image_id_ext", drop_counts)
    df = attach_image_metadata(df)
    df = drop_with_mask(
        df,
        df["image_error"].eq("missing_image"),
        "missing_image",
        drop_counts,
    )
    df = drop_with_mask(
        df,
        df["image_error"].eq("corrupted_image"),
        "corrupted_image",
        drop_counts,
    )
    small_mask = (df["width"].fillna(0) < int(min_width)) | (
        df["height"].fillna(0) < int(min_height)
    )
    df = drop_with_mask(df, small_mask, "small_image", drop_counts)

    n_texts = pd.to_numeric(df["n_texts"], errors="coerce").fillna(0).astype(int)
    df = drop_with_mask(df, n_texts > int(max_texts), "max_texts", drop_counts)

    if drop_person:
        person_mask = df["person_found"].map(to_bool)
        df = drop_with_mask(df, person_mask, "person_found", drop_counts)
    else:
        drop_counts["person_found"] = 0

    if drop_catalog:
        catalog_mask = df["is_catalog"].map(to_bool)
        df = drop_with_mask(df, catalog_mask, "is_catalog", drop_counts)
    else:
        drop_counts["is_catalog"] = 0

    original_ids, original_hashes = leakage_keys(manifest, splits)
    df = drop_with_mask(
        df,
        df["image_id_ext"].isin(original_ids),
        "leakage_image_id_ext",
        drop_counts,
    )
    df = drop_with_mask(
        df,
        df["hash_sha256"].notna() & df["hash_sha256"].astype(str).isin(original_hashes),
        "leakage_hash_sha256",
        drop_counts,
    )

    return df.reset_index(drop=True), drop_counts


def drop_internal_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    ordered = add_candidate_score(df)
    ordered = ordered.sort_values(
        [
            "class_id",
            "candidate_score",
            "perform_top_microcat_prob",
            "crop_area",
            "source",
            "image_id_ext",
        ],
        ascending=[True, False, False, False, True, True],
        kind="stable",
    )
    duplicate_id_rows = int(ordered.duplicated(subset=["image_id_ext"], keep="first").sum())
    ordered = ordered.drop_duplicates(subset=["image_id_ext"], keep="first")
    hashable = ordered.loc[ordered["hash_sha256"].notna()].copy()
    no_hash = ordered.loc[ordered["hash_sha256"].isna()].copy()
    duplicate_hash_rows = int(hashable.duplicated(subset=["hash_sha256"], keep="first").sum())
    hashable = hashable.drop_duplicates(subset=["hash_sha256"], keep="first")
    deduped = pd.concat([hashable, no_hash], ignore_index=True)
    return deduped.reset_index(drop=True), {
        "image_id_ext": duplicate_id_rows,
        "hash_sha256": duplicate_hash_rows,
    }


def select_by_quota(df: pd.DataFrame, max_added_per_class: dict[int, int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for class_id, group in df.groupby("class_id", sort=True):
        quota = int(max_added_per_class.get(int(class_id), 0))
        if quota <= 0:
            continue
        selected = group.sort_values(
            [
                "candidate_score",
                "perform_top_microcat_prob",
                "crop_area",
                "image_id_ext",
            ],
            ascending=[False, False, False, True],
            kind="stable",
        ).head(quota)
        selected = selected.copy()
        selected["selected_rank"] = range(1, len(selected) + 1)
        frames.append(selected)
    if not frames:
        return pd.DataFrame(columns=[*df.columns, "selected_rank"])
    return pd.concat(frames, ignore_index=True)


def clear_output_dir(output_image_dir: Path) -> None:
    output_image_dir.mkdir(parents=True, exist_ok=True)
    for path in output_image_dir.iterdir():
        if path.is_file():
            path.unlink()


def copy_selected_images(
    selected: pd.DataFrame, output_image_dir: Path, clean_output: bool = True
) -> pd.DataFrame:
    if clean_output:
        clear_output_dir(output_image_dir)
    else:
        output_image_dir.mkdir(parents=True, exist_ok=True)

    result = selected.copy()
    selected_paths: list[str] = []
    for row in result.to_dict(orient="records"):
        image_id_ext = normalize_image_id_ext(row["image_id_ext"])
        destination = output_image_dir / image_id_ext
        shutil.copy2(row["source_local_path"], destination)
        selected_paths.append(destination.as_posix())
    result["selected_local_path"] = selected_paths
    return result


def finalize_manifest(selected: pd.DataFrame) -> pd.DataFrame:
    manifest = selected.copy()
    if manifest.empty:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    manifest["n_texts"] = pd.to_numeric(manifest["n_texts"], errors="coerce").fillna(0).astype(int)
    manifest["person_found"] = manifest["person_found"].map(to_bool)
    manifest["is_catalog"] = manifest["is_catalog"].map(to_bool)
    manifest["crop_area"] = pd.to_numeric(manifest["crop_area"], errors="coerce").fillna(0.0)
    manifest["perform_top_microcat_prob"] = pd.to_numeric(
        manifest["perform_top_microcat_prob"], errors="coerce"
    ).fillna(0.0)
    manifest["perform_top_other_classes_prob"] = pd.to_numeric(
        manifest["perform_top_other_classes_prob"], errors="coerce"
    ).fillna(0.0)
    manifest["width"] = pd.to_numeric(manifest["width"], errors="coerce").astype(int)
    manifest["height"] = pd.to_numeric(manifest["height"], errors="coerce").astype(int)
    manifest["candidate_score"] = pd.to_numeric(manifest["candidate_score"], errors="coerce")
    manifest["leakage_checked"] = True
    manifest["selected_rank"] = pd.to_numeric(
        manifest["selected_rank"], errors="coerce"
    ).astype(int)
    return manifest[MANIFEST_COLUMNS].sort_values(
        ["class_id", "selected_rank"], kind="stable"
    ).reset_index(drop=True)


def summarize_selected(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {}
    counts = df.groupby("class_id").size().sort_index()
    return {str(int(class_id)): int(rows) for class_id, rows in counts.items()}


def build_weak_images(
    heuristic_sources: dict[str, pd.DataFrame],
    legacy_images_dir: Path,
    manifest: pd.DataFrame,
    splits: dict[str, Any],
    class_labels: dict[int, str],
    output_image_dir: Path,
    max_added_per_class: dict[int, int],
    weak_weight: float = DEFAULT_WEAK_WEIGHT,
    max_texts: int = 0,
    drop_catalog: bool = True,
    drop_person: bool = True,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
    clean_output: bool = True,
) -> WeakImagesResult:
    raw = raw_candidate_rows(
        heuristic_sources=heuristic_sources,
        legacy_images_dir=legacy_images_dir,
        class_labels=class_labels,
        weak_weight=weak_weight,
    )
    gated, drop_counts = apply_gates(
        candidates=raw,
        manifest=manifest,
        splits=splits,
        max_texts=max_texts,
        drop_person=drop_person,
        drop_catalog=drop_catalog,
        min_width=min_width,
        min_height=min_height,
    )
    deduped, duplicate_counts = drop_internal_duplicates(gated)
    selected = select_by_quota(deduped, max_added_per_class=max_added_per_class)
    copied = copy_selected_images(selected, output_image_dir=output_image_dir, clean_output=clean_output)
    final = finalize_manifest(copied)

    quota_dropped = int(len(deduped) - len(selected))
    audit = {
        "version": VERSION,
        "policy": {
            "weak_weight": float(weak_weight),
            "max_texts": int(max_texts),
            "drop_catalog": bool(drop_catalog),
            "drop_person": bool(drop_person),
            "min_width": int(min_width),
            "min_height": int(min_height),
            "max_added_per_class": {str(k): int(v) for k, v in max_added_per_class.items()},
            "candidate_score": "perform_top_microcat_prob - perform_top_other_classes_prob + 0.1 * crop_area - 0.01 * n_texts",
        },
        "input_rows_by_source": {
            source: int(len(df)) for source, df in sorted(heuristic_sources.items())
        },
        "drop_counts": {
            **drop_counts,
            "duplicate_image_id_ext": int(duplicate_counts["image_id_ext"]),
            "duplicate_hash_sha256": int(duplicate_counts["hash_sha256"]),
            "quota": quota_dropped,
        },
        "rows_after_gates": int(len(gated)),
        "rows_after_dedup": int(len(deduped)),
        "selected_rows": int(len(final)),
        "selected_rows_by_class": summarize_selected(final),
        "output_image_dir": output_image_dir.as_posix(),
    }
    return WeakImagesResult(manifest=final, audit=audit)


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


def build_download_report(audit: dict[str, Any]) -> str:
    input_rows = [[k, v] for k, v in sorted(audit["input_rows_by_source"].items())]
    drop_rows = [[k, v] for k, v in sorted(audit["drop_counts"].items())]
    selected_rows = [[k, v] for k, v in sorted(audit["selected_rows_by_class"].items())]

    return "\n".join(
        [
            "# Weak Images Download Report: weak_images_v1",
            "",
            "## Policy",
            f"- version: `{audit['version']}`",
            f"- weak_weight: `{audit['policy']['weak_weight']}`",
            f"- max_texts: `{audit['policy']['max_texts']}`",
            f"- drop_catalog: `{audit['policy']['drop_catalog']}`",
            f"- drop_person: `{audit['policy']['drop_person']}`",
            f"- min_size: `{audit['policy']['min_width']}x{audit['policy']['min_height']}`",
            f"- max_added_per_class: `{audit['policy']['max_added_per_class']}`",
            f"- candidate_score: `{audit['policy']['candidate_score']}`",
            "- leakage_checked: `image_id_ext` and `sha256` against train/val/test manifests/splits",
            "",
            "## Input rows by source",
            markdown_table(["source", "rows"], input_rows),
            "",
            "## Drop counts by reason",
            markdown_table(["reason", "rows"], drop_rows),
            "",
            "## Selection",
            f"- rows_after_gates: `{audit['rows_after_gates']}`",
            f"- rows_after_dedup: `{audit['rows_after_dedup']}`",
            f"- selected_rows: `{audit['selected_rows']}`",
            f"- output_image_dir: `{audit['output_image_dir']}`",
            "",
            "## Selected rows by class",
            markdown_table(["class_id", "rows"], selected_rows),
            "",
        ]
    )


def write_artifacts(df: pd.DataFrame, report_text: str, csv_path: Path, report_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    report_path.write_text(report_text, encoding="utf-8")


def parse_max_added_per_class(values: list[str] | None) -> dict[int, int]:
    if not values:
        return dict(DEFAULT_MAX_ADDED_PER_CLASS)
    quotas: dict[int, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Quota must use CLASS_ID=COUNT format, got {value!r}")
        class_text, count_text = value.split("=", 1)
        quotas[int(class_text)] = int(count_text)
    return quotas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weak_images_v1 curated image subset.")
    parser.add_argument("--heuristic-csv", type=Path, action="append", dest="heuristic_csvs")
    parser.add_argument("--legacy-images-dir", type=Path, default=DEFAULT_LEGACY_IMAGES_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--splits-json", type=Path, default=DEFAULT_SPLITS_JSON)
    parser.add_argument("--class-mapping", type=Path, default=DEFAULT_CLASS_MAPPING)
    parser.add_argument("--output-image-dir", type=Path, default=DEFAULT_OUTPUT_IMAGE_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument(
        "--max-added-per-class",
        nargs="+",
        default=[f"{k}={v}" for k, v in DEFAULT_MAX_ADDED_PER_CLASS.items()],
    )
    parser.add_argument("--weak-weight", type=float, default=DEFAULT_WEAK_WEIGHT)
    parser.add_argument("--max-texts", type=int, default=0)
    parser.add_argument("--drop-catalog", action="store_true")
    parser.add_argument("--drop-person", action="store_true")
    parser.add_argument("--min-width", type=int, default=DEFAULT_MIN_WIDTH)
    parser.add_argument("--min-height", type=int, default=DEFAULT_MIN_HEIGHT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heuristic_csvs = args.heuristic_csvs or list(DEFAULT_HEURISTIC_CSVS)
    result = build_weak_images(
        heuristic_sources=load_heuristic_sources(heuristic_csvs),
        legacy_images_dir=args.legacy_images_dir,
        manifest=load_manifest(args.manifest),
        splits=load_splits(args.splits_json),
        class_labels=load_class_mapping(args.class_mapping),
        output_image_dir=args.output_image_dir,
        max_added_per_class=parse_max_added_per_class(args.max_added_per_class),
        weak_weight=args.weak_weight,
        max_texts=args.max_texts,
        drop_catalog=args.drop_catalog,
        drop_person=args.drop_person,
        min_width=args.min_width,
        min_height=args.min_height,
        clean_output=True,
    )
    report_text = build_download_report(result.audit)
    write_artifacts(result.manifest, report_text, args.output_csv, args.report_md)
    print(f"weak_images_manifest -> {args.output_csv}")
    print(f"weak_images_report -> {args.report_md}")


if __name__ == "__main__":
    main()
