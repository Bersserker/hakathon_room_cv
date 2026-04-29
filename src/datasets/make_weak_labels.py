import hashlib
from pathlib import Path

import pandas as pd


# =========================
# Paths
# =========================

PROJECT_ROOT = Path.cwd()

# Если запускаешь из src/, тогда раскомментируй:
# PROJECT_ROOT = Path.cwd().parent

DATA_RAW = PROJECT_ROOT / "data" / "raw" 
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

HEURISTICS_IMAGES_DIR = DATA_RAW / "heuristics_images" / "heuristics_images"

OUTPUT_PARQUET = DATA_PROCESSED / "weak_labels_v1.parquet"
OUTPUT_REPORT = REPORTS_DIR / "weak_labels_audit.md"


# =========================
# Config
# =========================

HEURISTICS_CONFIG = {
    "heuristics_cabinet": {
        "csv": DATA_RAW / "heuristics_cabinet.csv",
        "class_id": 5,
    },
    "heuristics_detskaya": {
        "csv": DATA_RAW / "heuristics_detskaya.csv",
        "class_id": 6,
    },
    "heuristics_dressing_room": {
        "csv": DATA_RAW / "heuristics_dressing_room.csv",
        "class_id": 11,
    },
}

WEAK_WEIGHT = 0.3


# =========================
# Helpers
# =========================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def ensure_jpg(name: str) -> str:
    name = str(name)

    if name.lower().endswith((".jpg", ".jpeg", ".png")):
        return name

    return name + ".jpg"


def find_image_id_column(df: pd.DataFrame) -> str:
    candidates = ["image_id_ext", "image_id", "filename", "file_name", "name"]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Не нашёл колонку с именем изображения. "
        f"Есть колонки: {list(df.columns)}"
    )


def add_image_id_ext(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "image_id_ext" not in df.columns:
        image_col = find_image_id_column(df)
        df["image_id_ext"] = df[image_col].astype(str).apply(ensure_jpg)
    else:
        df["image_id_ext"] = df["image_id_ext"].astype(str).apply(ensure_jpg)

    return df


def add_hash(df: pd.DataFrame, images_dir: Path) -> pd.DataFrame:
    df = df.copy()

    df["path"] = df["image_id_ext"].apply(lambda x: images_dir / x)
    df["file_exists"] = df["path"].apply(lambda p: p.exists())

    df["hash"] = None

    existing_mask = df["file_exists"]

    df.loc[existing_mask, "hash"] = df.loc[existing_mask, "path"].apply(
        sha256_file
    )

    # важно для parquet
    df["path"] = df["path"].astype(str)

    return df


# =========================
# Main pipeline
# =========================

def build_weak_labels():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Load train ----------
    train_df = pd.read_csv(DATA_RAW / "train_df.csv")
    train_df = add_image_id_ext(train_df)

    # Если в train уже есть hash — используем его.
    # Если нет — пока сравниваем только по image_id_ext.
    train_has_hash = "hash" in train_df.columns

    # ---------- Load heuristics ----------
    weak_parts = []

    for source_name, cfg in HEURISTICS_CONFIG.items():
        df = pd.read_csv(cfg["csv"])
        df = add_image_id_ext(df)

        df["class_id"] = cfg["class_id"]
        df["source"] = source_name
        df["source_flag"] = "weak"
        df["weak_weight"] = WEAK_WEIGHT

        weak_parts.append(df)

    weak_df = pd.concat(weak_parts, ignore_index=True)

    rows_before = len(weak_df)

    # ---------- Add hash for weak ----------
    weak_df = add_hash(weak_df, HEURISTICS_IMAGES_DIR)

    missing_files_count = (~weak_df["file_exists"]).sum()

    # ---------- Check intersections with train ----------
    weak_df["is_in_train_by_image_id_ext"] = weak_df["image_id_ext"].isin(
        train_df["image_id_ext"]
    )

    if train_has_hash:
        weak_df["is_in_train_by_hash"] = weak_df["hash"].isin(train_df["hash"])
    else:
        weak_df["is_in_train_by_hash"] = False

    weak_df["is_in_train"] = (
        weak_df["is_in_train_by_image_id_ext"]
        | weak_df["is_in_train_by_hash"]
    )

    duplicates_with_train = weak_df["is_in_train"].sum()

    # ---------- Remove duplicates with train ----------
    weak_df_clean = weak_df[~weak_df["is_in_train"]].copy()

    # ---------- Remove duplicates inside weak itself ----------
    duplicates_inside_weak_by_id = weak_df_clean.duplicated("image_id_ext").sum()

    weak_df_clean = weak_df_clean.drop_duplicates(
        subset=["image_id_ext"],
        keep="first"
    )

    if "hash" in weak_df_clean.columns:
        hash_mask = weak_df_clean["hash"].notna()
        duplicates_inside_weak_by_hash = weak_df_clean.loc[hash_mask].duplicated("hash").sum()

        weak_df_clean = pd.concat(
            [
                weak_df_clean.loc[~hash_mask],
                weak_df_clean.loc[hash_mask].drop_duplicates("hash", keep="first"),
            ],
            ignore_index=True,
        )
    else:
        duplicates_inside_weak_by_hash = 0

    rows_after = len(weak_df_clean)

    # ---------- Final columns ----------
    final_columns = [
        "image_id_ext",
        "class_id",
        "weak_weight",
        "source",
        "source_flag",
        "is_in_train",
        "is_in_train_by_image_id_ext",
        "is_in_train_by_hash",
        "hash",
        "path",
        "file_exists",
    ]

    final_columns = [col for col in final_columns if col in weak_df_clean.columns]

    weak_df_clean = weak_df_clean[final_columns]

    # ---------- DOD checks ----------
    required_columns = [
        "image_id_ext",
        "class_id",
        "weak_weight",
        "source",
        "source_flag",
        "is_in_train",
    ]

    for col in required_columns:
        assert col in weak_df_clean.columns, f"Нет обязательной колонки: {col}"

    assert weak_df_clean["is_in_train"].sum() == 0, "Остались пересечения с train"
    assert weak_df_clean.duplicated("image_id_ext").sum() == 0, "Остались дубли по image_id_ext"

    # ---------- Save parquet ----------
    if "path" in weak_df_clean.columns:
        weak_df_clean["path"] = weak_df_clean["path"].astype(str)

    weak_df_clean.to_parquet(OUTPUT_PARQUET, index=False)

    # ---------- Audit report ----------
    class_distribution = weak_df_clean["class_id"].value_counts().sort_index()
    source_distribution = weak_df_clean["source"].value_counts()

    class_distribution_str = class_distribution.to_string()
    source_distribution_str = source_distribution.to_string()

    report = f"""# Weak labels audit

## Output

- Parquet: `{OUTPUT_PARQUET}`
- Report: `{OUTPUT_REPORT}`

## Sources

"""

    for source_name, cfg in HEURISTICS_CONFIG.items():
        report += f"- `{cfg['csv'].name}` -> class_id `{cfg['class_id']}`\n"

    report += f"""

## Counts

- Rows before deduplication: `{rows_before}`
- Missing image files: `{missing_files_count}`
- Duplicates with train: `{duplicates_with_train}`
- Duplicates inside weak by image_id_ext: `{duplicates_inside_weak_by_id}`
- Duplicates inside weak by hash: `{duplicates_inside_weak_by_hash}`
- Rows after cleaning: `{rows_after}`

## Class distribution

{class_distribution}

## Source distribution

{source_distribution}

## DOD checklist

- [x] Manifest contains `image_id_ext`
- [x] Manifest contains `class_id`
- [x] Manifest contains `weak_weight`
- [x] Manifest contains `source`
- [x] Manifest contains `source_flag`
- [x] Manifest contains `is_in_train`
- [x] Duplicates with train removed
- [x] Duplicates by `image_id_ext` removed
"""

    OUTPUT_REPORT.write_text(report, encoding="utf-8")

    print(f"Saved: {OUTPUT_PARQUET}")
    print(f"Saved: {OUTPUT_REPORT}")
    print(f"Rows before: {rows_before}")
    print(f"Rows after: {rows_after}")
    print(f"Duplicates with train: {duplicates_with_train}")
    print(f"Missing files: {missing_files_count}")


if __name__ == "__main__":
    build_weak_labels()