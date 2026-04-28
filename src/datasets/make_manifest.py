import argparse
import hashlib
import shutil
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError


DEFAULT_BASE_DIRS = (
    Path("data/raw/room_type"),
    Path("data/raw"),
)

SPLITS = {
    "train": ("train_df.csv", "train_images"),
    "val": ("val_df.csv", "val_images"),
    "test": ("test_df.csv", "test_images"),
}

OUT_PATH = Path("data/processed/data_manifest.parquet")
REPORT_PATH = Path("reports/data_integrity.md")
MISSING_REPORT_PATH = Path("reports/missing_files.csv")
CORRUPTED_REPORT_PATH = Path("reports/corrupted_files.csv")
MANIFEST_COLUMNS = [
    "image_id_ext",
    "split",
    "local_path",
    "hash_sha256",
    "width",
    "height",
    "status",
    "error",
]
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Root directory with split CSVs and image folders.",
    )
    return parser.parse_args()


def resolve_base_dir(base_dir: Path | None) -> Path:
    if base_dir is not None:
        return base_dir

    for candidate in DEFAULT_BASE_DIRS:
        if all((candidate / csv_name).exists() for csv_name, _ in SPLITS.values()):
            return candidate

    checked = ", ".join(str(path) for path in DEFAULT_BASE_DIRS)
    raise FileNotFoundError(f"Could not find dataset root. Checked: {checked}")


def find_split_dir(base_dir: Path, folder_name: str) -> Path:
    candidates = (
        base_dir / folder_name / folder_name,
        base_dir / folder_name,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def unpack_split_archive(base_dir: Path, folder_name: str) -> Path:
    image_dir = find_split_dir(base_dir, folder_name)
    if image_dir.is_dir():
        return image_dir

    archive_paths = [base_dir / f"{folder_name}{suffix}" for suffix in ARCHIVE_SUFFIXES]
    for archive_path in archive_paths:
        if not archive_path.is_file():
            continue
        image_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(archive_path), str(image_dir.parent))
        break

    return find_split_dir(base_dir, folder_name)


def normalize_image_name(image_id: str) -> str:
    image_name = str(image_id)
    return image_name if Path(image_name).suffix else f"{image_name}.jpg"


def url_file_name(image_url: str | None) -> str | None:
    if not image_url or pd.isna(image_url):
        return None
    url_path = urlparse(str(image_url)).path
    file_name = Path(url_path).name
    return file_name or None


def download_image(path: Path, image_url: str | None, expected_name: str) -> str | None:
    if not image_url or pd.isna(image_url):
        return None

    source_name = url_file_name(image_url)
    if source_name != expected_name:
        return f"url filename mismatch: expected {expected_name}, got {source_name}"

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.part")

    try:
        with requests.get(image_url, timeout=60, stream=True) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as file_obj:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_obj.write(chunk)
        tmp_path.replace(path)
        return None
    except requests.RequestException as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return f"download failed: {exc}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_image(path: Path, image_id: str, split: str, image_url: str | None = None):
    result = {
        "image_id_ext": image_id,
        "split": split,
        "local_path": str(path),
        "hash_sha256": None,
        "width": None,
        "height": None,
        "status": "ok",
        "error": None,
    }

    if not path.exists():
        download_error = download_image(path, image_url, image_id)
    else:
        download_error = None

    if not path.exists():
        result["status"] = "missing"
        result["error"] = download_error or "file not found"
        return result

    try:
        result["hash_sha256"] = sha256_file(path)

        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            result["width"], result["height"] = img.size

    except (UnidentifiedImageError, OSError) as exc:
        result["status"] = "corrupted"
        result["error"] = str(exc)

    return result


def process_split(split: str, df: pd.DataFrame, base_dir: Path, folder_name: str):
    image_col = "image_id_ext"
    if image_col not in df.columns:
        raise ValueError(f"Missing required column: {image_col}")

    image_url_col = "image" if "image" in df.columns else None
    folder = unpack_split_archive(base_dir, folder_name)

    rows = []
    for row in df.to_dict("records"):
        img_name = normalize_image_name(row[image_col])
        path = folder / img_name
        rows.append(check_image(path, img_name, split, row.get(image_url_col)))

    return rows


def build_report(manifest: pd.DataFrame, max_rows: int = 50) -> str:
    missing_df = manifest[manifest["status"] == "missing"]
    corrupted_df = manifest[manifest["status"] == "corrupted"]
    ok_df = manifest[manifest["status"] == "ok"]

    def format_block(df: pd.DataFrame, title: str) -> str:
        if len(df) == 0:
            return f"\n## {title}\nНет\n"

        sample = df.head(max_rows)
        lines = "\n".join(
            sample[["image_id_ext", "local_path", "error"]]
            .fillna("")
            .astype(str)
            .apply(
                lambda row: f"- {row['image_id_ext']} | {row['local_path']} | {row['error']}",
                axis=1,
            )
        )

        return f"\n## {title} ({len(df)})\n\n{lines}\n"

    total = len(manifest)
    missing_ratio = (len(missing_df) / total) if total else 0
    corrupted_ratio = (len(corrupted_df) / total) if total else 0

    report = f"""# Data integrity report

Total: {total}
OK: {len(ok_df)}
Missing: {len(missing_df)} ({missing_ratio:.2%})
Corrupted: {len(corrupted_df)} ({corrupted_ratio:.2%})
"""

    report += format_block(missing_df, "Missing")
    report += format_block(corrupted_df, "Corrupted")
    return report


def main():
    args = parse_args()
    base_dir = resolve_base_dir(args.base_dir)

    all_rows = []
    expected_total = 0
    processed_splits = set()

    for split, (csv_name, folder_name) in SPLITS.items():
        print(f"Processing {split}...")

        csv_path = base_dir / csv_name
        df = pd.read_csv(csv_path)

        expected_total += len(df)
        processed_splits.add(split)
        rows = process_split(split, df, base_dir, folder_name)
        all_rows.extend(rows)

    expected_splits = set(SPLITS.keys())
    if processed_splits != expected_splits:
        raise ValueError(f"Split mismatch! Expected {expected_splits}, got {processed_splits}")

    if len(all_rows) != expected_total:
        raise ValueError(f"Row count mismatch! Expected {expected_total}, got {len(all_rows)}")

    manifest = pd.DataFrame(all_rows, columns=MANIFEST_COLUMNS)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest.to_parquet(OUT_PATH, index=False, engine="pyarrow")

    missing_df = manifest[manifest["status"] == "missing"]
    corrupted_df = manifest[manifest["status"] == "corrupted"]

    missing_df.to_csv(MISSING_REPORT_PATH, index=False)
    corrupted_df.to_csv(CORRUPTED_REPORT_PATH, index=False)

    report = build_report(manifest)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("\n=== DONE ===")
    print(f"Base dir: {base_dir}")
    print(f"Expected rows: {expected_total}")
    print(f"Manifest rows: {len(manifest)}")
    print(f"Splits: {sorted(processed_splits)}")
    print(f"Missing: {len(missing_df)}")
    print(f"Corrupted: {len(corrupted_df)}")
    print(f"Manifest -> {OUT_PATH}")
    print(f"Report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
