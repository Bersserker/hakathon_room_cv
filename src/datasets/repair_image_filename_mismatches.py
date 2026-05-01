import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError


SPLITS = {
    "train": ("train_df.csv", "train_images"),
    "val": ("val_df.csv", "val_images"),
    "test": ("test_df.csv", "test_images"),
}
DEFAULT_REPORT_PATH = Path("reports/image_filename_mismatch_repair.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize images under image_id_ext names when CSV image URLs point to "
            "a different jpg filename. This keeps train/val/test complete before manifest/splits."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits", nargs="+", choices=sorted(SPLITS), default=sorted(SPLITS))
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    return parser.parse_args()


def normalize_image_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text if Path(text).suffix else f"{text}.jpg"


def url_file_name(image_url: object) -> str | None:
    if image_url is None or pd.isna(image_url):
        return None
    file_name = Path(urlparse(str(image_url)).path).name
    return file_name or None


def find_split_dir(base_dir: Path, folder_name: str) -> Path:
    for candidate in (base_dir / folder_name / folder_name, base_dir / folder_name):
        if candidate.is_dir():
            return candidate
    return base_dir / folder_name / folder_name


def verify_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


# Keep the downloaded bytes but store them under the expected image_id_ext filename.
def download_to_path(image_url: str, path: Path) -> str | None:
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
        return str(exc)


def write_from_source_or_url(
    source_path: Path,
    image_url: object,
    expected_path: Path,
    download: bool,
) -> tuple[str, str | None]:
    if source_path.exists():
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, expected_path)
        return "created_from_local", None

    if download and image_url is not None and not pd.isna(image_url):
        error = download_to_path(str(image_url), expected_path)
        if error is None:
            return "created_from_url", None
        return "download_failed", error

    return "source_missing", str(source_path)


def repair_split(
    base_dir: Path,
    split: str,
    *,
    download: bool = True,
    overwrite: bool = False,
) -> tuple[Counter[str], list[dict[str, str]]]:
    csv_name, folder_name = SPLITS[split]
    csv_path = base_dir / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, dtype={"image_id_ext": "string", "image": "string"})
    if "image_id_ext" not in df.columns:
        raise ValueError(f"{csv_path} missing required column image_id_ext")
    if "image" not in df.columns:
        raise ValueError(f"{csv_path} missing required column image")

    image_dir = find_split_dir(base_dir, folder_name)
    stats: Counter[str] = Counter()
    rows: list[dict[str, str]] = []

    for row in df.to_dict("records"):
        expected_name = normalize_image_name(row.get("image_id_ext"))
        source_name = url_file_name(row.get("image"))
        if not expected_name or not source_name or expected_name == source_name:
            stats["not_mismatched"] += 1
            continue

        stats["mismatched"] += 1
        expected_path = image_dir / expected_name
        source_path = image_dir / source_name

        if expected_path.exists() and not overwrite:
            try:
                verify_image(expected_path)
                status = "already_present"
                error = None
            except (UnidentifiedImageError, OSError) as exc:
                status = "invalid_existing"
                error = str(exc)
        else:
            status, error = write_from_source_or_url(
                source_path=source_path,
                image_url=row.get("image"),
                expected_path=expected_path,
                download=download,
            )

        if status.startswith("created") or status == "already_present":
            try:
                verify_image(expected_path)
            except (UnidentifiedImageError, OSError) as exc:
                status = "invalid_written"
                error = str(exc)

        stats[status] += 1
        rows.append(
            {
                "split": split,
                "image_id_ext": expected_name,
                "url_file_name": source_name,
                "expected_path": str(expected_path),
                "source_path": str(source_path),
                "status": status,
                "error": "" if error is None else str(error),
            }
        )

    return stats, rows


def repair_dataset(
    base_dir: Path,
    split_names: list[str],
    *,
    download: bool = True,
    overwrite: bool = False,
) -> tuple[Counter[str], list[dict[str, str]]]:
    total: Counter[str] = Counter()
    all_rows: list[dict[str, str]] = []
    for split in split_names:
        stats, rows = repair_split(base_dir, split, download=download, overwrite=overwrite)
        total.update({f"{split}.{key}": value for key, value in stats.items()})
        all_rows.extend(rows)
    return total, all_rows


def main() -> int:
    args = parse_args()
    stats, rows = repair_dataset(
        base_dir=args.base_dir,
        split_names=args.splits,
        download=not args.no_download,
        overwrite=args.overwrite,
    )

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.report_path, index=False)

    print("Image filename mismatch repair:")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"Report -> {args.report_path}")

    failure_statuses = {"source_missing", "download_failed", "invalid_existing", "invalid_written"}
    failures = [row for row in rows if row["status"] in failure_statuses]
    if failures and not args.no_strict:
        print(f"Unresolved mismatches: {len(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
