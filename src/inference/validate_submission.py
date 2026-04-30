from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

EXPECTED_COLUMNS = ["image_id_ext", "Predicted"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate room-classification submission CSV.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, default=Path("data/raw/test_df.csv"))
    parser.add_argument(
        "--class-mapping",
        type=Path,
        default=Path("configs/data/class_mapping.yaml"),
    )
    return parser.parse_args()


def normalize_image_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_valid_class_ids(class_mapping_path: Path) -> list[int]:
    with class_mapping_path.open("r", encoding="utf-8") as file:
        mapping = yaml.safe_load(file) or {}

    prediction = mapping.get("prediction", {}) if isinstance(mapping, dict) else {}
    if "valid_class_ids" in prediction:
        return sorted(int(value) for value in prediction["valid_class_ids"])

    if "id_to_label" in mapping:
        return sorted(int(value) for value in mapping["id_to_label"].keys())

    if "num_classes" in mapping:
        return list(range(int(mapping["num_classes"])))

    raise ValueError(f"{class_mapping_path} does not define valid class ids")


def read_csv_with_id(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"image_id_ext": "string"})


def validate_submission(
    submission_path: Path,
    test_csv_path: Path,
    class_mapping_path: Path,
) -> dict[str, Any]:
    if not submission_path.exists():
        raise ValueError(f"submission not found: {submission_path}")
    if not test_csv_path.exists():
        raise ValueError(f"test csv not found: {test_csv_path}")
    if not class_mapping_path.exists():
        raise ValueError(f"class mapping not found: {class_mapping_path}")

    submission = read_csv_with_id(submission_path)
    test_df = read_csv_with_id(test_csv_path)
    valid_class_ids = load_valid_class_ids(class_mapping_path)
    valid_class_set = set(valid_class_ids)

    if submission.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(
            f"submission columns must be exactly {EXPECTED_COLUMNS}, got {submission.columns.tolist()}"
        )
    if "image_id_ext" not in test_df.columns:
        raise ValueError(f"{test_csv_path} missing required column image_id_ext")

    if len(submission) != len(test_df):
        raise ValueError(f"row count mismatch: submission={len(submission)} test={len(test_df)}")

    if submission["image_id_ext"].isna().any():
        raise ValueError("submission has NaN image_id_ext")
    if submission["Predicted"].isna().any():
        raise ValueError("submission has NaN Predicted")

    submission_ids = submission["image_id_ext"].map(normalize_image_id)
    test_ids = test_df["image_id_ext"].map(normalize_image_id)

    duplicate_ids = submission_ids[submission_ids.duplicated()].unique().tolist()
    if duplicate_ids:
        raise ValueError(
            f"submission has duplicate image_id_ext values, sample={duplicate_ids[:10]}"
        )

    missing_ids = sorted(set(test_ids).difference(submission_ids))
    extra_ids = sorted(set(submission_ids).difference(test_ids))
    if missing_ids or extra_ids:
        raise ValueError(
            "submission ids must match test ids exactly: "
            f"missing_count={len(missing_ids)} extra_count={len(extra_ids)} "
            f"missing_sample={missing_ids[:10]} extra_sample={extra_ids[:10]}"
        )

    predicted_numeric = pd.to_numeric(submission["Predicted"], errors="coerce")
    if predicted_numeric.isna().any():
        bad_values = submission.loc[predicted_numeric.isna(), "Predicted"].head(10).tolist()
        raise ValueError(f"Predicted must be numeric integer classes, bad_sample={bad_values}")

    non_integer_mask = (predicted_numeric % 1) != 0
    if non_integer_mask.any():
        bad_values = submission.loc[non_integer_mask, "Predicted"].head(10).tolist()
        raise ValueError(f"Predicted must contain integer classes, bad_sample={bad_values}")

    predicted = predicted_numeric.astype(int)
    invalid_classes = sorted(set(predicted).difference(valid_class_set))
    if invalid_classes:
        raise ValueError(
            f"Predicted contains classes outside schema: {invalid_classes}; "
            f"valid_range={valid_class_ids[0]}..{valid_class_ids[-1]}"
        )

    checksum = file_sha256(submission_path)
    return {
        "status": "ok",
        "rows": int(len(submission)),
        "unique_ids": int(submission_ids.nunique()),
        "class_min": int(predicted.min()) if len(predicted) else None,
        "class_max": int(predicted.max()) if len(predicted) else None,
        "valid_class_ids": valid_class_ids,
        "sha256": checksum,
    }


def main() -> None:
    args = parse_args()
    result = validate_submission(args.submission, args.test_csv, args.class_mapping)
    print("Submission validation: OK")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
