from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.inference.validate_submission import validate_submission


def write_class_mapping(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "num_classes": 3,
                "prediction": {"valid_class_ids": [0, 1, 2]},
                "id_to_label": {0: "a", 1: "b", 2: "c"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def write_test_csv(path: Path) -> None:
    path.write_text("image_id_ext\nimg1\nimg2\nimg3\n", encoding="utf-8")


def test_validate_submission_accepts_valid_csv(tmp_path):
    mapping = tmp_path / "class_mapping.yaml"
    test_csv = tmp_path / "test.csv"
    submission = tmp_path / "submission.csv"
    write_class_mapping(mapping)
    write_test_csv(test_csv)
    submission.write_text("image_id_ext,Predicted\nimg1,0\nimg2,1\nimg3,2\n", encoding="utf-8")

    result = validate_submission(submission, test_csv, mapping)

    assert result["status"] == "ok"
    assert result["rows"] == 3
    assert result["valid_class_ids"] == [0, 1, 2]


@pytest.mark.parametrize(
    ("csv_text", "message"),
    [
        ("image_id_ext,Predicted\nimg1,0\nimg1,1\nimg3,2\n", "duplicate"),
        ("image_id_ext,Predicted\nimg1,0\nimg2,1\n", "row count"),
        ("image_id_ext,Predicted\nimg1,0\nimg2,9\nimg3,2\n", "outside schema"),
        ("image_id_ext,Predicted\nimg1,0\nimg2,bad\nimg3,2\n", "numeric integer"),
        ("image_id_ext,pred\nimg1,0\nimg2,1\nimg3,2\n", "columns"),
    ],
)
def test_validate_submission_rejects_invalid_csv(tmp_path, csv_text, message):
    mapping = tmp_path / "class_mapping.yaml"
    test_csv = tmp_path / "test.csv"
    submission = tmp_path / "submission.csv"
    write_class_mapping(mapping)
    write_test_csv(test_csv)
    submission.write_text(csv_text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_submission(submission, test_csv, mapping)
