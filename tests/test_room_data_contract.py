from __future__ import annotations

import json

import pytest

from src.utils.room_data_contract import (
    ClassSchema,
    build_fold_frames,
    image_id_with_extension,
    load_splits,
    normalize_image_id,
)


def test_image_id_normalization_contract():
    assert normalize_image_id(" 12.0 ") == "12"
    assert image_id_with_extension(12.0) == "12.jpg"
    assert image_id_with_extension("x.png") == "x.png"
    assert image_id_with_extension(None) == ""


def test_load_splits_validates_summary_and_shadow_leakage(tmp_path):
    splits_path = tmp_path / "splits.json"
    payload = {
        "version": "splits_v1",
        "folds": [
            {"fold": 0, "records": [{"image_id_ext": "a", "result": 0}]},
            {"fold": 1, "records": [{"image_id_ext": "b.jpg", "result": 1}]},
        ],
        "shadow_holdout": {"records": [{"image_id_ext": "s", "result": 0}]},
        "summary": {
            "train_pool_rows_after_filters": 2,
            "shadow_holdout_rows_after_filters": 1,
        },
    }
    splits_path.write_text(json.dumps(payload), encoding="utf-8")

    splits = load_splits(splits_path)
    train_df, valid_df = build_fold_frames(splits, 0)

    assert train_df["image_id_ext"].tolist() == ["b.jpg"]
    assert valid_df["image_id_ext"].tolist() == ["a.jpg"]

    payload["shadow_holdout"]["records"] = [{"image_id_ext": "a.jpg", "result": 0}]
    splits_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Shadow holdout leaks"):
        load_splits(splits_path)


def test_class_schema_loads_valid_ids_and_labels(tmp_path):
    path = tmp_path / "class_mapping.yaml"
    path.write_text(
        "prediction:\n  valid_class_ids: [0, 2]\nid_to_label:\n  0: kitchen\n  2: room\n",
        encoding="utf-8",
    )

    schema = ClassSchema.from_yaml(path)

    assert schema.valid_class_ids == [0, 2]
    assert schema.id_to_label == {0: "kitchen", 2: "room"}
