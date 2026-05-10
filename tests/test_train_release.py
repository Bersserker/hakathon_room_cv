from __future__ import annotations

import pandas as pd
import pytest

from src.training.train_release import (
    build_release_split,
    fill_missing_local_paths,
    release_checkpoint_path,
)


def test_build_release_split_is_group_safe_by_item_and_hash():
    pool = pd.DataFrame(
        {
            "image_id_ext": [f"img_{idx}.jpg" for idx in range(20)],
            "item_id": list(range(18)) + [100, 101],
            "result": [0, 1] * 10,
            "label": ["a", "b"] * 10,
            "ratio": [1.0] * 20,
            "source_dataset": ["train_df"] * 10 + ["val_df"] * 10,
            "local_path": [f"/tmp/img_{idx}.jpg" for idx in range(20)],
            "content_hash": [None] * 18 + ["same_hash", "same_hash"],
        }
    )
    cfg = {
        "data": {"label_col": "result"},
        "train": {"seed": 42},
        "release_training": {"val_fraction": 0.2, "validation_fold_index": 0},
    }

    train_df, valid_df, summary = build_release_split(pool, cfg)

    assert summary["n_splits"] == 5
    assert len(train_df) + len(valid_df) == len(pool)
    assert set(train_df["split_group"]).isdisjoint(set(valid_df["split_group"]))

    same_hash_splits = pd.concat([train_df, valid_df]).loc[
        lambda df: df["content_hash"] == "same_hash", "release_split"
    ]
    assert same_hash_splits.nunique() == 1


def test_build_release_split_rejects_invalid_fraction():
    pool = pd.DataFrame({"image_id_ext": [], "item_id": [], "result": []})
    cfg = {"release_training": {"val_fraction": 1.0}}

    with pytest.raises(ValueError, match="val_fraction"):
        build_release_split(pool, cfg)


def test_fill_missing_local_paths_uses_source_specific_image_dirs():
    df = pd.DataFrame(
        {
            "image_id_ext": ["train_img", "val_img.jpg"],
            "source_dataset": ["train_df", "val_df"],
            "local_path": [None, None],
        }
    )
    cfg = {
        "data": {
            "images_train_dir": "data/raw/train_images/train_images",
            "images_val_dir": "data/raw/val_images/val_images",
        }
    }

    result = fill_missing_local_paths(df, cfg)

    assert result["local_path"].tolist() == [
        "data/raw/train_images/train_images/train_img.jpg",
        "data/raw/val_images/val_images/val_img.jpg",
    ]


def test_release_checkpoint_path_uses_explicit_output_path():
    cfg = {
        "checkpoint": {"output_path": "artifacts/checkpoints/release.ckpt"},
        "model": {"backbone": "convnext_tiny.in12k_ft_in1k"},
        "experiment": {"version": "release_v1"},
        "data": {"image_size": 224},
    }

    assert release_checkpoint_path(cfg).as_posix() == "artifacts/checkpoints/release.ckpt"
