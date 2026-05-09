from __future__ import annotations

import pandas as pd

from src.datasets.data01_preprocess import (
    drop_exact_test_leaks,
    fatal_exact_leaks,
    pairwise_overlaps,
)


def test_drop_exact_test_leaks_removes_content_hash_overlap():
    train = pd.DataFrame(
        {
            "image_id_ext": ["1", "2"],
            "image_id_ext_file": ["1.jpg", "2.jpg"],
            "image": ["http://example/1.jpg", "http://example/2.jpg"],
            "content_hash": ["hash_1", "hash_2"],
        }
    )
    test = pd.DataFrame(
        {
            "image_id_ext": ["9"],
            "image_id_ext_file": ["9.jpg"],
            "image": ["http://example/9.jpg"],
            "content_hash": ["hash_2"],
        }
    )

    clean, summary = drop_exact_test_leaks(train, test, ["image_id_ext", "image"])

    assert clean["image_id_ext"].tolist() == ["1"]
    assert summary["dropped_rows"] == 1
    assert summary["counts_by_key"]["content_hash"] == 1


def test_pairwise_overlaps_has_no_fatal_key_after_filtering():
    frames = {
        "train": pd.DataFrame(
            {
                "item_id": ["same_item"],
                "image_id_ext_file": ["1.jpg"],
                "image": ["http://example/1.jpg"],
                "content_hash": ["hash_1"],
            }
        ),
        "val": pd.DataFrame(
            {
                "item_id": ["val_item"],
                "image_id_ext_file": ["2.jpg"],
                "image": ["http://example/2.jpg"],
                "content_hash": ["hash_2"],
            }
        ),
        "test": pd.DataFrame(
            {
                "item_id": ["same_item"],
                "image_id_ext_file": ["3.jpg"],
                "image": ["http://example/3.jpg"],
                "content_hash": ["hash_3"],
            }
        ),
    }

    overlaps = pairwise_overlaps(frames)

    assert overlaps["train_vs_test"]["item_id"]["intersection_count"] == 1
    assert fatal_exact_leaks(overlaps) == []
