from __future__ import annotations

import pandas as pd
import pytest

from src.datasets.weak_labels_v1 import (
    DEFAULT_WEAK_WEIGHT,
    OUTPUT_COLUMNS,
    build_audit_report,
    build_weak_labels,
    load_heuristic_sources,
)


def test_build_weak_labels_maps_deduplicates_removes_train_overlaps_and_summarizes():
    heuristic_sources = {
        "heuristics_cabinet": pd.DataFrame({"image_id_ext": ["a", "a", "train_id"]}),
        "heuristics_detskaya": pd.DataFrame({"image_id_ext": ["hash_overlap", "missing"]}),
        "heuristics_dressing_room": pd.DataFrame({"image_id_ext": ["d", "e"]}),
    }
    train = pd.DataFrame({"image_id_ext": ["train_id", "train_hash_img"]})
    manifest = pd.DataFrame(
        {
            "image_id_ext": [
                "a.jpg",
                "train_id.jpg",
                "hash_overlap.jpg",
                "train_hash_img.jpg",
                "d.jpg",
                "e.jpg",
            ],
            "hash_sha256": ["h_a", "h_train_id", "h_train", "h_train", "h_dup", "h_dup"],
        }
    )

    result = build_weak_labels(heuristic_sources, train, manifest)

    assert result.weak_labels.columns.tolist() == OUTPUT_COLUMNS
    assert result.weak_labels.to_dict(orient="records") == [
        {
            "image_id_ext": "a.jpg",
            "class_id": 5,
            "weak_weight": DEFAULT_WEAK_WEIGHT,
            "source": "heuristics_cabinet",
            "hash_sha256": "h_a",
            "is_train_overlap_image_id_ext": False,
            "is_train_overlap_hash_sha256": False,
        },
        {
            "image_id_ext": "missing.jpg",
            "class_id": 6,
            "weak_weight": DEFAULT_WEAK_WEIGHT,
            "source": "heuristics_detskaya",
            "hash_sha256": None,
            "is_train_overlap_image_id_ext": False,
            "is_train_overlap_hash_sha256": False,
        },
        {
            "image_id_ext": "d.jpg",
            "class_id": 11,
            "weak_weight": DEFAULT_WEAK_WEIGHT,
            "source": "heuristics_dressing_room",
            "hash_sha256": "h_dup",
            "is_train_overlap_image_id_ext": False,
            "is_train_overlap_hash_sha256": False,
        },
    ]

    audit = result.audit
    assert audit["input_rows_by_source"] == {
        "heuristics_cabinet": 3,
        "heuristics_detskaya": 2,
        "heuristics_dressing_room": 2,
    }
    assert audit["mapped_class_ids_by_source"] == {
        "heuristics_cabinet": 5,
        "heuristics_detskaya": 6,
        "heuristics_dressing_room": 11,
    }
    assert audit["missing_hash_rows"] == 1
    assert audit["train_overlap_rows"] == {
        "image_id_ext": 1,
        "hash_sha256": 2,
        "either": 2,
    }
    assert audit["weak_internal_duplicate_rows"] == {
        "image_id_ext_rows": 1,
        "hash_sha256_rows": 1,
    }
    assert audit["final"]["rows"] == 3
    assert audit["final"]["by_class"] == {5: 1, 6: 1, 11: 1}


def test_build_audit_report_contains_required_quality_sections():
    result = build_weak_labels(
        {"heuristics_cabinet": pd.DataFrame({"image_id_ext": ["1"]})},
        pd.DataFrame({"image_id_ext": []}),
        pd.DataFrame({"image_id_ext": ["1.jpg"], "hash_sha256": ["h1"]}),
    )

    report = build_audit_report(result.audit)

    assert "# Weak Labels Audit: weak_labels_v1" in report
    assert "## Source to class mapping" in report
    assert "## Train overlaps removed" in report
    assert "## Weak internal duplicates removed" in report
    assert "## Final row count" in report
    assert "weak_weight" in report


def test_build_weak_labels_fails_on_unknown_source():
    with pytest.raises(ValueError, match="Unknown weak-label heuristic sources"):
        build_weak_labels(
            {"heuristics_unknown": pd.DataFrame({"image_id_ext": ["1"]})},
            pd.DataFrame({"image_id_ext": []}),
            pd.DataFrame({"image_id_ext": ["1.jpg"], "hash_sha256": ["h1"]}),
        )


def test_build_weak_labels_fails_on_missing_required_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        build_weak_labels(
            {"heuristics_cabinet": pd.DataFrame({"bad": ["1"]})},
            pd.DataFrame({"image_id_ext": []}),
            pd.DataFrame({"image_id_ext": ["1.jpg"], "hash_sha256": ["h1"]}),
        )


def test_load_heuristic_sources_fails_on_unknown_file_stem(tmp_path):
    path = tmp_path / "heuristics_unknown.csv"
    path.write_text("image_id_ext\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown weak-label heuristic source"):
        load_heuristic_sources([path])
