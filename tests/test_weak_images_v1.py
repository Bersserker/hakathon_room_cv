from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.datasets.weak_images_v1 import (
    MANIFEST_COLUMNS,
    build_download_report,
    build_weak_images,
    load_heuristic_sources,
    normalize_image_id_ext,
    raw_candidate_rows,
)


CLASS_LABELS = {
    5: "кабинет",
    6: "детская",
    11: "гардеробная / кладовая / постирочная",
}


def make_image(path: Path, size: tuple[int, int] = (80, 80), color=(10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def empty_manifest() -> pd.DataFrame:
    return pd.DataFrame({"image_id_ext": pd.Series(dtype=str), "hash_sha256": pd.Series(dtype=str)})


def empty_splits() -> dict:
    return {"folds": [], "shadow_holdout": {"records": []}}


def test_source_to_class_mapping_and_normalize_image_id_ext(tmp_path):
    sources = {
        "heuristics_cabinet": pd.DataFrame({"image_id_ext": ["1"]}),
        "heuristics_detskaya": pd.DataFrame({"image_id_ext": [2.0]}),
        "heuristics_dressing_room": pd.DataFrame({"image_id_ext": ["3.jpg"]}),
    }

    rows = raw_candidate_rows(sources, tmp_path, CLASS_LABELS, weak_weight=0.35)

    assert normalize_image_id_ext(2.0) == "2.jpg"
    assert rows[["image_id_ext", "class_id", "label"]].to_dict(orient="records") == [
        {"image_id_ext": "1.jpg", "class_id": 5, "label": "кабинет"},
        {"image_id_ext": "2.jpg", "class_id": 6, "label": "детская"},
        {
            "image_id_ext": "3.jpg",
            "class_id": 11,
            "label": "гардеробная / кладовая / постирочная",
        },
    ]


def test_unknown_source_fails(tmp_path):
    path = tmp_path / "heuristics_unknown.csv"
    path.write_text("image_id_ext\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown weak-image heuristic source"):
        load_heuristic_sources([path])


def test_hard_gates_and_report_drop_counts(tmp_path):
    legacy = tmp_path / "legacy"
    for name in ["text.jpg", "person.jpg", "catalog.jpg", "good.jpg"]:
        make_image(legacy / name)
    make_image(legacy / "small.jpg", size=(20, 80))
    (legacy / "corrupt.jpg").write_text("not an image", encoding="utf-8")

    sources = {
        "heuristics_cabinet": pd.DataFrame(
            {
                "image_id_ext": [
                    "missing",
                    "corrupt",
                    "small",
                    "text",
                    "person",
                    "catalog",
                    "good",
                ],
                "n_texts": [0, 0, 0, 1, 0, 0, 0],
                "person_found": [False, False, False, False, "true", False, False],
                "is_catalog": [False, False, False, False, False, True, False],
                "perform_top_microcat_prob": [0.9] * 7,
                "perform_top_other_classes_prob": [0.1] * 7,
                "crop_area": [0.5] * 7,
            }
        )
    }

    result = build_weak_images(
        heuristic_sources=sources,
        legacy_images_dir=legacy,
        manifest=empty_manifest(),
        splits=empty_splits(),
        class_labels=CLASS_LABELS,
        output_image_dir=tmp_path / "out",
        max_added_per_class={5: 10},
        max_texts=0,
        drop_person=True,
        drop_catalog=True,
        min_width=64,
        min_height=64,
    )

    assert result.manifest["image_id_ext"].tolist() == ["good.jpg"]
    assert result.audit["drop_counts"]["missing_image"] == 1
    assert result.audit["drop_counts"]["corrupted_image"] == 1
    assert result.audit["drop_counts"]["small_image"] == 1
    assert result.audit["drop_counts"]["max_texts"] == 1
    assert result.audit["drop_counts"]["person_found"] == 1
    assert result.audit["drop_counts"]["is_catalog"] == 1

    report = build_download_report(result.audit)
    assert "## Drop counts by reason" in report
    assert "missing_image" in report
    assert "person_found" in report


def test_leakage_internal_duplicates_quota_scoring_and_copy(tmp_path):
    legacy = tmp_path / "legacy"
    official = tmp_path / "official.jpg"
    make_image(official, color=(200, 1, 1))

    for name, color in [
        ("leak_id.jpg", (1, 1, 1)),
        ("rank_low.jpg", (2, 2, 2)),
        ("rank_high.jpg", (3, 3, 3)),
        ("dup_id.jpg", (4, 4, 4)),
        ("dup_hash_a.jpg", (5, 5, 5)),
        ("dup_hash_b.jpg", (5, 5, 5)),
    ]:
        make_image(legacy / name, color=color)
    shutil.copy2(official, legacy / "leak_hash.jpg")

    sources = {
        "heuristics_cabinet": pd.DataFrame(
            {
                "image_id_ext": [
                    "leak_id",
                    "leak_hash",
                    "rank_low",
                    "rank_high",
                    "dup_id",
                    "dup_id",
                    "dup_hash_a",
                    "dup_hash_b",
                ],
                "n_texts": [0] * 8,
                "person_found": [False] * 8,
                "is_catalog": [False] * 8,
                "perform_top_microcat_prob": [0.9, 0.9, 0.3, 0.9, 0.4, 0.5, 0.7, 0.6],
                "perform_top_other_classes_prob": [0.1] * 8,
                "crop_area": [0.5] * 8,
            }
        )
    }
    manifest = pd.DataFrame(
        {
            "image_id_ext": ["official_hash.jpg"],
            "hash_sha256": [file_hash(official)],
        }
    )
    splits = {"folds": [{"records": [{"image_id_ext": "leak_id.jpg"}]}], "shadow_holdout": {"records": []}}

    result = build_weak_images(
        heuristic_sources=sources,
        legacy_images_dir=legacy,
        manifest=manifest,
        splits=splits,
        class_labels=CLASS_LABELS,
        output_image_dir=tmp_path / "out",
        max_added_per_class={5: 1},
        max_texts=0,
        drop_person=True,
        drop_catalog=True,
    )

    assert result.manifest.columns.tolist() == MANIFEST_COLUMNS
    assert result.manifest["image_id_ext"].tolist() == ["rank_high.jpg"]
    assert result.manifest.loc[0, "selected_rank"] == 1
    assert Path(result.manifest.loc[0, "selected_local_path"]).exists()
    assert result.audit["drop_counts"]["leakage_image_id_ext"] == 1
    assert result.audit["drop_counts"]["leakage_hash_sha256"] == 1
    assert result.audit["drop_counts"]["duplicate_image_id_ext"] == 1
    assert result.audit["drop_counts"]["duplicate_hash_sha256"] == 1
    assert result.audit["drop_counts"]["quota"] == 3
