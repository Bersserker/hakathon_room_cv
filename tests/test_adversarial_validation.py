from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.diagnostics.adversarial_cv import make_group_stratified_folds, run_adversarial_cv
from src.diagnostics.adversarial_data import (
    HOLDOUT_DOMAIN,
    HOLDOUT_DOMAIN_LABEL,
    TRAIN_DOMAIN,
    TRAIN_DOMAIN_LABEL,
    assemble_domain_dataset,
    class_balance_sources,
    label_shift_table,
    load_domain_frames_from_splits,
)
from src.diagnostics.adversarial_features import (
    assert_no_forbidden_features,
    build_metadata_matrix,
    build_visual_matrix,
)
from src.diagnostics.adversarial_report import build_adversarial_markdown_report
from src.diagnostics.embeddings import ensure_embeddings


def make_record(
    tmp_path: Path,
    *,
    image_id: str,
    item_id: str,
    class_id: int,
    domain: str,
    width: int = 100,
    height: int = 100,
    ratio: float = 1.0,
) -> dict:
    image_path = tmp_path / f"{domain}_{image_id}.jpg"
    image_path.write_bytes(b"fake image bytes")
    return {
        "image_id_ext": f"{image_id}.jpg",
        "item_id": item_id,
        "result": class_id,
        "label": f"class_{class_id}",
        "ratio": ratio,
        "image": f"http://example/{image_id}.jpg",
        "source_dataset": "train_df" if domain == TRAIN_DOMAIN else "val_df",
        "local_path": image_path.as_posix(),
        "width": width,
        "height": height,
        "status": "ok",
        "content_hash": f"hash_{domain}_{image_id}",
    }


def make_domain_df(tmp_path: Path) -> pd.DataFrame:
    train = []
    holdout = []
    for index in range(6):
        train.append(
            make_record(
                tmp_path,
                image_id=f"tr{index}",
                item_id=f"tr_item_{index}",
                class_id=index % 2,
                domain=TRAIN_DOMAIN,
                width=100 + index,
                height=100,
                ratio=0.9,
            )
        )
        holdout.append(
            make_record(
                tmp_path,
                image_id=f"ho{index}",
                item_id=f"ho_item_{index}",
                class_id=index % 2,
                domain=HOLDOUT_DOMAIN,
                width=150 + index,
                height=100,
                ratio=0.8,
            )
        )
    return assemble_domain_dataset(pd.DataFrame(train), pd.DataFrame(holdout))


def test_dataset_assembly_from_splits_assigns_domains_and_keeps_manifest_metadata(tmp_path):
    train_records = [
        make_record(
            tmp_path,
            image_id="train0",
            item_id="item_train0",
            class_id=0,
            domain=TRAIN_DOMAIN,
            width=640,
            height=480,
        ),
        make_record(
            tmp_path,
            image_id="train1",
            item_id="item_train1",
            class_id=1,
            domain=TRAIN_DOMAIN,
            width=800,
            height=600,
        ),
    ]
    holdout_records = [
        make_record(
            tmp_path,
            image_id="hold0",
            item_id="item_hold0",
            class_id=0,
            domain=HOLDOUT_DOMAIN,
            width=320,
            height=240,
        )
    ]
    splits = {
        "folds": [
            {"fold": 0, "records": [train_records[0]]},
            {"fold": 1, "records": [train_records[1]]},
        ],
        "shadow_holdout": {"records": holdout_records},
    }
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(splits), encoding="utf-8")

    frames = load_domain_frames_from_splits(path)

    assert frames.train["width"].tolist() == [640, 800]
    assert frames.holdout["content_hash"].tolist() == ["hash_shadow_holdout_hold0"]
    assert frames.combined["domain"].tolist() == [TRAIN_DOMAIN, TRAIN_DOMAIN, HOLDOUT_DOMAIN]
    assert frames.combined["domain_label"].tolist() == [TRAIN_DOMAIN_LABEL, TRAIN_DOMAIN_LABEL, HOLDOUT_DOMAIN_LABEL]


def test_dataset_assembly_fails_on_missing_required_columns(tmp_path):
    image_path = tmp_path / "x.jpg"
    image_path.write_bytes(b"x")
    splits = {
        "folds": [{"fold": 0, "records": [{"image_id_ext": "x.jpg", "local_path": image_path.as_posix()}]}],
        "shadow_holdout": {"records": []},
    }
    path = tmp_path / "bad_splits.json"
    path.write_text(json.dumps(splits), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_domain_frames_from_splits(path)


def test_forbidden_columns_do_not_enter_metadata_or_visual_feature_matrices(tmp_path):
    df = make_domain_df(tmp_path)

    metadata_x, metadata_features = build_metadata_matrix(df)
    visual_embeddings = pd.DataFrame(
        {
            "image_id_ext": df["image_id_ext"],
            "emb_0000": np.arange(len(df), dtype=float),
            "emb_0001": np.arange(len(df), dtype=float) + 1.0,
        }
    )
    visual_x, visual_features = build_visual_matrix(df, visual_embeddings)

    assert metadata_x.shape[0] == len(df)
    assert visual_x.shape == (len(df), 2)
    forbidden = {
        "result",
        "label",
        "item_id",
        "image_id_ext",
        "image",
        "content_hash",
        "local_path",
    }
    assert forbidden.isdisjoint(metadata_features)
    assert forbidden.isdisjoint(visual_features)
    with pytest.raises(ValueError, match="Forbidden"):
        assert_no_forbidden_features(["width", "item_id"])


def test_class_balanced_sampling_excludes_missing_classes_and_matches_counts(tmp_path):
    train_records = []
    holdout_records = []
    for index, class_id in enumerate([0, 0, 0, 1, 1, 18]):
        train_records.append(
            make_record(
                tmp_path,
                image_id=f"tr_bal_{index}",
                item_id=f"tr_bal_item_{index}",
                class_id=class_id,
                domain=TRAIN_DOMAIN,
            )
        )
    for index, class_id in enumerate([0, 0, 1]):
        holdout_records.append(
            make_record(
                tmp_path,
                image_id=f"ho_bal_{index}",
                item_id=f"ho_bal_item_{index}",
                class_id=class_id,
                domain=HOLDOUT_DOMAIN,
            )
        )
    df = assemble_domain_dataset(pd.DataFrame(train_records), pd.DataFrame(holdout_records))

    balanced, audit = class_balance_sources(df, seed=123)

    assert audit["shared_classes"] == [0, 1]
    assert audit["missing_from_holdout_classes"] == [18]
    assert set(balanced["result"].unique()) == {0, 1}
    counts = balanced.groupby(["result", "domain_label"]).size().unstack(fill_value=0)
    assert counts.loc[0, TRAIN_DOMAIN_LABEL] == counts.loc[0, HOLDOUT_DOMAIN_LABEL] == 2
    assert counts.loc[1, TRAIN_DOMAIN_LABEL] == counts.loc[1, HOLDOUT_DOMAIN_LABEL] == 1


def test_group_safe_folds_do_not_split_item_groups(tmp_path):
    df = make_domain_df(tmp_path)
    extra = df.iloc[[0, 6]].copy()
    extra["image_id_ext"] = ["extra_train.jpg", "extra_holdout.jpg"]
    df = pd.concat([df, extra], ignore_index=True)

    folds = make_group_stratified_folds(df, n_splits=2, seed=123)

    for train_index, eval_index in folds:
        train_groups = set(df.iloc[train_index]["item_id"].astype(str))
        eval_groups = set(df.iloc[eval_index]["item_id"].astype(str))
        assert train_groups.isdisjoint(eval_groups)


def test_metric_pipeline_outputs_stable_summary_keys_and_valid_ranges(tmp_path):
    df = make_domain_df(tmp_path)
    x = np.column_stack(
        [
            df["domain_label"].to_numpy(dtype=float),
            np.arange(len(df), dtype=float) / len(df),
        ]
    )

    result = run_adversarial_cv(
        df,
        x,
        ["emb_0000", "emb_0001"],
        mode="raw",
        feature_set="visual",
        n_splits=2,
        seed=123,
    )

    assert result.summary["n_folds"] == 2
    assert 0.0 <= result.summary["roc_auc_mean"] <= 1.0
    assert 0.0 <= result.summary["pr_auc_mean"] <= 1.0
    assert 0.0 <= result.summary["balanced_accuracy_mean"] <= 1.0
    assert "roc_auc_ci95_half_width" in result.summary
    assert len(result.predictions) == len(df)
    assert set(result.fold_metrics["fold"]) == {0, 1}


def test_markdown_report_contains_required_sections(tmp_path):
    df = make_domain_df(tmp_path)
    balanced, audit = class_balance_sources(df, seed=123)
    label_shift = label_shift_table(df)
    x_raw = df[["domain_label"]].to_numpy(dtype=float)
    x_balanced = balanced[["domain_label"]].to_numpy(dtype=float)
    raw_metadata = run_adversarial_cv(
        df,
        x_raw,
        ["ratio"],
        mode="raw",
        feature_set="metadata",
        n_splits=2,
        seed=123,
    )
    balanced_visual = run_adversarial_cv(
        balanced,
        x_balanced,
        ["emb_0000"],
        mode="class_balanced",
        feature_set="visual",
        n_splits=2,
        seed=123,
    )

    report = build_adversarial_markdown_report(
        label_shift=label_shift,
        results=[raw_metadata, balanced_visual],
        balance_audit=audit,
        command="python scripts/run_adversarial_validation.py",
        generated_at="2026-05-06T00:00:00Z",
    )

    assert "## Label shift report" in report
    assert "## Metadata shift" in report
    assert "## Visual shift" in report
    assert "raw" in report
    assert "class_balanced" in report
    assert "## Interpretation thresholds" in report
    assert "## Excluded features" in report
    assert "low adversarial auc" in report.lower()


class FakeEmbeddingExtractor:
    embedding_name = "fake"

    def __init__(self) -> None:
        self.extracted_paths: list[Path] = []

    def extract(self, image_paths):
        self.extracted_paths.extend(image_paths)
        rows = []
        for index, path in enumerate(image_paths):
            rows.append([float(len(path.name)), float(index)])
        return np.asarray(rows, dtype=np.float32)


def test_embedding_cache_reuses_fake_extractor_outputs(tmp_path):
    df = make_domain_df(tmp_path).head(4)
    cache_path = tmp_path / "embeddings.parquet"
    extractor = FakeEmbeddingExtractor()

    first = ensure_embeddings(df, extractor=extractor, cache_path=cache_path, batch_size=2)
    first_call_count = len(extractor.extracted_paths)
    second = ensure_embeddings(df, extractor=extractor, cache_path=cache_path, batch_size=2)

    assert cache_path.exists()
    assert first_call_count == 4
    assert len(extractor.extracted_paths) == 4
    assert first[["image_id_ext", "emb_0000", "emb_0001"]].equals(
        second[["image_id_ext", "emb_0000", "emb_0001"]]
    )
