from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.training.train_image import (
    FocalLoss,
    RoomDataset,
    build_class_aware_mixture_sampler,
    build_criterion,
    build_train_frame_with_optional_weak,
    compute_class_weights,
    compute_effective_class_counts,
    metric_improved,
    weighted_batch_loss,
)


def test_sqrt_inverse_class_weights_are_clipped_and_normalized():
    labels = np.array([0] * 100 + [1] * 25 + [2] * 4)

    weights = compute_class_weights(
        labels,
        num_classes=3,
        policy="sqrt_inv",
        clip_min=0.5,
        clip_max=2.5,
    )

    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[2] > weights[1] > weights[0]


def test_class_aware_mixture_sampler_gives_rare_class_more_weight_per_sample():
    labels = np.array([0, 0, 0, 0, 1])

    sampler = build_class_aware_mixture_sampler(labels, num_classes=2, mixture_lambda=0.5)

    weights = sampler.weights.numpy()
    assert weights[-1] > weights[0]


def test_metric_improved_respects_mode_and_min_delta():
    assert metric_improved(0.62, None, "max", 0.001)
    assert metric_improved(0.622, 0.62, "max", 0.001)
    assert not metric_improved(0.6205, 0.62, "max", 0.001)
    assert metric_improved(0.9, 1.0, "min", 0.05)
    assert not metric_improved(0.98, 1.0, "min", 0.05)


def test_weighted_batch_loss_normalizes_by_sum_of_weights():
    loss_per_sample = torch.tensor([2.0, 4.0])
    labels = torch.tensor([0, 1])
    sample_weights = torch.tensor([0.5, 1.0])

    loss = weighted_batch_loss(loss_per_sample, labels, sample_weights)

    assert loss.item() == pytest.approx((2.0 * 0.5 + 4.0) / 1.5)


def test_focal_loss_gamma_zero_matches_cross_entropy():
    logits = torch.tensor([[3.0, 0.1], [0.2, 1.7]])
    labels = torch.tensor([0, 1])

    focal = FocalLoss(gamma=0.0)(logits, labels)
    ce = torch.nn.functional.cross_entropy(logits, labels)

    assert focal.item() == pytest.approx(ce.item())


def test_focal_loss_downweights_easy_examples_more_than_hard_examples():
    logits = torch.tensor([[5.0, 0.0], [0.1, 0.0]])
    labels = torch.tensor([0, 0])

    focal = FocalLoss(gamma=2.0, reduction="none")(logits, labels)
    ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    ratios = focal / ce

    assert ratios[0] < ratios[1]


def test_build_focal_criterion_uses_none_reduction_when_weighting_is_enabled():
    criterion = build_criterion(
        "focal",
        use_sample_weights=True,
        class_weights=None,
        label_smoothing=0.0,
        cfg={"experiment": {"focal_gamma": 2.0}},
    )

    loss = criterion(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1]))

    assert loss.shape == (2,)


def test_room_dataset_prefers_existing_local_path_and_returns_source_x_ratio_weight(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    fallback = images_dir / "fallback.jpg"
    local = tmp_path / "weak.jpg"
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(fallback)
    Image.new("RGB", (17, 17), color=(0, 255, 0)).save(local)

    dataset = RoomDataset(
        pd.DataFrame(
            {
                "image_id_ext": ["fallback.jpg"],
                "local_path": [local.as_posix()],
                "result": [5],
                "ratio": [0.5],
                "weak_weight": [0.35],
            }
        ),
        images_dir=images_dir,
        image_col="image_id_ext",
        label_col="result",
        ratio_policy="clip_075_100",
        sample_weight_policy="source_x_ratio",
    )

    image, label, sample_weight = dataset[0]

    assert image.size == (17, 17)
    assert label == 5
    assert sample_weight.item() == pytest.approx(0.35 * 0.75)


def test_build_train_frame_with_optional_weak_injects_train_only(tmp_path):
    weak_manifest = tmp_path / "weak.csv"
    weak_manifest.write_text(
        "image_id_ext,class_id,label,weak_weight,selected_local_path,hash_sha256\n"
        "weak.jpg,5,кабинет,0.35,/tmp/weak.jpg,hweak\n",
        encoding="utf-8",
    )
    splits = {
        "folds": [
            {"fold": 0, "records": [{"image_id_ext": "valid.jpg", "result": 1}]},
            {"fold": 1, "records": [{"image_id_ext": "train.jpg", "result": 2}]},
        ],
        "shadow_holdout": {"records": [{"image_id_ext": "shadow.jpg", "result": 3}]},
    }
    cfg = {
        "data": {"weak_manifest": weak_manifest.as_posix(), "label_col": "result"},
        "experiment": {"weak_label_flag": True},
    }

    train_df, valid_df = build_train_frame_with_optional_weak(splits, fold=0, cfg=cfg)

    assert train_df["image_id_ext"].tolist() == ["train.jpg", "weak.jpg"]
    assert valid_df["image_id_ext"].tolist() == ["valid.jpg"]
    assert "weak.jpg" not in valid_df["image_id_ext"].tolist()
    assert train_df.loc[train_df["image_id_ext"] == "weak.jpg", "result"].item() == 5


def test_sqrt_median_effective_class_weights_use_sample_weights_and_clip():
    df = pd.DataFrame({"result": [0, 0, 1, 2], "sample_weight": [1.0, 1.0, 0.25, 0.25]})

    counts = compute_effective_class_counts(df, "result", "sample_weight")
    weights = compute_class_weights(
        df["result"].to_numpy(),
        num_classes=3,
        policy="sqrt_median_effective",
        sample_weights=df["sample_weight"].to_numpy(),
        clip_max=2.5,
    )

    assert counts.to_dict() == {0: 2.0, 1: 0.25, 2: 0.25}
    assert weights.tolist() == pytest.approx([0.353553, 1.0, 1.0], rel=1e-5)
