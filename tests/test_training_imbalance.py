from __future__ import annotations

import numpy as np
import pytest
import torch

from src.training.train_image import (
    build_class_aware_mixture_sampler,
    compute_class_weights,
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


def test_weighted_batch_loss_normalizes_by_sum_of_weights():
    loss_per_sample = torch.tensor([2.0, 4.0])
    labels = torch.tensor([0, 1])
    sample_weights = torch.tensor([0.5, 1.0])

    loss = weighted_batch_loss(loss_per_sample, labels, sample_weights)

    assert loss.item() == pytest.approx((2.0 * 0.5 + 4.0) / 1.5)
