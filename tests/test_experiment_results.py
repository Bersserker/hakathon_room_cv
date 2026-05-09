from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.results import (
    macro_f1_from_scores,
    metrics_from_frame,
    present_label_macro_f1,
    scores_and_targets,
    validate_prediction_frame,
)


def test_metrics_and_present_macro_f1_use_one_contract():
    frame = pd.DataFrame(
        {
            "target": [0, 1, 1, 2],
            "pred": [0, 1, 2, 2],
            "prob_0": [0.8, 0.1, 0.2, 0.1],
            "prob_1": [0.1, 0.7, 0.3, 0.1],
            "prob_2": [0.1, 0.2, 0.5, 0.8],
        }
    )

    metrics = metrics_from_frame(frame, [0, 1, 2])
    scores, targets = scores_and_targets(frame, [0, 1, 2])

    assert metrics["rows"] == 4
    assert metrics["macro_f1"] == pytest.approx((1.0 + 2 / 3 + 2 / 3) / 3)
    assert present_label_macro_f1(frame) == pytest.approx(metrics["macro_f1"])
    assert targets.tolist() == [0, 1, 1, 2]
    assert macro_f1_from_scores(scores, targets, np.zeros(3), [0, 1, 2]) == pytest.approx(
        metrics["macro_f1"]
    )


def test_prediction_frame_validation_rejects_schema_drift():
    frame = pd.DataFrame({"target": [0], "pred": [9]})

    with pytest.raises(ValueError, match="outside schema"):
        validate_prediction_frame(frame, [0, 1, 2])
