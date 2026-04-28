from pathlib import Path

import pytest

from src.training.config_loader import load_config, normalize_artifact_paths

CONFIG_PATH = Path("configs/model/image_baseline_v1.yaml")


def test_load_config_exposes_required_sections():
    cfg = load_config(CONFIG_PATH)

    for section in ("data", "model", "train", "debug", "artifacts", "mlflow"):
        assert section in cfg


def test_artifact_paths_are_centralized_and_backward_compatible():
    cfg = load_config(CONFIG_PATH)

    roots = cfg["artifacts"]["roots"]
    assert roots["checkpoints"] == "artifacts/checkpoints"
    assert roots["logs"] == "artifacts/logs"
    assert roots["reports"] == "reports"
    assert roots["mlflow"] == "artifacts/logs/mlruns"
    assert cfg["checkpoint"]["dir"] == roots["checkpoints"]
    assert cfg["mlflow"]["artifact_root"] == roots["mlflow"]


def test_mlflow_tracking_configuration_is_stable():
    cfg = load_config(CONFIG_PATH)

    assert cfg["mlflow"]["tracking_uri"] == "sqlite:///artifacts/logs/mlflow.db"
    assert cfg["mlflow"]["experiment_name"] == "room_type_image_baseline"
    assert cfg["mlflow"]["artifact_root"] == cfg["artifacts"]["roots"]["mlflow"]


def test_legacy_artifact_config_is_normalized():
    cfg = normalize_artifact_paths(
        {
            "data": {
                "splits_json": "data/splits/splits_v1.json",
                "image_col": "image_id_ext",
                "label_col": "result",
                "num_classes": 20,
            },
            "model": {},
            "train": {},
            "debug": {},
            "checkpoint": {"dir": "custom/checkpoints"},
            "artifacts": {},
            "mlflow": {
                "tracking_uri": "sqlite:///custom/mlflow.db",
                "experiment_name": "exp",
                "artifact_root": "custom/mlruns",
            },
        }
    )

    assert cfg["artifacts"]["roots"]["checkpoints"] == "custom/checkpoints"
    assert cfg["artifacts"]["roots"]["mlflow"] == "custom/mlruns"
    assert cfg["checkpoint"]["dir"] == "custom/checkpoints"


def test_missing_required_config_section_fails_predictably(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("data: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="required"):
        load_config(config_path)
