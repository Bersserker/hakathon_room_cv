from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.inference.room_predictor import RoomPredictor


def config_path() -> Path:
    return Path(os.getenv("CONFIG_PATH", "configs/release/rc1.yaml"))


@lru_cache(maxsize=1)
def load_model() -> RoomPredictor:
    return RoomPredictor.from_config_path(config_path())


def get_model_info(predictor: RoomPredictor) -> dict[str, Any]:
    cfg = predictor.cfg
    release_cfg = cfg.get("release", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    checkpoints = model_cfg.get("checkpoints") or []
    if not checkpoints and model_cfg.get("checkpoint"):
        checkpoints = [model_cfg["checkpoint"]]

    return {
        "model_name": str(
            release_cfg.get("candidate")
            or release_cfg.get("name")
            or model_cfg.get("backbone")
            or "room_classifier"
        ),
        "num_classes": int(predictor.num_classes),
        "input_size": int(data_cfg.get("image_size", 224)),
        "framework": "pytorch/timm",
        "status": "loaded",
        "device": str(predictor.device),
        "checkpoints": len(checkpoints),
    }
