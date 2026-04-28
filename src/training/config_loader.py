from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REQUIRED_SECTIONS = ("data", "model", "train", "debug", "artifacts", "mlflow")


def _require_mapping(cfg: dict[str, Any], section: str) -> dict[str, Any]:
    value = cfg.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section}' is required")
    return value


def normalize_artifact_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return config with centralized artifact roots, preserving old keys."""
    cfg = deepcopy(cfg)
    artifacts = cfg.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("Config section 'artifacts' must be a mapping")

    checkpoint = cfg.setdefault("checkpoint", {})
    if not isinstance(checkpoint, dict):
        raise ValueError("Config section 'checkpoint' must be a mapping")

    mlflow_cfg = _require_mapping(cfg, "mlflow")
    roots = artifacts.setdefault("roots", {})
    if not isinstance(roots, dict):
        raise ValueError("Config section 'artifacts.roots' must be a mapping")

    roots.setdefault("checkpoints", checkpoint.get("dir", "artifacts/checkpoints"))
    roots.setdefault("logs", "artifacts/logs")
    roots.setdefault("reports", "reports")
    roots.setdefault("mlflow", mlflow_cfg.get("artifact_root", "artifacts/logs/mlruns"))

    checkpoint.setdefault("dir", roots["checkpoints"])
    artifacts.setdefault("oof_dir", "artifacts/oof/baseline_v1")
    artifacts.setdefault("report_path", f"{roots['reports']}/baseline_metrics_v1.md")
    mlflow_cfg.setdefault("artifact_root", roots["mlflow"])
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    for section in REQUIRED_SECTIONS:
        _require_mapping(cfg, section)

    _require_mapping(cfg["artifacts"], "roots")
    for key in ("checkpoints", "logs", "reports", "mlflow"):
        if not cfg["artifacts"]["roots"].get(key):
            raise ValueError(f"Config field 'artifacts.roots.{key}' is required")

    for field in ("tracking_uri", "experiment_name", "artifact_root"):
        if not cfg["mlflow"].get(field):
            raise ValueError(f"Config field 'mlflow.{field}' is required")

    for field in ("splits_json", "image_col", "label_col", "num_classes"):
        if field not in cfg["data"]:
            raise ValueError(f"Config field 'data.{field}' is required")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config through OmegaConf and return a plain dict."""
    loaded = OmegaConf.load(path)
    cfg = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a mapping")
    normalized = normalize_artifact_paths(cfg)
    validate_config(normalized)
    return normalized
