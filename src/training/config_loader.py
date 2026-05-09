from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REQUIRED_SECTIONS = ("data", "model", "train", "debug", "artifacts", "experiment", "mlflow")
SUPPORTED_LOSSES = {"ce", "ce_label_smoothing_005", "focal", "ratio_ce", "weighted_ce"}
SUPPORTED_SAMPLERS = {"shuffle", "balanced", "class_aware_mixture", "repeat_factor"}
SUPPORTED_AUGMENTATION_LIBRARIES = {"torchvision", "albumentations"}
SUPPORTED_AUGMENTATION_POLICIES = {"baseline_v1", "safe_v1", "albumentations_v1"}


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
    experiment = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
    experiment_id = str(experiment.get("version") or "baseline_v1")
    artifacts.setdefault("oof_dir", f"artifacts/oof/{experiment_id}")
    artifacts.setdefault("report_path", f"{roots['reports']}/{experiment_id}.md")
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

    experiment = cfg["experiment"]
    loss = experiment.get("loss")
    if loss not in SUPPORTED_LOSSES:
        raise ValueError(f"Unsupported experiment.loss: {loss}")
    sampler = experiment.get("sampler")
    if sampler not in SUPPORTED_SAMPLERS:
        raise ValueError(f"Unsupported experiment.sampler: {sampler}")
    if not experiment.get("version"):
        raise ValueError("Config field 'experiment.version' is required")

    backbone = cfg["model"].get("backbone")
    whitelist = cfg["model"].get("whitelist") or []
    if whitelist and backbone not in whitelist:
        raise ValueError(f"Backbone {backbone!r} is not in model.whitelist")

    augmentation = cfg.get("augmentation", {}) or {}
    if not isinstance(augmentation, dict):
        raise ValueError("Config section 'augmentation' must be a mapping")
    library = augmentation.get("library", "torchvision")
    if library not in SUPPORTED_AUGMENTATION_LIBRARIES:
        raise ValueError(f"Unsupported augmentation.library: {library}")
    policy = augmentation.get("policy", "baseline_v1")
    if policy not in SUPPORTED_AUGMENTATION_POLICIES:
        raise ValueError(f"Unsupported augmentation.policy: {policy}")


def validate_release_config(cfg: dict[str, Any], *, require_existing: bool = True) -> None:
    for section in ("release", "data", "model", "inference"):
        _require_mapping(cfg, section)
    if not cfg["release"].get("name"):
        raise ValueError("Config field 'release.name' is required")
    if not cfg["release"].get("candidate"):
        raise ValueError("Config field 'release.candidate' is required")
    for field in ("test_csv", "images_test_dir", "class_mapping", "num_classes"):
        if field not in cfg["data"]:
            raise ValueError(f"Config field 'data.{field}' is required")

    checkpoints = cfg["model"].get("checkpoints")
    if not checkpoints:
        checkpoint = cfg["model"].get("checkpoint")
        checkpoints = [checkpoint] if checkpoint else []
    if not checkpoints:
        raise ValueError("Release config must define model.checkpoint or model.checkpoints")

    for item in checkpoints:
        path = Path(item["path"] if isinstance(item, dict) else item)
        if require_existing and not path.exists():
            raise ValueError(f"Release checkpoint does not exist: {path}")


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    loaded = OmegaConf.load(path)
    cfg = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a mapping")
    return cfg


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a training YAML config and return a validated plain dict."""
    normalized = normalize_artifact_paths(_load_yaml_mapping(path))
    validate_config(normalized)
    return normalized


def load_release_config(path: str | Path, *, require_existing: bool = True) -> dict[str, Any]:
    """Load a release YAML config and return a validated plain dict."""
    cfg = _load_yaml_mapping(path)
    validate_release_config(cfg, require_existing=require_existing)
    return cfg
