#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score

CLASS_IDS = list(range(20))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build experiment registry from OOF artifacts.")
    parser.add_argument("--oof-root", type=Path, default=Path("artifacts/oof"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiment_registry.csv"))
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    return payload if isinstance(payload, dict) else {}


def metrics(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    y_true = frame["target"].to_numpy()
    y_pred = frame["pred"].to_numpy()
    per_class = f1_score(y_true, y_pred, average=None, labels=CLASS_IDS, zero_division=0)
    return {
        "rows": int(len(frame)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=CLASS_IDS, zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "class_5_f1": float(per_class[5]),
        "class_11_f1": float(per_class[11]),
        "class_18_f1": float(per_class[18]),
    }


def shadow_metrics(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    present = sorted(int(value) for value in frame["target"].unique())
    return {
        "shadow_rows": int(len(frame)),
        "shadow_macro_f1_all": float(
            f1_score(
                frame["target"], frame["pred"], average="macro", labels=CLASS_IDS, zero_division=0
            )
        ),
        "shadow_macro_f1_present": float(
            f1_score(
                frame["target"], frame["pred"], average="macro", labels=present, zero_division=0
            )
        ),
        "shadow_accuracy": float(accuracy_score(frame["target"], frame["pred"])),
    }


def main() -> None:
    args = parse_args()
    rows = []
    for run_dir in sorted(path for path in args.oof_root.iterdir() if path.is_dir()):
        oof_path = run_dir / "oof_predictions.parquet"
        if not oof_path.exists():
            continue
        cfg = load_yaml(run_dir / "config.yaml")
        experiment = cfg.get("experiment", {})
        data = cfg.get("data", {})
        model = cfg.get("model", {})
        row = {
            "run": run_dir.name,
            "config": str(run_dir / "config.yaml"),
            "split_version": data.get("split_version"),
            "data_version": data.get("dataset_version"),
            "backbone": model.get("backbone"),
            "loss": experiment.get("loss"),
            "sampler": experiment.get("sampler"),
            "ratio_policy": experiment.get("ratio_policy"),
            "checkpoint": cfg.get("checkpoint", {}).get("dir"),
            "notes": "generated_from_oof_artifacts",
            "decision": "candidate" if run_dir.name == "cv03_balanced_sampler" else "reviewed",
        }
        row.update(metrics(oof_path))
        shadow_path = run_dir / "shadow_holdout_predictions.parquet"
        if shadow_path.exists():
            row.update(shadow_metrics(shadow_path))
        rows.append(row)

    registry = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
