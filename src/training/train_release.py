from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedGroupKFold

try:
    from src.datasets.data02_build_splits import (
        REQUIRED_CSV_COLUMNS,
        build_split_groups,
        drop_image_id_duplicates,
        filter_usable_rows,
        load_manifest,
        merge_manifest,
    )
    from src.training.config_loader import load_config
    from src.training.train_image import (
        add_sample_weight_column,
        build_balanced_sampler,
        build_class_aware_mixture_sampler,
        build_criterion,
        build_loader,
        build_repeat_factor_sampler,
        compute_class_weights,
        create_model,
        get_device,
        get_label_smoothing,
        get_transforms,
        load_model_from_checkpoint,
        log_mlflow_params,
        metric_improved,
        normalize_mlflow_experiment,
        predict,
        prediction_frame,
        resolve_sample_weight_policy,
        safe_log_artifact,
        set_seed,
        slug,
        train_one_epoch,
        write_final_config,
    )
    from src.utils.labeled_data import load_labeled_csv
    from src.utils.room_data_contract import image_id_with_extension
except ModuleNotFoundError:  # pragma: no cover - keeps direct script execution working
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.datasets.data02_build_splits import (
        REQUIRED_CSV_COLUMNS,
        build_split_groups,
        drop_image_id_duplicates,
        filter_usable_rows,
        load_manifest,
        merge_manifest,
    )
    from src.training.config_loader import load_config
    from src.training.train_image import (
        add_sample_weight_column,
        build_balanced_sampler,
        build_class_aware_mixture_sampler,
        build_criterion,
        build_loader,
        build_repeat_factor_sampler,
        compute_class_weights,
        create_model,
        get_device,
        get_label_smoothing,
        get_transforms,
        load_model_from_checkpoint,
        log_mlflow_params,
        metric_improved,
        normalize_mlflow_experiment,
        predict,
        prediction_frame,
        resolve_sample_weight_policy,
        safe_log_artifact,
        set_seed,
        slug,
        train_one_epoch,
        write_final_config,
    )
    from src.utils.labeled_data import load_labeled_csv
    from src.utils.room_data_contract import image_id_with_extension


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one release model on train_df + val_df.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/release_training/train_cv03_balanced_sampler.yaml"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def fill_missing_local_paths(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    result = df.copy()
    if "local_path" not in result.columns:
        result["local_path"] = None

    source_dirs = {
        "train_df": Path(cfg["data"]["images_train_dir"]),
        "val_df": Path(cfg["data"].get("images_val_dir", cfg["data"]["images_train_dir"])),
    }
    missing_path = result["local_path"].isna()
    for source_dataset, image_dir in source_dirs.items():
        mask = missing_path & (result["source_dataset"] == source_dataset)
        result.loc[mask, "local_path"] = result.loc[mask, "image_id_ext"].map(
            lambda value, image_dir=image_dir: str(image_dir / image_id_with_extension(value))
        )
    return result.reset_index(drop=True)


def load_release_pool(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    data_cfg = cfg["data"]
    train_df = load_labeled_csv(
        Path(data_cfg["train_df"]),
        required_columns=REQUIRED_CSV_COLUMNS,
        source_dataset="train_df",
        ratio_column="ratio",
    )
    val_df = load_labeled_csv(
        Path(data_cfg["val_df"]),
        required_columns=REQUIRED_CSV_COLUMNS,
        source_dataset="val_df",
        ratio_column="ratio",
    )

    manifest_path = Path(data_cfg.get("manifest", "data/processed/data_manifest.parquet"))
    use_manifest = bool(data_cfg.get("use_manifest", True)) and manifest_path.exists()
    manifest_df = None
    manifest_hash_source = None
    manifest_duplicate_rows = 0
    if use_manifest:
        manifest_df, manifest_hash_source, manifest_duplicate_rows = load_manifest(manifest_path)

    train_df = merge_manifest(train_df, manifest_df)
    val_df = merge_manifest(val_df, manifest_df)
    train_df, train_duplicate_rows = drop_image_id_duplicates(train_df)
    val_df, val_duplicate_rows = drop_image_id_duplicates(val_df)
    train_df, train_status_counts = filter_usable_rows(train_df, use_manifest)
    val_df, val_status_counts = filter_usable_rows(val_df, use_manifest)

    pool = pd.concat([train_df, val_df], ignore_index=True, sort=False)
    pool, combined_duplicate_rows = drop_image_id_duplicates(pool)
    pool = fill_missing_local_paths(pool, cfg)

    summary = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "combined_rows": int(len(pool)),
        "manifest_used": bool(use_manifest),
        "manifest_path": str(manifest_path),
        "manifest_hash_source_column": manifest_hash_source,
        "manifest_duplicate_image_id_ext_rows": int(manifest_duplicate_rows),
        "train_duplicate_image_id_ext_rows": int(train_duplicate_rows),
        "val_duplicate_image_id_ext_rows": int(val_duplicate_rows),
        "combined_duplicate_image_id_ext_rows": int(combined_duplicate_rows),
        "train_status_counts": train_status_counts if use_manifest else None,
        "val_status_counts": val_status_counts if use_manifest else None,
    }
    return pool.reset_index(drop=True), summary


def build_release_split(
    pool: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    release_cfg = cfg.get("release_training", {}) or {}
    val_fraction = float(release_cfg.get("val_fraction", 0.1))
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("release_training.val_fraction must be in (0, 1)")

    n_splits = int(round(1.0 / val_fraction))
    n_splits = max(2, n_splits)
    validation_fold_index = int(release_cfg.get("validation_fold_index", 0))
    if validation_fold_index < 0 or validation_fold_index >= n_splits:
        raise ValueError(
            "release_training.validation_fold_index must be between "
            f"0 and {n_splits - 1} for val_fraction={val_fraction}"
        )

    assignments = pool.sort_values(["item_id", "image_id_ext"], kind="stable").reset_index(
        drop=True
    )
    assignments = assignments.copy()
    assignments["split_group"] = build_split_groups(assignments)
    assignments["release_split"] = "train"

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(cfg["train"].get("seed", 42)),
    )
    split_iter = splitter.split(
        assignments[["image_id_ext"]],
        assignments[cfg["data"]["label_col"]].astype(int),
        groups=assignments["split_group"],
    )
    for fold_index, (_, valid_index) in enumerate(split_iter):
        if fold_index == validation_fold_index:
            assignments.loc[valid_index, "release_split"] = "valid"
            break

    train_df = assignments.loc[assignments["release_split"] == "train"].copy()
    valid_df = assignments.loc[assignments["release_split"] == "valid"].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError("Release train/valid split produced an empty split")

    train_groups = set(train_df["split_group"].tolist())
    valid_groups = set(valid_df["split_group"].tolist())
    leaked_groups = sorted(train_groups.intersection(valid_groups))
    if leaked_groups:
        raise RuntimeError(f"Release split group leakage detected: {leaked_groups[:10]}")

    summary = {
        "val_fraction": val_fraction,
        "n_splits": n_splits,
        "validation_fold_index": validation_fold_index,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "train_groups": int(train_df["split_group"].nunique()),
        "valid_groups": int(valid_df["split_group"].nunique()),
    }
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), summary


def build_train_sampler(labels: np.ndarray, cfg: dict[str, Any]):
    sampler_type = cfg["experiment"].get("sampler", "shuffle")
    num_classes = int(cfg["data"]["num_classes"])
    if sampler_type == "balanced":
        return build_balanced_sampler(labels, num_classes), False
    if sampler_type == "class_aware_mixture":
        return (
            build_class_aware_mixture_sampler(
                labels,
                num_classes,
                mixture_lambda=cfg["experiment"].get("sampler_mixture_lambda", 0.5),
            ),
            False,
        )
    if sampler_type == "repeat_factor":
        return (
            build_repeat_factor_sampler(
                labels,
                num_classes,
                target_freq=cfg["experiment"].get("repeat_factor_target_freq"),
                repeat_factor_cap=cfg["experiment"].get("repeat_factor_cap", 4.0),
            ),
            False,
        )
    if sampler_type == "shuffle":
        return None, True
    raise ValueError(f"Unsupported sampler: {sampler_type}")


def release_checkpoint_path(cfg: dict[str, Any]) -> Path:
    checkpoint_cfg = cfg.get("checkpoint", {}) or {}
    if checkpoint_cfg.get("output_path"):
        return Path(checkpoint_cfg["output_path"])

    backbone_tag = slug(cfg["model"]["backbone"])
    version = cfg["experiment"]["version"]
    image_size = cfg["data"]["image_size"]
    return Path(checkpoint_cfg.get("dir", "artifacts/checkpoints")) / (
        f"roomclf_{backbone_tag}_release_{image_size}_{version}.ckpt"
    )


def write_server_release_config(cfg: dict[str, Any], checkpoint_path: Path) -> Path:
    output_cfg = cfg.get("release_output", {}) or {}
    release_config_path = Path(output_cfg.get("config_path", "configs/release/rc1_single.yaml"))
    release_config_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "release": {
            "name": output_cfg.get("name", "rc1_single"),
            "candidate": output_cfg.get("candidate", cfg["experiment"]["version"]),
            "notes": (
                "Single release model trained on train_df + val_df with a 90/10 "
                "group-safe validation split for early stopping."
            ),
        },
        "data": {
            "test_csv": cfg["data"].get("test_df", "data/raw/test_df.csv"),
            "images_test_dir": cfg["data"].get(
                "images_test_dir", "data/raw/test_images/test_images"
            ),
            "class_mapping": cfg["data"].get(
                "class_mapping", "configs/data/class_mapping.yaml"
            ),
            "num_classes": int(cfg["data"].get("num_classes", 20)),
            "image_size": int(cfg["data"].get("image_size", 224)),
            "resize_size": int(cfg["data"].get("resize_size", 256)),
        },
        "model": {
            "backbone": cfg["model"]["backbone"],
            "checkpoint": checkpoint_path.as_posix(),
        },
        "postprocess": {"class_bias": None},
        "inference": {
            "seed": int(cfg["train"].get("seed", 42)),
            "device": output_cfg.get("inference_device", "mps"),
            "batch_size": int(cfg.get("inference", {}).get("batch_size", 64)),
            "num_workers": int(cfg.get("inference", {}).get("num_workers", 0)),
            "tta": bool(cfg["experiment"].get("tta_flag", False)),
            "validate_after": True,
            "output_submission": output_cfg.get(
                "output_submission", "releases/rc1_single/submission.csv"
            ),
            "output_predictions": output_cfg.get(
                "output_predictions", "releases/rc1_single/predictions.parquet"
            ),
        },
    }
    with release_config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)
    return release_config_path


def write_release_report(
    cfg: dict[str, Any],
    output_dir: Path,
    checkpoint_path: Path,
    data_summary: dict[str, Any],
    split_summary: dict[str, Any],
    best_metrics: dict[str, Any],
    release_config_path: Path,
) -> Path:
    report_path = Path(cfg["artifacts"]["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    content = "\n".join(
        [
            "# Release training report",
            "",
            f"- experiment: `{cfg['experiment']['version']}`",
            f"- backbone: `{cfg['model']['backbone']}`",
            f"- checkpoint: `{checkpoint_path}`",
            f"- manifest_used: `{data_summary['manifest_used']}`",
            f"- combined_rows: `{data_summary['combined_rows']}`",
            f"- train_rows: `{split_summary['train_rows']}`",
            f"- valid_rows: `{split_summary['valid_rows']}`",
            f"- val_fraction: `{split_summary['val_fraction']}`",
            f"- best_epoch: `{best_metrics['best_epoch']}`",
            f"- best_val_macro_f1: `{best_metrics['best_val_macro_f1']:.6f}`",
            f"- best_score: `{best_metrics['best_score']:.6f}`",
            f"- monitor_metric: `{best_metrics['monitor_metric']}`",
            "",
            "## Use on server",
            "",
            "```bash",
            f"CONFIG_PATH={release_config_path.as_posix()} \\",
            "uv run uvicorn serving.app.main:app --host 0.0.0.0 --port 8000",
            "```",
            "",
        ]
    )
    report_path.write_text(content, encoding="utf-8")
    return report_path


def run_release_training(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.debug:
        cfg["train"]["epochs"] = cfg["debug"]["epochs"]
        cfg["train"]["num_workers"] = cfg["debug"]["num_workers"]

    set_seed(int(cfg["train"].get("seed", 42)))
    device = get_device(cfg)
    print(f"Device: {device}")

    pool, data_summary = load_release_pool(cfg)
    train_df, valid_df, split_summary = build_release_split(pool, cfg)
    if args.debug:
        train_df = train_df.head(min(cfg["debug"]["train_samples"], len(train_df)))
        valid_df = valid_df.head(min(cfg["debug"]["val_samples"], len(valid_df)))
        split_summary = {
            **split_summary,
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "debug": True,
        }
    valid_df["fold"] = "release_valid"
    print(
        "Release split: "
        f"train_rows={split_summary['train_rows']} | valid_rows={split_summary['valid_rows']}"
    )

    train_transform, val_transform = get_transforms(cfg)
    ratio_policy = cfg["experiment"].get("ratio_policy", "none")
    sample_weight_policy = resolve_sample_weight_policy(cfg)
    train_labels = train_df[cfg["data"]["label_col"]].astype(int).to_numpy()
    train_sampler, train_shuffle = build_train_sampler(train_labels, cfg)

    train_loader = build_loader(
        train_df,
        cfg["data"]["images_train_dir"],
        train_transform,
        cfg,
        device,
        shuffle=train_shuffle,
        sampler=train_sampler,
        ratio_policy=ratio_policy,
        sample_weight_policy=sample_weight_policy,
    )
    valid_loader = build_loader(
        valid_df,
        cfg["data"]["images_train_dir"],
        val_transform,
        cfg,
        device,
        shuffle=False,
        sampler=None,
        ratio_policy="none",
        sample_weight_policy="none",
    )

    model = create_model(cfg, device, debug=args.debug)
    loss_name = cfg["experiment"].get("loss", "ce")
    class_weight_policy = cfg["experiment"].get("class_weight_policy", "none")
    if loss_name == "weighted_ce" and class_weight_policy == "none":
        class_weight_policy = "raw_inverse"

    train_sample_weights = None
    if sample_weight_policy != "none":
        weighted_train_df = add_sample_weight_column(
            train_df,
            ratio_policy=ratio_policy,
            sample_weight_policy=sample_weight_policy,
            weak_weight=cfg["experiment"].get("weak_weight", 0.35),
        )
        train_sample_weights = weighted_train_df["sample_weight"].to_numpy(dtype=np.float64)

    class_weights = None
    if class_weight_policy != "none":
        class_weights = compute_class_weights(
            train_labels,
            int(cfg["data"]["num_classes"]),
            policy=class_weight_policy,
            clip_min=cfg["experiment"].get("weight_clip_min"),
            clip_max=cfg["experiment"].get("weight_clip_max"),
            effective_beta=cfg["experiment"].get("effective_beta", 0.999),
            sample_weights=train_sample_weights,
        ).to(device)

    use_sample_weights = sample_weight_policy != "none"
    criterion = build_criterion(
        loss_name,
        use_sample_weights=use_sample_weights,
        class_weights=class_weights,
        label_smoothing=get_label_smoothing(cfg),
        cfg=cfg,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    early_stopping = cfg.get("early_stopping", {}) or {}
    early_stopping_enabled = bool(early_stopping.get("enabled", True))
    early_stopping_patience = int(early_stopping.get("patience", 10))
    early_stopping_min_delta = float(early_stopping.get("min_delta", 0.0))
    monitor_metric = early_stopping.get("monitor", "val_macro_f1")
    monitor_mode = early_stopping.get(
        "mode", "min" if str(monitor_metric).endswith("loss") else "max"
    )
    if early_stopping_enabled and early_stopping_patience < 1:
        raise ValueError("early_stopping.patience must be >= 1 when enabled")

    checkpoint_path = release_checkpoint_path(cfg)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(cfg["artifacts"]["oof_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = normalize_mlflow_experiment(cfg)
    best_score = None
    best_val_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    with mlflow.start_run(experiment_id=experiment_id, run_name=cfg["experiment"]["version"]):
        log_mlflow_params(cfg, fold=-1, device=device, debug=args.debug)
        mlflow.log_params(
            {
                "release_train_rows": split_summary["train_rows"],
                "release_valid_rows": split_summary["valid_rows"],
                "release_val_fraction": split_summary["val_fraction"],
            }
        )

        for epoch in range(int(cfg["train"]["epochs"])):
            print(f"\nEpoch {epoch + 1}/{cfg['train']['epochs']}")
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
                loss_name=loss_name,
                class_weights=class_weights,
                use_sample_weights=use_sample_weights,
            )
            val_result = predict(
                model=model,
                loader=valid_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                desc="Release Eval",
                loss_name=loss_name,
                class_weights=class_weights,
                use_sample_weights=use_sample_weights,
            )
            print(
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_result['loss']:.4f} | "
                f"val_macro_f1={val_result['macro_f1']:.4f} | "
                f"val_acc={val_result['accuracy']:.4f}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_result["loss"], step=epoch)
            mlflow.log_metric("val_macro_f1", val_result["macro_f1"], step=epoch)
            mlflow.log_metric("val_accuracy", val_result["accuracy"], step=epoch)

            monitor_values = {
                "val_loss": float(val_result["loss"]),
                "val_macro_f1": float(val_result["macro_f1"]),
                "val_accuracy": float(val_result["accuracy"]),
            }
            if monitor_metric not in monitor_values:
                raise ValueError(f"Unsupported monitor metric: {monitor_metric}")
            current_score = monitor_values[monitor_metric]

            if metric_improved(current_score, best_score, monitor_mode, early_stopping_min_delta):
                best_score = current_score
                best_val_macro_f1 = float(val_result["macro_f1"])
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "backbone": cfg["model"]["backbone"],
                        "num_classes": cfg["data"]["num_classes"],
                        "epoch": epoch,
                        "best_f1": best_val_macro_f1,
                        "best_score": best_score,
                        "monitor_metric": monitor_metric,
                        "release_training": {
                            "data_summary": data_summary,
                            "split_summary": split_summary,
                        },
                        "config": cfg,
                    },
                    checkpoint_path,
                )
                safe_log_artifact(checkpoint_path, artifact_path="checkpoints")
                print(f"Saved best release checkpoint: {checkpoint_path}")
            else:
                epochs_without_improvement += 1
                if early_stopping_enabled:
                    print(
                        "Early stopping wait: "
                        f"{epochs_without_improvement}/{early_stopping_patience}"
                    )
                    if epochs_without_improvement >= early_stopping_patience:
                        mlflow.log_metric("early_stopping_epoch", epoch + 1, step=epoch)
                        mlflow.set_tag("early_stopped", "true")
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

        if best_score is None:
            raise RuntimeError("Training finished without saving a checkpoint")
        mlflow.log_metric("best_val_macro_f1", best_val_macro_f1)
        mlflow.log_metric(f"best_{monitor_metric}", best_score)

        best_model = load_model_from_checkpoint(checkpoint_path, device)
        valid_result = predict(
            model=best_model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            desc="Release Best Eval",
            loss_name=loss_name,
            class_weights=class_weights,
            use_sample_weights=use_sample_weights,
        )
        valid_frame = prediction_frame(valid_df, valid_result, int(cfg["data"]["num_classes"]))
        valid_predictions_path = output_dir / "validation_predictions.parquet"
        valid_frame.to_parquet(valid_predictions_path, index=False)
        safe_log_artifact(valid_predictions_path, artifact_path="release_validation")
        config_path = write_final_config(cfg, output_dir)
        safe_log_artifact(config_path, artifact_path="config")

        release_config_path = write_server_release_config(cfg, checkpoint_path)
        safe_log_artifact(release_config_path, artifact_path="config")
        best_metrics = {
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "best_score": float(best_score),
            "monitor_metric": monitor_metric,
        }
        report_path = write_release_report(
            cfg=cfg,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            data_summary=data_summary,
            split_summary=split_summary,
            best_metrics=best_metrics,
            release_config_path=release_config_path,
        )
        safe_log_artifact(report_path, artifact_path="reports")

    return {
        "checkpoint": checkpoint_path,
        "validation_predictions": valid_predictions_path,
        "config": config_path,
        "release_config": release_config_path,
        "report": report_path,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
    }


def main() -> None:
    result = run_release_training(parse_args())
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
