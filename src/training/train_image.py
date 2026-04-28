import argparse
import copy
import json
import random
import re
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import yaml
from config_loader import load_config
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_image_id_ext(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text if Path(text).suffix else f"{text}.jpg"


def slug(value: str) -> str:
    value = value.replace(".in12k_ft_in1k", "")
    value = value.replace("convnext_tiny", "convnexttiny")
    value = value.replace("efficientnet_b0", "efficientnetb0")
    value = re.sub(r"[^a-zA-Z0-9]+", "", value)
    return value.lower()


def git_commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def current_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class RoomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir,
        image_col,
        label_col,
        transform=None,
        num_classes=20,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = normalize_image_id_ext(row[self.image_col])
        label = int(row[self.label_col])

        assert 0 <= label < self.num_classes, f"Bad label: {label}, idx={idx}, file={img_name}"

        local_path = row.get("local_path")
        img_path = Path(local_path) if isinstance(local_path, str) and local_path else None
        if img_path is None or not img_path.exists():
            img_path = self.images_dir / img_name

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_device(cfg):
    requested = cfg["train"].get("device", "cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_transforms(cfg):
    image_size = int(cfg["data"].get("image_size", 224))
    resize_size = int(cfg["data"].get("resize_size", 256))

    train_transform = transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, val_transform


def load_splits(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("version") != "splits_v1":
        raise ValueError(f"Unsupported split version: {data.get('version')!r}")
    return data


def records_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records).copy()
    df["image_id_ext"] = df["image_id_ext"].map(normalize_image_id_ext)
    return df.reset_index(drop=True)


def build_fold_frames(splits: dict[str, Any], fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = splits["folds"]
    valid_records = folds[fold]["records"]
    train_records: list[dict[str, Any]] = []
    for fold_payload in folds:
        if int(fold_payload["fold"]) != fold:
            train_records.extend(fold_payload["records"])
    return records_to_df(train_records), records_to_df(valid_records)


def build_loader(df, images_dir, transform, cfg, device, shuffle):
    dataset = RoomDataset(
        df=df,
        images_dir=images_dir,
        image_col=cfg["data"]["image_col"],
        label_col=cfg["data"]["label_col"],
        transform=transform,
        num_classes=cfg["data"]["num_classes"],
    )
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=device.type == "cuda",
    )


def create_model(cfg, device, debug=False):
    backbone = cfg["model"]["backbone"]
    if "whitelist" in cfg["model"]:
        assert backbone in cfg["model"]["whitelist"], f"Backbone {backbone} is not in whitelist"

    try:
        return timm.create_model(
            backbone,
            pretrained=cfg["model"]["pretrained"],
            num_classes=cfg["data"]["num_classes"],
        ).to(device)
    except Exception:
        if cfg["model"]["pretrained"] and debug:
            print("Pretrained weights unavailable in debug; retry pretrained=False")
            return timm.create_model(
                backbone,
                pretrained=False,
                num_classes=cfg["data"]["num_classes"],
            ).to(device)
        raise


def load_model_from_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = timm.create_model(ckpt["backbone"], pretrained=False, num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp=False):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="Train"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, criterion, device, use_amp=False, desc="Eval"):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for images, labels in tqdm(loader, desc=desc):
        images = images.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        all_logits.append(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    labels = np.array(all_labels)
    preds = probs.argmax(axis=1)
    return {
        "loss": total_loss / len(loader.dataset),
        "logits": logits,
        "probs": probs,
        "labels": labels,
        "preds": preds,
        "macro_f1": f1_score(
            labels,
            preds,
            average="macro",
            labels=list(range(probs.shape[1])),
            zero_division=0,
        ),
        "accuracy": accuracy_score(labels, preds),
    }


def prediction_frame(df: pd.DataFrame, result: dict[str, Any], num_classes: int) -> pd.DataFrame:
    keep_cols = [
        "image_id_ext",
        "item_id",
        "result",
        "label",
        "fold",
        "source_dataset",
        "local_path",
        "content_hash",
    ]
    frame = df[[col for col in keep_cols if col in df.columns]].reset_index(drop=True)
    frame["target"] = result["labels"].astype(int)
    frame["pred"] = result["preds"].astype(int)
    for class_id in range(num_classes):
        frame[f"logit_{class_id}"] = result["logits"][:, class_id]
        frame[f"prob_{class_id}"] = result["probs"][:, class_id]
    return frame


def class_names_from_splits(splits: dict[str, Any], num_classes: int) -> list[str]:
    names = [str(index) for index in range(num_classes)]
    for fold_payload in splits["folds"]:
        for row in fold_payload["records"]:
            names[int(row["result"])] = str(row["label"])
    for row in splits.get("shadow_holdout", {}).get("records", []):
        names[int(row["result"])] = str(row["label"])
    return names


def metrics_from_frame(df: pd.DataFrame, num_classes: int) -> dict[str, Any]:
    labels = df["target"].to_numpy()
    preds = df["pred"].to_numpy()
    class_ids = list(range(num_classes))
    return {
        "rows": int(len(df)),
        "macro_f1": float(
            f1_score(labels, preds, average="macro", labels=class_ids, zero_division=0)
        ),
        "accuracy": float(accuracy_score(labels, preds)),
        "per_class_f1": f1_score(labels, preds, average=None, labels=class_ids, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds, labels=class_ids),
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def write_metrics_report(
    cfg,
    splits,
    oof_df,
    shadow_df,
    output_dir: Path,
    run_ids: dict[int, str],
    debug: bool,
):
    report_path = Path(cfg["artifacts"]["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = int(cfg["data"]["num_classes"])
    class_names = class_names_from_splits(splits, num_classes)
    oof_metrics = metrics_from_frame(oof_df, num_classes)
    shadow_metrics = metrics_from_frame(shadow_df, num_classes) if shadow_df is not None else None

    cm_path = output_dir / "confusion_matrix_oof.csv"
    pd.DataFrame(
        oof_metrics["confusion_matrix"],
        index=[f"true_{idx}" for idx in range(num_classes)],
        columns=[f"pred_{idx}" for idx in range(num_classes)],
    ).to_csv(cm_path)

    per_class_rows = []
    support = oof_df["target"].value_counts().reindex(range(num_classes), fill_value=0)
    for class_id in range(num_classes):
        per_class_rows.append(
            [
                class_id,
                class_names[class_id],
                int(support.loc[class_id]),
                f"{oof_metrics['per_class_f1'][class_id]:.4f}",
            ]
        )

    cm_rows = []
    for class_id, row in enumerate(oof_metrics["confusion_matrix"]):
        cm_rows.append([class_id, *[int(value) for value in row]])

    total_train_rows = int(splits["summary"]["train_pool_rows_after_filters"])
    total_shadow_rows = int(splits["summary"]["shadow_holdout_rows_after_filters"])
    coverage = f"{len(oof_df)}/{total_train_rows}"
    status = (
        "debug_smoke" if debug else "full_cv" if len(oof_df) == total_train_rows else "partial_cv"
    )
    full_oof_ready = len(oof_df) == total_train_rows
    shadow_ready = shadow_df is not None and len(shadow_df) == total_shadow_rows
    gate_decision = (
        "pending_u1_threshold" if full_oof_ready and shadow_ready else "pending_full_cv"
    )

    lines = [
        "# Baseline Metrics V1",
        "",
        "## Run",
        f"- generated_at_utc: `{current_utc()}`",
        f"- status: `{status}`",
        f"- debug: `{debug}`",
        f"- split_version: `{cfg['data']['split_version']}`",
        f"- dataset_version: `{cfg['data']['dataset_version']}`",
        f"- backbone: `{cfg['model']['backbone']}`",
        f"- image_size: `{cfg['data']['image_size']}`",
        f"- seed: `{cfg['train']['seed']}`",
        f"- folds: `{sorted(run_ids)}`",
        f"- mlflow_run_ids: `{run_ids}`",
        "",
        "## Quality Gate",
        f"- decision: `{gate_decision}`",
        "- threshold_source: `not_found_in_repo`",
        f"- oof_full_coverage: `{full_oof_ready}`",
        f"- shadow_holdout_metric_ready: `{shadow_ready}`",
        "- required_metrics: `macro_f1, per_class_f1, confusion_matrix, shadow_holdout_macro_f1`",
        "",
        "## OOF",
        f"- rows: `{len(oof_df)}`",
        f"- train_development_pool_coverage: `{coverage}`",
        f"- macro_f1: `{oof_metrics['macro_f1']:.6f}`",
        f"- accuracy: `{oof_metrics['accuracy']:.6f}`",
        f"- predictions: `{(output_dir / 'oof_predictions.parquet').as_posix()}`",
        f"- confusion_matrix_csv: `{cm_path.as_posix()}`",
        "",
        "## Per-Class F1",
        markdown_table(["class_id", "label", "support", "f1"], per_class_rows),
        "",
        "## Confusion Matrix OOF",
        markdown_table(["true/pred", *[str(idx) for idx in range(num_classes)]], cm_rows),
    ]

    if shadow_metrics is not None:
        lines.extend(
            [
                "",
                "## Shadow Holdout",
                f"- rows: `{len(shadow_df)}`",
                f"- macro_f1: `{shadow_metrics['macro_f1']:.6f}`",
                f"- accuracy: `{shadow_metrics['accuracy']:.6f}`",
                f"- predictions: `{(output_dir / 'shadow_holdout_predictions.parquet').as_posix()}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Re-run",
            "```bash",
            "uv run python src/training/train_image.py --config configs/model/image_baseline_v1.yaml --all-folds",
            "```",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def sqlite_path_from_uri(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme != "sqlite":
        return None
    if parsed.path.startswith("/"):
        return Path(parsed.path[1:] if not parsed.netloc else parsed.path)
    return Path(parsed.path)


def normalize_mlflow_experiment(cfg) -> str:
    mlflow_cfg = cfg["mlflow"]
    tracking_uri = mlflow_cfg["tracking_uri"]
    experiment_name = mlflow_cfg["experiment_name"]
    artifact_root = Path(
        mlflow_cfg.get(
            "artifact_root",
            cfg.get("artifacts", {}).get("roots", {}).get("mlflow", "artifacts/logs/mlruns"),
        )
    ).resolve()
    experiment_artifact_uri = (artifact_root / experiment_name).as_uri()

    db_path = sqlite_path_from_uri(tracking_uri)
    if db_path is not None and db_path.exists():
        with sqlite3.connect(db_path) as conn:
            for experiment_id, name, artifact_location in conn.execute(
                "select experiment_id, name, artifact_location from experiments"
            ).fetchall():
                expected_experiment_uri = (artifact_root / name).as_uri()
                if artifact_location != expected_experiment_uri:
                    conn.execute(
                        "update experiments set artifact_location = ? where experiment_id = ?",
                        (expected_experiment_uri, experiment_id),
                    )
                for run_id, artifact_uri in conn.execute(
                    "select run_uuid, artifact_uri from runs where experiment_id = ?",
                    (experiment_id,),
                ).fetchall():
                    expected_uri = (artifact_root / name / run_id / "artifacts").as_uri()
                    if artifact_uri != expected_uri:
                        conn.execute(
                            "update runs set artifact_uri = ? where run_uuid = ?",
                            (expected_uri, run_id),
                        )
                conn.commit()

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name, artifact_location=experiment_artifact_uri
        )
    else:
        experiment_id = experiment.experiment_id
    return str(experiment_id)


def log_mlflow_params(cfg, fold, device, debug):
    mlflow.log_params(
        {
            "fold": fold,
            "device": str(device),
            "debug": debug,
            "backbone": cfg["model"]["backbone"],
            "pretrained": cfg["model"]["pretrained"],
            "num_classes": cfg["data"]["num_classes"],
            "image_size": cfg["data"]["image_size"],
            "batch_size": cfg["train"]["batch_size"],
            "epochs": cfg["train"]["epochs"],
            "lr": cfg["train"]["lr"],
            "weight_decay": cfg["train"]["weight_decay"],
            "amp": cfg["train"]["amp"],
            "dataset_version": cfg["data"]["dataset_version"],
            "split_version": cfg["data"]["split_version"],
            "loss": cfg["experiment"]["loss"],
            "sampler": cfg["experiment"]["sampler"],
            "ratio_policy": cfg["experiment"]["ratio_policy"],
            "weak_label_flag": cfg["experiment"]["weak_label_flag"],
            "feature_flags": cfg["experiment"]["feature_flags"],
            "tta_flag": cfg["experiment"]["tta_flag"],
        }
    )
    mlflow.set_tags(
        {
            "commit_sha": git_commit_sha(),
            "dataset_version": cfg["data"]["dataset_version"],
            "split_version": cfg["data"]["split_version"],
            "seed": str(cfg["train"]["seed"]),
            "backbone": cfg["model"]["backbone"],
            "image_size": str(cfg["data"]["image_size"]),
            "loss": cfg["experiment"]["loss"],
            "sampler": cfg["experiment"]["sampler"],
            "ratio_policy": cfg["experiment"]["ratio_policy"],
            "weak_label_flag": str(cfg["experiment"]["weak_label_flag"]),
            "feature_flags": cfg["experiment"]["feature_flags"],
            "tta_flag": str(cfg["experiment"]["tta_flag"]),
        }
    )


def safe_log_artifact(path: Path, artifact_path: str | None = None) -> None:
    try:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as exc:
        print(f"MLflow artifact log failed: {exc}")


def run_name(cfg, fold: int) -> str:
    split_tag = cfg["data"]["split_version"].replace("splits_", "split")
    ds_tag = cfg["data"]["dataset_version"].replace("_", "")
    backbone_tag = slug(cfg["model"]["backbone"])
    image_size = cfg["data"]["image_size"]
    loss = cfg["experiment"]["loss"]
    sampler = cfg["experiment"]["sampler"]
    feature_flags = cfg["experiment"]["feature_flags"]
    seed = cfg["train"]["seed"]
    version = cfg["experiment"]["version"]
    return (
        f"rt_{split_tag}_{ds_tag}_{backbone_tag}_{image_size}_{loss}_"
        f"{sampler}_{feature_flags}_s{seed}_{version}_fold{fold}"
    )


def checkpoint_path(cfg, fold: int) -> Path:
    checkpoint_dir = Path(
        cfg.get("artifacts", {}).get("roots", {}).get("checkpoints", cfg["checkpoint"]["dir"])
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    backbone_tag = slug(cfg["model"]["backbone"])
    image_size = cfg["data"]["image_size"]
    version = cfg["experiment"]["version"]
    return checkpoint_dir / f"roomclf_{backbone_tag}_fold{fold}_{image_size}_{version}.ckpt"


def run_fold(
    args, cfg, splits, fold: int, experiment_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    set_seed(int(cfg["train"]["seed"]) + fold)
    device = get_device(cfg)
    print(f"Device: {device}")
    print(f"Fold: {fold}")

    train_df, valid_df = build_fold_frames(splits, fold)
    valid_df["fold"] = fold
    shadow_df = records_to_df(splits["shadow_holdout"]["records"])
    shadow_df["fold"] = fold

    if args.debug:
        print("DEBUG_MODE_ON")
        train_df = train_df.head(min(cfg["debug"]["train_samples"], len(train_df)))
        valid_df = valid_df.head(min(cfg["debug"]["val_samples"], len(valid_df)))
        shadow_df = shadow_df.head(min(cfg["debug"]["val_samples"], len(shadow_df)))
        cfg["train"]["epochs"] = cfg["debug"]["epochs"]
        cfg["train"]["num_workers"] = cfg["debug"]["num_workers"]

    train_transform, val_transform = get_transforms(cfg)
    train_loader = build_loader(
        train_df,
        cfg["data"]["images_train_dir"],
        train_transform,
        cfg,
        device,
        shuffle=True,
    )
    valid_loader = build_loader(
        valid_df,
        cfg["data"]["images_train_dir"],
        val_transform,
        cfg,
        device,
        shuffle=False,
    )
    shadow_loader = build_loader(
        shadow_df,
        cfg["data"]["images_val_dir"],
        val_transform,
        cfg,
        device,
        shuffle=False,
    )

    ckpt_path = checkpoint_path(cfg, fold)
    if cfg["checkpoint"].get("resume", False) and ckpt_path.exists():
        print("Loading model from checkpoint...")
        model = load_model_from_checkpoint(ckpt_path, device)
    else:
        print("Creating new model...")
        model = create_model(cfg, device, debug=args.debug)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1 = -1.0

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name(cfg, fold),
    ) as run:
        log_mlflow_params(cfg, fold, device, args.debug)

        for epoch in range(cfg["train"]["epochs"]):
            print(f"\nEpoch {epoch + 1}/{cfg['train']['epochs']}")
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
            )
            val_result = predict(
                model=model,
                loader=valid_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                desc="Eval",
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

            if val_result["macro_f1"] > best_f1:
                best_f1 = val_result["macro_f1"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "backbone": cfg["model"]["backbone"],
                        "num_classes": cfg["data"]["num_classes"],
                        "fold": fold,
                        "epoch": epoch,
                        "best_f1": best_f1,
                        "config": cfg,
                    },
                    ckpt_path,
                )
                safe_log_artifact(ckpt_path, artifact_path="checkpoints")
                print(f"Saved best checkpoint: {ckpt_path}")

        mlflow.log_metric("best_val_macro_f1", best_f1)

        model = load_model_from_checkpoint(ckpt_path, device)
        oof_result = predict(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            desc="OOF",
        )
        shadow_result = predict(
            model=model,
            loader=shadow_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            desc="Shadow",
        )

        oof_frame = prediction_frame(valid_df, oof_result, cfg["data"]["num_classes"])
        shadow_frame = prediction_frame(shadow_df, shadow_result, cfg["data"]["num_classes"])

        output_dir = Path(cfg["artifacts"]["oof_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        fold_oof_path = output_dir / f"fold{fold}_oof_predictions.parquet"
        fold_shadow_path = output_dir / f"fold{fold}_shadow_predictions.parquet"
        oof_frame.to_parquet(fold_oof_path, index=False)
        shadow_frame.to_parquet(fold_shadow_path, index=False)
        safe_log_artifact(fold_oof_path, artifact_path="oof")
        safe_log_artifact(fold_shadow_path, artifact_path="shadow")

        mlflow.log_metric("final_oof_macro_f1", oof_result["macro_f1"])
        mlflow.log_metric("final_shadow_macro_f1", shadow_result["macro_f1"])
        mlflow.log_metric("final_shadow_accuracy", shadow_result["accuracy"])

        return oof_frame, shadow_frame, run.info.run_id


def aggregate_shadow(shadow_frames: list[pd.DataFrame], num_classes: int) -> pd.DataFrame:
    if len(shadow_frames) == 1:
        return shadow_frames[0]

    key_cols = [
        "image_id_ext",
        "item_id",
        "result",
        "label",
        "source_dataset",
        "local_path",
        "content_hash",
    ]
    base_cols = [col for col in key_cols if col in shadow_frames[0].columns]
    merged = shadow_frames[0][base_cols + ["target"]].copy().reset_index(drop=True)

    probs = np.stack(
        [
            frame[[f"prob_{class_id}" for class_id in range(num_classes)]].to_numpy()
            for frame in shadow_frames
        ],
        axis=0,
    ).mean(axis=0)
    merged["fold"] = "ensemble"
    merged["pred"] = probs.argmax(axis=1).astype(int)
    for class_id in range(num_classes):
        merged[f"prob_{class_id}"] = probs[:, class_id]
    return merged


def write_final_config(cfg, output_dir: Path) -> Path:
    path = output_dir / "config.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    splits = load_splits(cfg["data"]["splits_json"])
    experiment_id = normalize_mlflow_experiment(cfg)

    if args.all_folds:
        folds = [int(fold_payload["fold"]) for fold_payload in splits["folds"]]
    else:
        folds = [args.fold]

    oof_frames: list[pd.DataFrame] = []
    shadow_frames: list[pd.DataFrame] = []
    run_ids: dict[int, str] = {}
    for fold in folds:
        fold_cfg = copy.deepcopy(cfg)
        oof_frame, shadow_frame, run_id = run_fold(args, fold_cfg, splits, fold, experiment_id)
        oof_frames.append(oof_frame)
        shadow_frames.append(shadow_frame)
        run_ids[fold] = run_id

    output_dir = Path(cfg["artifacts"]["oof_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_df = pd.concat(oof_frames, ignore_index=True)
    shadow_df = aggregate_shadow(shadow_frames, cfg["data"]["num_classes"])

    oof_path = output_dir / "oof_predictions.parquet"
    shadow_path = output_dir / "shadow_holdout_predictions.parquet"
    oof_df.to_parquet(oof_path, index=False)
    shadow_df.to_parquet(shadow_path, index=False)
    config_path = write_final_config(cfg, output_dir)

    write_metrics_report(
        cfg=cfg,
        splits=splits,
        oof_df=oof_df,
        shadow_df=shadow_df,
        output_dir=output_dir,
        run_ids=run_ids,
        debug=args.debug,
    )
    print(f"OOF -> {oof_path}")
    print(f"Shadow -> {shadow_path}")
    print(f"Config -> {config_path}")
    print(f"Report -> {cfg['artifacts']['report_path']}")


if __name__ == "__main__":
    main()
