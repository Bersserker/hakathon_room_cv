import yaml
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

import timm
from sklearn.metrics import f1_score, accuracy_score
import os

import mlflow

def log_mlflow_params(cfg, fold, device):
    mlflow.log_params({
        "fold": fold,
        "device": device,
        "backbone": cfg["model"]["backbone"],
        "pretrained": cfg["model"]["pretrained"],
        "num_classes": cfg["data"]["num_classes"],
        "batch_size": cfg["train"]["batch_size"],
        "epochs": cfg["train"]["epochs"],
        "lr": cfg["train"]["lr"],
        "weight_decay": cfg["train"]["weight_decay"],
        "amp": cfg["train"]["amp"],
    })

#загрузка конфиг файла
def load_config(path):
    with open (path, "r") as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

class RoomDataset(Dataset):
    def __init__(self, csv_path, 
                 images_dir, 
                 image_col, 
                 label_col, 
                 transform=None, 
                 num_classes=20):
        self.df = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.image_col = image_col
        self.labael_col = label_col
        self.transform = transform
        self.num_classes = num_classes
        

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = str(row[self.image_col])+'.jpg'
        label = int(row[self.labael_col])

        assert 0 <= label < self.num_classes, f"Bad label: {label}, idx={idx}, file={img_name}"

        img_path = self.images_dir / img_name
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, val_transform

def train_one_epoch(model,
                    loader,
                    criterion,
                    optimizer,
                    device,
                    scaler = None,
                    use_amp=False):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="Train"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=use_amp):
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
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Eval"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    accuracy = accuracy_score(all_labels, all_preds)

    return val_loss, macro_f1, accuracy

def load_model_from_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = timm.create_model(
        ckpt["backbone"],
        pretrained=False,  # важно!
        num_classes=ckpt["num_classes"]
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    return model

def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["train"]["seed"])

    device = cfg["train"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Device: {device}")

    fold = args.fold

    images_train_dir = cfg["data"]["images_train_dir"]
    images_val_dir = cfg["data"]["images_val_dir"]

    train_csv = Path(cfg["data"]["train_df"])
    val_csv = Path(cfg["data"]["val_df"])

    image_col = cfg["data"]["image_col"]
    label_col = cfg["data"]["label_col"]
    num_classes = cfg["data"]["num_classes"]

    train_transform, val_transform = get_transforms()

    train_dataset = RoomDataset(
        csv_path=train_csv,
        images_dir=images_train_dir,
        image_col=image_col,
        label_col=label_col,
        transform=train_transform,
        num_classes=num_classes,
    )

    val_dataset = RoomDataset(
        csv_path=val_csv,
        images_dir=images_val_dir,
        image_col=image_col,
        label_col=label_col,
        transform=val_transform,
        num_classes=num_classes,
    )

    if args.debug:
        print("DEBUG_MODE_ON")

        train_dataset = Subset(
            train_dataset,
            range(min(cfg["debug"]["train_samples"], len(train_dataset)))
        )

        val_dataset = Subset(
            val_dataset,
            range(min(cfg["debug"]["val_samples"], len(val_dataset)))
        )

        cfg["train"]["epochs"] = cfg["debug"]["epochs"]
        cfg["train"]["num_workers"] = cfg["debug"]["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True if device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True if device == "cuda" else False,
    )

    backbone = cfg["model"]["backbone"]

    if "whitelist" in cfg["model"]:
        assert backbone in cfg["model"]["whitelist"], (
            f"Backbone {backbone} is not in whitelist"
        )

    checkpoint_dir = Path(cfg["checkpoint"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"image_baseline_v1_fold{fold}_best.pt"

    resume = cfg["checkpoint"].get("resume", False)

    if resume and checkpoint_path.exists():
        print("Loading model from checkpoint...")
        model = load_model_from_checkpoint(checkpoint_path, device)
    else:
        print("Creating new model...")
        model = timm.create_model(
            backbone,
            pretrained=cfg["model"]["pretrained"],
            num_classes=num_classes,
        ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    use_amp = bool(cfg["train"]["amp"]) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    best_f1 = -1.0

    with mlflow.start_run(run_name=f"{backbone}_fold{fold}"):
        log_mlflow_params(cfg, fold, device)

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

            val_loss, val_f1, val_acc = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            print(
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_macro_f1={val_f1:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_macro_f1", val_f1, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            if val_f1 > best_f1:
                best_f1 = val_f1

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "backbone": backbone,
                        "num_classes": num_classes,
                        "fold": fold,
                        "epoch": epoch,
                        "best_f1": best_f1,
                        "config": cfg,
                    },
                    checkpoint_path,
                )

                mlflow.log_artifact(str(checkpoint_path))

                print(f"Saved best checkpoint: {checkpoint_path}")

        mlflow.log_metric("best_val_macro_f1", best_f1)

    print(f"\nBest Macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
