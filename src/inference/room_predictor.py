from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import timm
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.training.config_loader import load_release_config
from src.utils.room_data_contract import ClassSchema, image_id_with_extension, normalize_image_id

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class PredictionBatch:
    image_ids: list[str]
    logits: np.ndarray
    probs: np.ndarray
    preds: np.ndarray


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def image_filename(value: Any) -> str:
    return image_id_with_extension(value)


def build_val_transform(image_size: int, resize_size: int):
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def checkpoint_paths(cfg: dict[str, Any]) -> list[Path]:
    checkpoints = cfg.get("model", {}).get("checkpoints")
    if not checkpoints:
        checkpoint = cfg.get("model", {}).get("checkpoint")
        checkpoints = [checkpoint] if checkpoint else []
    paths = [Path(item["path"] if isinstance(item, dict) else item) for item in checkpoints]
    if not paths:
        raise ValueError("Inference config must define model.checkpoint or model.checkpoints")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"Checkpoint files not found: {missing}")
    return paths


def load_model(path: Path, device: torch.device, cfg: dict[str, Any]) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    backbone = checkpoint.get("backbone", cfg.get("model", {}).get("backbone"))
    num_classes = int(checkpoint.get("num_classes", cfg.get("data", {}).get("num_classes", 20)))
    if backbone is None:
        raise ValueError(f"Cannot infer backbone for checkpoint: {path}")

    model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_class_bias(
    cfg: dict[str, Any], num_classes: int, device: torch.device
) -> torch.Tensor | None:
    postprocess = cfg.get("postprocess", {}) or {}
    bias = postprocess.get("class_bias")

    bias_path = postprocess.get("class_bias_path")
    if bias_path:
        payload = load_yaml(Path(bias_path))
        bias = payload.get("class_bias", payload.get("bias", bias))

    if bias is None:
        return None
    if len(bias) != num_classes:
        raise ValueError(f"class_bias length must be {num_classes}, got {len(bias)}")
    return torch.tensor(bias, dtype=torch.float32, device=device).view(1, -1)


class TestImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: Path, transform) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = self.df.iloc[index]
        image_id = normalize_image_id(row["image_id_ext"])
        path = self.images_dir / image_id_with_extension(row["image_id_ext"])
        if not path.exists():
            raise FileNotFoundError(f"Test image not found: {path}")
        image = Image.open(path).convert("RGB")
        return self.transform(image), image_id


class RoomPredictor:
    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        device: torch.device,
        models: list[torch.nn.Module],
        transform,
        class_bias: torch.Tensor | None,
        use_tta: bool,
        class_schema: ClassSchema,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.models = models
        self.transform = transform
        self.class_bias = class_bias
        self.use_tta = use_tta
        self.class_schema = class_schema
        self.num_classes = class_schema.num_classes

    @classmethod
    def from_config_path(cls, config_path: Path) -> RoomPredictor:
        cfg = load_release_config(config_path)
        data_cfg = cfg.get("data", {})
        inference_cfg = cfg.get("inference", {})
        seed = int(inference_cfg.get("seed", cfg.get("train", {}).get("seed", 42)))
        set_deterministic(seed)

        device = get_device(str(inference_cfg.get("device", "cuda")))
        class_mapping = Path(data_cfg.get("class_mapping", "configs/data/class_mapping.yaml"))
        class_schema = ClassSchema.from_yaml(class_mapping)
        num_classes = int(data_cfg.get("num_classes", class_schema.num_classes))
        if num_classes != class_schema.num_classes:
            raise ValueError(
                f"data.num_classes={num_classes} does not match class schema {class_schema.num_classes}"
            )

        transform = build_val_transform(
            image_size=int(data_cfg.get("image_size", 224)),
            resize_size=int(data_cfg.get("resize_size", 256)),
        )
        models = [load_model(path, device, cfg) for path in checkpoint_paths(cfg)]
        class_bias = load_class_bias(cfg, num_classes=num_classes, device=device)
        return cls(
            cfg=cfg,
            device=device,
            models=models,
            transform=transform,
            class_bias=class_bias,
            use_tta=bool(inference_cfg.get("tta", False)),
            class_schema=class_schema,
        )

    @torch.no_grad()
    def logits_for_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        logits_sum = None
        for model in self.models:
            logits = model(tensor)
            if self.use_tta:
                logits = (logits + model(torch.flip(tensor, dims=[-1]))) / 2.0
            logits_sum = logits if logits_sum is None else logits_sum + logits
        if logits_sum is None:
            raise ValueError("RoomPredictor has no loaded models")
        logits = logits_sum / len(self.models)
        if self.class_bias is not None:
            logits = logits + self.class_bias
        return logits

    @torch.no_grad()
    def predict_loader(self, loader: DataLoader) -> PredictionBatch:
        image_ids: list[str] = []
        chunks: list[np.ndarray] = []
        for images, batch_ids in tqdm(loader, desc="Infer"):
            images = images.to(self.device)
            logits = self.logits_for_tensor(images)
            chunks.append(logits.detach().cpu().numpy())
            image_ids.extend(str(value) for value in batch_ids)
        logits_np = np.concatenate(chunks, axis=0)
        probs = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()
        preds = probs.argmax(axis=1).astype(int)
        return PredictionBatch(image_ids=image_ids, logits=logits_np, probs=probs, preds=preds)

    @torch.no_grad()
    def predict_image(self, image: Image.Image) -> PredictionBatch:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.logits_for_tensor(tensor).detach().cpu().numpy()
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        preds = probs.argmax(axis=1).astype(int)
        return PredictionBatch(
            image_ids=["uploaded_image"], logits=logits, probs=probs, preds=preds
        )

    def topk_labels(self, image: Image.Image, k: int = 3) -> dict[str, float]:
        batch = self.predict_image(image)
        probs = torch.from_numpy(batch.probs[0])
        top_k = min(int(k), len(probs))
        top_probs, top_ids = torch.topk(probs, k=top_k)
        return {
            f"{int(class_id)} — {self.class_schema.id_to_label[int(class_id)]}": float(prob)
            for class_id, prob in zip(top_ids.tolist(), top_probs.tolist(), strict=True)
        }


@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for images, _image_ids in tqdm(loader, desc="Infer"):
        images = images.to(device)
        logits = model(images)
        if use_tta:
            flipped_logits = model(torch.flip(images, dims=[-1]))
            logits = (logits + flipped_logits) / 2.0
        chunks.append(logits.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def build_test_loader(
    test_df: pd.DataFrame,
    images_dir: Path,
    transform,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TestImageDataset(test_df, images_dir=images_dir, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def prediction_frame(batch: PredictionBatch, num_classes: int) -> pd.DataFrame:
    frame = pd.DataFrame({"image_id_ext": batch.image_ids, "Predicted": batch.preds})
    for class_id in range(num_classes):
        frame[f"logit_{class_id}"] = batch.logits[:, class_id]
        frame[f"prob_{class_id}"] = batch.probs[:, class_id]
    return frame


def submission_frame(image_ids: Iterable[Any], preds: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id_ext": [normalize_image_id(value) for value in image_ids],
            "Predicted": preds.astype(int),
        }
    )


def save_predictions(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
