from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import timm
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.inference.validate_submission import normalize_image_id, validate_submission

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic room-classification inference."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


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
    text = normalize_image_id(value)
    return text if Path(text).suffix else f"{text}.jpg"


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
        path = self.images_dir / image_filename(row["image_id_ext"])
        if not path.exists():
            raise FileNotFoundError(f"Test image not found: {path}")
        image = Image.open(path).convert("RGB")
        return self.transform(image), image_id


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


def save_predictions(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def run_inference(config_path: Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    data_cfg = cfg.get("data", {})
    inference_cfg = cfg.get("inference", {})

    seed = int(inference_cfg.get("seed", cfg.get("train", {}).get("seed", 42)))
    set_deterministic(seed)

    device = get_device(str(inference_cfg.get("device", "cuda")))
    num_classes = int(data_cfg.get("num_classes", 20))
    image_size = int(data_cfg.get("image_size", 224))
    resize_size = int(data_cfg.get("resize_size", 256))
    test_csv = Path(data_cfg.get("test_csv", "data/raw/test_df.csv"))
    images_dir = Path(data_cfg.get("images_test_dir", "data/raw/test_images/test_images"))

    test_df = pd.read_csv(test_csv, dtype={"image_id_ext": "string"})
    if "image_id_ext" not in test_df.columns:
        raise ValueError(f"{test_csv} missing image_id_ext column")

    transform = build_val_transform(image_size=image_size, resize_size=resize_size)
    dataset = TestImageDataset(test_df, images_dir=images_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=int(inference_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(inference_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    logits_sum: np.ndarray | None = None
    paths = checkpoint_paths(cfg)
    for checkpoint_path in paths:
        print(f"Loading checkpoint: {checkpoint_path}")
        model = load_model(checkpoint_path, device, cfg)
        logits = predict_logits(
            model=model,
            loader=loader,
            device=device,
            use_tta=bool(inference_cfg.get("tta", False)),
        )
        logits_sum = logits if logits_sum is None else logits_sum + logits
        del model

    assert logits_sum is not None
    logits = logits_sum / len(paths)
    logits_tensor = torch.from_numpy(logits).to(device)
    class_bias = load_class_bias(cfg, num_classes=num_classes, device=device)
    if class_bias is not None:
        logits_tensor = logits_tensor + class_bias

    probs = torch.softmax(logits_tensor, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1).astype(int)

    output_submission = Path(inference_cfg.get("output_submission", "releases/rc1/submission.csv"))
    output_predictions = Path(
        inference_cfg.get("output_predictions", "releases/rc1/predictions.parquet")
    )
    output_submission.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame(
        {
            "image_id_ext": test_df["image_id_ext"].map(normalize_image_id),
            "Predicted": preds,
        }
    )
    submission.to_csv(output_submission, index=False)

    prediction_frame = submission.copy()
    for class_id in range(num_classes):
        prediction_frame[f"logit_{class_id}"] = logits[:, class_id]
        prediction_frame[f"prob_{class_id}"] = probs[:, class_id]
    save_predictions(output_predictions, prediction_frame)

    validation_result = None
    class_mapping = data_cfg.get("class_mapping", "configs/data/class_mapping.yaml")
    if bool(inference_cfg.get("validate_after", True)):
        validation_result = validate_submission(output_submission, test_csv, Path(class_mapping))

    return {
        "submission": str(output_submission),
        "predictions": str(output_predictions),
        "checkpoints": [str(path) for path in paths],
        "device": str(device),
        "rows": int(len(submission)),
        "validation": validation_result,
    }


def main() -> None:
    result = run_inference(parse_args().config)
    print("Inference complete")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
