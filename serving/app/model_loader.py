import os
from pathlib import Path

import torch
import yaml
import timm


def load_model():
    model_path = Path(os.getenv("MODEL_PATH", "releases/rc1/model.pt"))
    config_path = Path(os.getenv("CONFIG_PATH", "releases/rc1/config.yaml"))

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    backbone = cfg["model"]["backbone"]
    num_classes = cfg["data"]["num_classes"]
    image_size = cfg["data"]["image_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(
        backbone,
        pretrained=False,
        num_classes=num_classes,
    )

    checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    class_names = {
        i: f"class_{i}"
        for i in range(num_classes)
    }

    return model, device, image_size, class_names