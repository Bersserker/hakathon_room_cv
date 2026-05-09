from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from torchvision import transforms


CLASS_NAMES: Dict[int, str] = {
    0: "unknown",
    1: "class_1",
    2: "class_2",
    3: "class_3",
    4: "class_4",
    5: "class_5",
    6: "class_6",
    7: "class_7",
    8: "class_8",
    9: "class_9",
    10: "class_10",
    11: "class_11",
    12: "class_12",
    13: "class_13",
    14: "class_14",
    15: "class_15",
    16: "class_16",
    17: "class_17",
    18: "class_18",
    19: "class_19",
}


def build_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    image_size: int = 224,
    class_names: Optional[Dict[int, str]] = None,
) -> dict:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if class_names is None:
        class_names = CLASS_NAMES

    transform = build_transform(image_size)

    image = Image.open(path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)

        confidence, pred_class = torch.max(probs, dim=1)

    class_id = int(pred_class.item())

    return {
        "class_id": class_id,
        "class_name": class_names.get(class_id, f"class_{class_id}"),
        "confidence": float(confidence.item()),
    }