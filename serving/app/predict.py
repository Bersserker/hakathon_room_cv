from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.inference.room_predictor import RoomPredictor


def _prediction_response(predictor: RoomPredictor, image: Image.Image) -> dict:
    batch = predictor.predict_image(image)
    class_id = int(batch.preds[0])
    confidence = float(batch.probs[0][class_id])

    return {
        "class_id": class_id,
        "class_name": predictor.class_schema.id_to_label.get(class_id, f"class_{class_id}"),
        "confidence": confidence,
    }


def predict_image(predictor: RoomPredictor, image_path: str) -> dict:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(path) as image:
        return _prediction_response(predictor, image.convert("RGB"))


def predict_uploaded_image(predictor: RoomPredictor, image: Image.Image) -> dict:
    return _prediction_response(predictor, image.convert("RGB"))
