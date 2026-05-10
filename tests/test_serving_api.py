from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from serving.app import main


class FakeClassSchema:
    id_to_label = {3: "гостиная"}


class FakePredictor:
    cfg = {
        "release": {"candidate": "fake_model"},
        "data": {"image_size": 224},
        "model": {"checkpoints": [{"path": "fake.ckpt"}]},
    }
    class_schema = FakeClassSchema()
    device = "cpu"
    num_classes = 20

    def predict_image(self, image: Image.Image):
        assert image.mode == "RGB"
        probs = np.zeros((1, 20), dtype=np.float32)
        probs[0, 3] = 0.87
        return SimpleNamespace(preds=np.array([3]), probs=probs)


def make_jpeg() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(120, 90, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_predict_upload_returns_class_prediction(monkeypatch):
    monkeypatch.setattr(main, "load_model", lambda: FakePredictor())
    client = TestClient(main.app)

    response = client.post(
        "/predict_upload",
        files={"file": ("room.jpg", make_jpeg(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "class_id": 3,
        "class_name": "гостиная",
        "confidence": 0.8700000047683716,
    }


def test_predict_upload_rejects_non_image_file():
    client = TestClient(main.app)

    response = client.post(
        "/predict_upload",
        files={"file": ("room.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file must be an image"


def test_index_page_is_available():
    client = TestClient(main.app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Отправить" in response.text


def test_model_info_uses_loaded_predictor(monkeypatch):
    monkeypatch.setattr(main, "load_model", lambda: FakePredictor())
    client = TestClient(main.app)

    response = client.get("/model/info")

    assert response.status_code == 200
    assert response.json()["model_name"] == "fake_model"
    assert response.json()["checkpoints"] == 1
