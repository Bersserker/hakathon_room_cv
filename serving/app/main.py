from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from serving.app.model_loader import get_model_info, load_model
from serving.app.predict import predict_image, predict_uploaded_image
from serving.app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)


app = FastAPI(
    title="Room Type Classification API",
    version="0.1.0",
)

INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"


def get_predictor():
    try:
        return load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model is not available: {exc}") from exc


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    return get_model_info(get_predictor())


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        return predict_image(get_predictor(), request.image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="File is not a valid image") from exc


@app.post("/predict_upload", response_model=PredictResponse)
async def predict_upload(file: Annotated[UploadFile, File(...)]):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        with Image.open(BytesIO(payload)) as image:
            rgb_image = image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    return predict_uploaded_image(get_predictor(), rgb_image)


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    predictor = get_predictor()
    predictions = []

    for image_path in request.image_paths:
        try:
            result = predict_image(predictor, image_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"File is not a valid image: {image_path}") from exc

        result["image_path"] = image_path
        predictions.append(result)

    return {"predictions": predictions}
