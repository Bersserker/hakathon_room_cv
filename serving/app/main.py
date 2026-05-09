from fastapi import FastAPI, HTTPException

from serving.app.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
)

from serving.app.predict import predict_image
from serving.app.model_loader import load_model


app = FastAPI(
    title="Room Type Classification API",
    version="0.1.0",
)

model, device, image_size, class_names = load_model()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        return predict_image(
            model=model,
            image_path=request.image_path,
            device=device,
            image_size=image_size,
            class_names=class_names,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    predictions = []

    for image_path in request.image_paths:
        result = predict_image(
            model=model,
            image_path=image_path,
            device=device,
            image_size=image_size,
            class_names=class_names,
        )

        result["image_path"] = image_path
        predictions.append(result)

    return {"predictions": predictions}