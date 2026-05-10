from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image_path: str = Field(..., json_schema_extra={"example": "/app/data/example.jpg"})


class PredictResponse(BaseModel):
    class_id: int
    class_name: str | None = None
    confidence: float


class BatchPredictRequest(BaseModel):
    image_paths: list[str] = Field(
        ...,
        json_schema_extra={"example": ["/app/data/img1.jpg", "/app/data/img2.jpg"]},
    )


class BatchPredictionItem(BaseModel):
    image_path: str
    class_id: int
    class_name: str | None = None
    confidence: float


class BatchPredictResponse(BaseModel):
    predictions: list[BatchPredictionItem]


class ModelInfoResponse(BaseModel):
    model_name: str
    num_classes: int
    input_size: int
    framework: str
    status: str
    device: str | None = None
    checkpoints: int | None = None
