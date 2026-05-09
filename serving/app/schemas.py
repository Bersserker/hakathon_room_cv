from pydantic import BaseModel, Field
from typing import List, Optional


class PredictRequest(BaseModel):
    image_path: str = Field(..., example="/app/data/example.jpg")


class PredictResponse(BaseModel):
    class_id: int
    class_name: Optional[str] = None
    confidence: float


class BatchPredictRequest(BaseModel):
    image_paths: List[str] = Field(..., example=[
        "/app/data/img1.jpg",
        "/app/data/img2.jpg",
    ])


class BatchPredictionItem(BaseModel):
    image_path: str
    class_id: int
    class_name: Optional[str] = None
    confidence: float


class BatchPredictResponse(BaseModel):
    predictions: List[BatchPredictionItem]


class ModelInfoResponse(BaseModel):
    model_name: str
    num_classes: int
    input_size: int
    framework: str
    status: str