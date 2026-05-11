from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.inference.room_predictor import (
        RoomPredictor,
        TestImageDataset,
        build_test_loader,
        build_val_transform,
        checkpoint_paths,
        get_device,
        image_filename,
        load_class_bias,
        load_model,
        load_yaml,
        predict_logits,
        prediction_frame,
        save_predictions,
        set_deterministic,
        submission_frame,
    )
    from src.inference.validate_submission import validate_submission
except ModuleNotFoundError:  # pragma: no cover - keeps direct script execution working
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.inference.room_predictor import (
        RoomPredictor,
        TestImageDataset,
        build_test_loader,
        build_val_transform,
        checkpoint_paths,
        get_device,
        image_filename,
        load_class_bias,
        load_model,
        load_yaml,
        predict_logits,
        prediction_frame,
        save_predictions,
        set_deterministic,
        submission_frame,
    )
    from src.inference.validate_submission import validate_submission


__all__ = [
    "RoomPredictor",
    "TestImageDataset",
    "build_test_loader",
    "build_val_transform",
    "checkpoint_paths",
    "get_device",
    "image_filename",
    "load_class_bias",
    "load_model",
    "load_yaml",
    "parse_args",
    "predict_logits",
    "prediction_frame",
    "run_inference",
    "save_predictions",
    "set_deterministic",
    "submission_frame",
    "validate_submission",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic room-classification inference."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def run_inference(config_path: Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    data_cfg = cfg.get("data", {})
    inference_cfg = cfg.get("inference", {})

    predictor = RoomPredictor.from_config_path(config_path)
    test_csv = Path(data_cfg.get("test_csv", "data/raw/test_df.csv"))
    images_dir = Path(data_cfg.get("images_test_dir", "data/raw/test_images/test_images"))

    test_df = pd.read_csv(test_csv, dtype={"image_id_ext": "string"})
    if "image_id_ext" not in test_df.columns:
        raise ValueError(f"{test_csv} missing image_id_ext column")

    loader = build_test_loader(
        test_df,
        images_dir=images_dir,
        transform=predictor.transform,
        batch_size=int(inference_cfg.get("batch_size", 64)),
        num_workers=int(inference_cfg.get("num_workers", 0)),
        pin_memory=predictor.device.type == "cuda",
    )
    batch = predictor.predict_loader(loader)

    output_submission = Path(inference_cfg.get("output_submission", "releases/rc1/submission.csv"))
    output_predictions = Path(
        inference_cfg.get("output_predictions", "releases/rc1/predictions.parquet")
    )
    output_submission.parent.mkdir(parents=True, exist_ok=True)

    submission = submission_frame(test_df["image_id_ext"], batch.preds)
    submission.to_csv(output_submission, index=False)
    save_predictions(output_predictions, prediction_frame(batch, predictor.num_classes))

    validation_result = None
    class_mapping = data_cfg.get("class_mapping", "configs/data/class_mapping.yaml")
    if bool(inference_cfg.get("validate_after", True)):
        validation_result = validate_submission(output_submission, test_csv, Path(class_mapping))

    checkpoint_items = cfg.get("model", {}).get("checkpoints") or [
        cfg.get("model", {}).get("checkpoint")
    ]
    checkpoints = [
        str(item["path"] if isinstance(item, dict) else item) for item in checkpoint_items if item
    ]

    return {
        "submission": str(output_submission),
        "predictions": str(output_predictions),
        "checkpoints": checkpoints,
        "device": str(predictor.device),
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
