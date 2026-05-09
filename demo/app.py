from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
from PIL import Image

try:
    from src.inference.room_predictor import RoomPredictor
except ModuleNotFoundError:  # pragma: no cover - keeps direct script execution working
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.inference.room_predictor import RoomPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio demo for room type classification.")
    parser.add_argument("--config", type=Path, default=Path("configs/release/rc1.yaml"))
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


class DemoPredictor:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.predictor = RoomPredictor.from_config_path(config_path)

    def __call__(self, image: Image.Image | None) -> tuple[dict[str, float], str]:
        if image is None:
            return {}, "Upload an image."
        result = self.predictor.topk_labels(image, k=3)
        info = (
            f"config={self.config_path} | "
            f"device={self.predictor.device} | "
            f"checkpoints={len(self.predictor.models)}"
        )
        return result, info


def main() -> None:
    args = parse_args()
    predictor = DemoPredictor(args.config)
    demo = gr.Interface(
        fn=predictor,
        inputs=gr.Image(type="pil", label="Room image"),
        outputs=[
            gr.Label(num_top_classes=3, label="Top-3 room classes"),
            gr.Textbox(label="Model version"),
        ],
        title="Room Type Classifier RC1",
        description="Upload a room/property image and get top-3 class probabilities.",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
