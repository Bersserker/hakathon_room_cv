from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import torch
import yaml
from PIL import Image

from src.inference.predict import (
    build_val_transform,
    checkpoint_paths,
    get_device,
    load_class_bias,
    load_model,
    load_yaml,
    set_deterministic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio demo for room type classification.")
    parser.add_argument("--config", type=Path, default=Path("configs/release/rc1.yaml"))
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def load_labels(path: Path) -> dict[int, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {int(key): str(value) for key, value in payload["id_to_label"].items()}


class DemoPredictor:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.cfg = load_yaml(config_path)
        data_cfg = self.cfg.get("data", {})
        inference_cfg = self.cfg.get("inference", {})
        set_deterministic(int(inference_cfg.get("seed", 42)))
        self.device = get_device(str(inference_cfg.get("device", "cuda")))
        self.num_classes = int(data_cfg.get("num_classes", 20))
        self.labels = load_labels(
            Path(data_cfg.get("class_mapping", "configs/data/class_mapping.yaml"))
        )
        self.transform = build_val_transform(
            image_size=int(data_cfg.get("image_size", 224)),
            resize_size=int(data_cfg.get("resize_size", 256)),
        )
        self.models = [
            load_model(path, self.device, self.cfg) for path in checkpoint_paths(self.cfg)
        ]
        self.class_bias = load_class_bias(self.cfg, self.num_classes, self.device)
        self.use_tta = bool(inference_cfg.get("tta", False))

    @torch.no_grad()
    def __call__(self, image: Image.Image | None) -> tuple[dict[str, float], str]:
        if image is None:
            return {}, "Upload an image."
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits_sum = None
        for model in self.models:
            logits = model(tensor)
            if self.use_tta:
                logits = (logits + model(torch.flip(tensor, dims=[-1]))) / 2.0
            logits_sum = logits if logits_sum is None else logits_sum + logits
        logits = logits_sum / len(self.models)
        if self.class_bias is not None:
            logits = logits + self.class_bias
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()
        top_probs, top_ids = torch.topk(probs, k=3)
        result = {
            f"{int(class_id)} — {self.labels[int(class_id)]}": float(prob)
            for class_id, prob in zip(top_ids.tolist(), top_probs.tolist(), strict=True)
        }
        info = f"config={self.config_path} | device={self.device} | checkpoints={len(self.models)}"
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
