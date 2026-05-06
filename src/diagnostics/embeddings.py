from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd
from PIL import Image


class EmbeddingExtractor(Protocol):
    embedding_name: str

    def extract(self, image_paths: Sequence[Path]) -> np.ndarray:
        """Return one embedding row per image path."""


class ClipEmbeddingExtractor:
    """Frozen CLIP image embedding extractor loaded from local cache only."""

    embedding_name = "clip"

    def __init__(
        self,
        model_name_or_path: str = "openai/clip-vit-base-patch32",
        *,
        processor_name_or_path: str | None = None,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import CLIPModel
        except ImportError as exc:  # pragma: no cover - depends on local runtime extras
            raise RuntimeError(
                "CLIP embeddings require local torch and transformers installations."
            ) from exc

        model_path, processor_path = resolve_clip_paths(model_name_or_path, processor_name_or_path)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = load_clip_processor(processor_path)
            self.model = CLIPModel.from_pretrained(
                model_path,
                local_files_only=True,
            )
        except Exception as exc:  # pragma: no cover - message exercised manually
            raise RuntimeError(
                "Could not load local frozen CLIP model/processor. "
                f"Requested model `{model_path}` and processor `{processor_path}` with "
                "local_files_only=True; no external download was attempted. Pass "
                "--clip-model-name/--clip-processor-name to local paths/cache ids or "
                "precompute --embedding-cache."
            ) from exc
        self.model.to(self.device)
        self.model.eval()

    def extract(self, image_paths: Sequence[Path]) -> np.ndarray:
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            features = self.model.get_image_features(**inputs)
            if not isinstance(features, self.torch.Tensor):
                if getattr(features, "image_embeds", None) is not None:
                    features = features.image_embeds
                elif getattr(features, "pooler_output", None) is not None:
                    features = features.pooler_output
                    if (
                        hasattr(self.model, "visual_projection")
                        and features.shape[-1] == self.model.visual_projection.in_features
                    ):
                        features = self.model.visual_projection(features)
                else:
                    features = features[0]
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.detach().cpu().numpy().astype(np.float32)


def resolve_clip_paths(
    model_name_or_path: str,
    processor_name_or_path: str | None = None,
) -> tuple[str, str]:
    """Resolve repo-local split CLIP save dirs before falling back to HF cache ids."""
    if processor_name_or_path is not None:
        return model_name_or_path, processor_name_or_path

    model_path = Path(model_name_or_path)
    sibling_processor = model_path.parent / "saved_clip_processor"
    if model_path.exists() and model_path.name == "saved_clip_model" and sibling_processor.exists():
        return model_name_or_path, sibling_processor.as_posix()

    repo_model = Path("models/saved_clip_model")
    repo_processor = Path("models/saved_clip_processor")
    if model_name_or_path == "openai/clip-vit-base-patch32" and repo_model.exists() and repo_processor.exists():
        return repo_model.as_posix(), repo_processor.as_posix()

    return model_name_or_path, model_name_or_path


def load_clip_processor(processor_name_or_path: str):
    """Load CLIPProcessor, including the repo's split processor_config.json format."""
    from transformers import CLIPImageProcessor, CLIPProcessor, CLIPTokenizer

    try:
        return CLIPProcessor.from_pretrained(processor_name_or_path, local_files_only=True)
    except OSError:
        processor_path = Path(processor_name_or_path)
        processor_config_path = processor_path / "processor_config.json"
        if not processor_config_path.exists():
            raise
        config = json.loads(processor_config_path.read_text(encoding="utf-8"))
        image_processor_config = config.get("image_processor")
        if not isinstance(image_processor_config, dict):
            raise
        image_processor = CLIPImageProcessor(**image_processor_config)
        tokenizer = CLIPTokenizer.from_pretrained(processor_path, local_files_only=True)
        return CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)


def _embedding_columns(dimension: int) -> list[str]:
    return [f"emb_{index:04d}" for index in range(dimension)]


def load_embedding_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame()
    if cache_path.suffix.lower() == ".csv":
        cache = pd.read_csv(cache_path)
    else:
        cache = pd.read_parquet(cache_path)
    if "image_id_ext" not in cache.columns:
        raise ValueError(f"Embedding cache {cache_path} missing image_id_ext column.")
    cache = cache.copy()
    cache["image_id_ext"] = cache["image_id_ext"].astype(str)
    return cache.drop_duplicates("image_id_ext", keep="last").reset_index(drop=True)


def save_embedding_cache(cache_path: Path, cache: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.suffix.lower() == ".csv":
        cache.to_csv(cache_path, index=False)
    else:
        cache.to_parquet(cache_path, index=False)


def ensure_embeddings(
    df: pd.DataFrame,
    *,
    extractor: EmbeddingExtractor,
    cache_path: Path | None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Load reusable embeddings and extract only missing image ids."""
    required = {"image_id_ext", "local_path"}
    missing_columns = sorted(required.difference(df.columns))
    if missing_columns:
        raise ValueError(f"embedding dataframe missing required columns: {missing_columns}")

    cache = load_embedding_cache(cache_path) if cache_path is not None else pd.DataFrame()
    cached_ids = set(cache["image_id_ext"].astype(str)) if not cache.empty else set()

    unique_rows = (
        df[["image_id_ext", "local_path"]]
        .drop_duplicates("image_id_ext", keep="first")
        .assign(image_id_ext=lambda frame: frame["image_id_ext"].astype(str))
        .reset_index(drop=True)
    )
    missing_rows = unique_rows.loc[~unique_rows["image_id_ext"].isin(cached_ids)].copy()

    new_tables: list[pd.DataFrame] = []
    if not missing_rows.empty:
        paths = [Path(str(value)) for value in missing_rows["local_path"].tolist()]
        absent = [str(path) for path in paths if not path.exists()]
        if absent:
            raise FileNotFoundError(f"Missing image paths for embedding extraction: {absent[:10]}")

        for start in range(0, len(missing_rows), batch_size):
            batch_rows = missing_rows.iloc[start : start + batch_size]
            batch_paths = [Path(str(value)) for value in batch_rows["local_path"].tolist()]
            vectors = extractor.extract(batch_paths)
            if vectors.ndim != 2 or vectors.shape[0] != len(batch_rows):
                raise ValueError(
                    "Embedding extractor returned an array with invalid shape: "
                    f"{vectors.shape}, expected ({len(batch_rows)}, dim)."
                )
            emb_cols = _embedding_columns(vectors.shape[1])
            table = pd.DataFrame(vectors, columns=emb_cols)
            table.insert(0, "image_id_ext", batch_rows["image_id_ext"].to_numpy())
            table.insert(1, "embedding_name", getattr(extractor, "embedding_name", "unknown"))
            new_tables.append(table)

    if new_tables:
        cache = pd.concat([cache, *new_tables], ignore_index=True, sort=False)
        cache = cache.drop_duplicates("image_id_ext", keep="last").reset_index(drop=True)
        if cache_path is not None:
            save_embedding_cache(cache_path, cache)

    if cache.empty:
        raise ValueError("No embeddings are available after cache/extraction step.")

    cache["image_id_ext"] = cache["image_id_ext"].astype(str)
    return cache
