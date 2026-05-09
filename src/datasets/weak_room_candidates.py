from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.room_data_contract import image_id_with_extension

SOURCE_POLICY: dict[str, dict[str, Any]] = {
    "heuristics_cabinet": {"class_id": 5, "source": "heuristics_cabinet"},
    "heuristics_detskaya": {"class_id": 6, "source": "heuristics_detskaya"},
    "heuristics_dressing_room": {
        "class_id": 11,
        "source": "heuristics_dressing_room",
    },
}


def normalize_image_id_ext(value: Any) -> str:
    return image_id_with_extension(value)


def source_name_from_path(path: Path, *, kind: str) -> str:
    source = path.stem
    if source not in SOURCE_POLICY:
        known = ", ".join(sorted(SOURCE_POLICY))
        raise ValueError(f"Unknown {kind} heuristic source {source!r}. Known: {known}")
    return source


def source_to_class_id() -> dict[str, int]:
    return {source: int(policy["class_id"]) for source, policy in SOURCE_POLICY.items()}
