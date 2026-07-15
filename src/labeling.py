"""Pure label helpers shared by training, evaluation, and prediction."""

from __future__ import annotations

import json
from pathlib import Path


DEFAULT_CLASS_INDICES = {"fire": 0, "no_fire": 1}


def labels_from_class_indices(class_indices: dict[str, int]) -> list[str]:
    return [label for label, _ in sorted(class_indices.items(), key=lambda item: item[1])]


def load_class_indices(model_path: str | Path) -> dict[str, int]:
    class_indices_path = Path(model_path).parent / "class_indices.json"
    if class_indices_path.exists():
        return json.loads(class_indices_path.read_text(encoding="utf-8"))
    return DEFAULT_CLASS_INDICES.copy()


def fire_probability_from_binary_output(raw_probability: float, class_indices: dict[str, int]) -> float:
    fire_index = class_indices.get("fire")
    if fire_index == 0:
        return 1.0 - raw_probability
    return raw_probability
