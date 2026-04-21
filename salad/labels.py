from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from paths import path


OUTSIDE_LABEL = "outside"
LABEL_MAP_FILE = path("salad", "salad_label_map_file")


def slugify_label(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    slug = slug.strip("_")
    return slug or "label"


def build_label_map(category_labels: list[str]) -> dict[str, int]:
    labels = ["O"]
    for category in category_labels:
        slug = slugify_label(category)
        labels.extend([f"B-{slug}", f"I-{slug}"])
    return {label: idx for idx, label in enumerate(labels)}


def save_label_map(category_labels: list[str], file_path: Path = LABEL_MAP_FILE) -> dict[str, int]:
    label2id = build_label_map(category_labels)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    return label2id


def load_label_map(file_path: Path = LABEL_MAP_FILE) -> dict[str, int]:
    if not file_path.exists():
        raise FileNotFoundError(f"Label map not found: {file_path}")
    with file_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {file_path}")
    label2id = {str(label): int(idx) for label, idx in data.items()}
    expected_ids = list(range(len(label2id)))
    actual_ids = sorted(label2id.values())
    if actual_ids != expected_ids:
        raise ValueError(f"Label ids must be contiguous starting at 0; got {actual_ids}")
    return label2id


def id2label(label2id: dict[str, int]) -> dict[int, str]:
    return {idx: label for label, idx in label2id.items()}


def normalize_label(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int):
        return str(value)
    raise TypeError(f"Unsupported label type: {type(value)!r}")
