from __future__ import annotations

import json

from paths import path


def load_label_map() -> dict[str, int]:
    label_map_file = path("root", "label_map_file")
    if not label_map_file.exists():
        raise FileNotFoundError(f"Label map not found: {label_map_file}")

    with label_map_file.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {label_map_file}")

    label2id = {str(label): int(idx) for label, idx in data.items()}
    expected_ids = list(range(len(label2id)))
    actual_ids = sorted(label2id.values())
    if actual_ids != expected_ids:
        raise ValueError(f"Label ids must be contiguous starting at 0; got {actual_ids}")
    return label2id


LABEL2ID = load_label_map()
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
LABEL_NAMES = [ID2LABEL[idx] for idx in sorted(ID2LABEL)]

SENTIMENT_LABELS = ("neg", "neu", "pos")
SENTIMENT_LABEL2ID = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}
SENTIMENT_ID2LABEL = {idx: label for label, idx in SENTIMENT_LABEL2ID.items()}
