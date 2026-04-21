from __future__ import annotations

from pathlib import Path


DATASET_NAME = "OpenSafetyLab/Salad-Data"
SUBSET = "base_set"
TEXT_COLUMN = "question"
LABEL_COLUMN = "1-category"
MAX_SENTENCES = 3
MIN_LATIN_RATIO = 0.5
CACHE_DIR = Path("salad_cache")

