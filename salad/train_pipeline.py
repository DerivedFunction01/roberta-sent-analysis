from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from salad.cache import ensure_clean_salad_cache
from salad.defaults import CACHE_DIR, DATASET_NAME, LABEL_COLUMN, MAX_SENTENCES, MIN_LATIN_RATIO, SUBSET, TEXT_COLUMN


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    label_splits, cache_meta = ensure_clean_salad_cache(
        DATASET_NAME,
        SUBSET,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN,
        max_sentences=MAX_SENTENCES,
        min_latin_ratio=MIN_LATIN_RATIO,
        cache_dir=CACHE_DIR,
    )

    print("=" * 80)
    print("BUILDING SALAD-DATA SAFETY CACHE")
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME} / {SUBSET}")
    print(f"Text column: {TEXT_COLUMN}")
    print(f"Label column: {LABEL_COLUMN}")
    print(f"Max sentences: {MAX_SENTENCES}")
    print(f"Min Latin ratio: {MIN_LATIN_RATIO:.2f}")
    print(f"Output: {CACHE_DIR}")
    print(f"Kept rows: {cache_meta['filter_stats']['kept']}")
    print(f"Class counts: {cache_meta['label_counts']}")

    save_json(CACHE_DIR / "summary.json", cache_meta)
    print("Wrote Salad-Data cache files:")
    for label_name, dataset in label_splits.items():
        print(f"  {label_name}: {len(dataset):,} examples")


if __name__ == "__main__":
    main()
