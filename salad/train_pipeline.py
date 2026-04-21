from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from datasets import DatasetDict, concatenate_datasets
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from paths import path
from salad.cache import ensure_clean_salad_cache, ensure_openhermes_outside_cache
from salad.data import build_tokenized_split
from salad.defaults import (
    BALANCED_COVERAGE_RATIO,
    CACHE_DIR,
    DATASET_NAME,
    LABEL_COLUMN,
    MAX_LENGTH,
    MAX_SENTENCES,
    MIXED_CLASS_RATIO,
    MIN_LATIN_RATIO,
    NEUTRAL_CACHE_DIR,
    NEUTRAL_DATASET_NAME,
    NEUTRAL_MAX_SENTENCES,
    NEUTRAL_MIN_LATIN_RATIO,
    NEUTRAL_SAMPLE_FRACTION,
    NEUTRAL_SPLIT,
    NEUTRAL_TEXT_COLUMN,
    PIPELINE_RESULTS_DIR,
    REUSE_LIMIT,
    SAME_CLASS_RATIO,
    STANDALONE_RATIO,
    TEST_EXAMPLES,
    TOKENIZED_DATASET_DIR,
    TOKENIZED_DATASET_META,
    TRAIN_EXAMPLES,
    VALIDATION_EXAMPLES,
    SUBSET,
    TEXT_COLUMN,
)
from salad.labels import OUTSIDE_LABEL, save_label_map, slugify_label


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    unsafe_label_splits, cache_meta = ensure_clean_salad_cache(
        DATASET_NAME,
        SUBSET,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN,
        max_sentences=MAX_SENTENCES,
        min_latin_ratio=MIN_LATIN_RATIO,
        cache_dir=CACHE_DIR,
    )
    outside_split, outside_meta = ensure_openhermes_outside_cache(
        NEUTRAL_DATASET_NAME,
        split_name=NEUTRAL_SPLIT,
        conversations_column=NEUTRAL_TEXT_COLUMN,
        max_sentences=NEUTRAL_MAX_SENTENCES,
        min_latin_ratio=NEUTRAL_MIN_LATIN_RATIO,
        sample_fraction=NEUTRAL_SAMPLE_FRACTION,
        cache_dir=NEUTRAL_CACHE_DIR,
    )

    category_labels = list(cache_meta["label_names"])
    label2id = save_label_map(category_labels)
    id2label = {idx: label for label, idx in label2id.items()}
    category_to_slug = {label: slugify_label(label) for label in category_labels}

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-2021-124m", use_fast=True)

    combined_split = concatenate_datasets([*unsafe_label_splits.values(), outside_split])

    if TOKENIZED_DATASET_DIR.exists():
        shutil.rmtree(TOKENIZED_DATASET_DIR)
    TOKENIZED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BUILDING SALAD TOKEN CLASSIFICATION CACHE")
    print("=" * 80)
    print(f"Unsafe dataset: {DATASET_NAME} / {SUBSET}")
    print(f"Unsafe labels: {category_labels}")
    print(f"Outside dataset: {NEUTRAL_DATASET_NAME} / {NEUTRAL_SPLIT}")
    print(f"Tokenizer: cardiffnlp/twitter-roberta-base-2021-124m")
    print(f"Output: {TOKENIZED_DATASET_DIR}")
    print(f"Label map: {path('salad', 'salad_label_map_file')}")
    print(f"Unsafe kept rows: {cache_meta['filter_stats']['kept']}")
    print(f"Outside kept rows: {outside_meta['filter_stats']['kept']}")

    split_specs = [
        ("train", TRAIN_EXAMPLES),
        ("validation", VALIDATION_EXAMPLES),
        ("test", TEST_EXAMPLES),
    ]

    tokenized_splits = {}
    split_summaries = {}
    for split_name, num_examples in tqdm(split_specs, desc="Building salad splits"):
        split, summary = build_tokenized_split(
            combined_split,
            num_examples=num_examples,
            standalone_ratio=STANDALONE_RATIO,
            same_class_ratio=SAME_CLASS_RATIO,
            mixed_class_ratio=MIXED_CLASS_RATIO,
            balanced_coverage_ratio=BALANCED_COVERAGE_RATIO,
            reuse_limit=REUSE_LIMIT,
            seed=42,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
            label2id=label2id,
            category_labels=category_labels,
            text_column="text",
            label_column="label",
        )
        tokenized_splits[split_name] = split
        split_summaries[split_name] = summary
        save_json(TOKENIZED_DATASET_DIR / f"{split_name}_summary.json", summary)

    dataset_dict = DatasetDict(tokenized_splits)
    dataset_dict.save_to_disk(str(TOKENIZED_DATASET_DIR))
    save_json(
        TOKENIZED_DATASET_META,
        {
            "dataset_name": DATASET_NAME,
            "subset": SUBSET,
            "unsafe_cache_dir": str(CACHE_DIR),
            "outside_cache_dir": str(NEUTRAL_CACHE_DIR),
            "outside_label": OUTSIDE_LABEL,
            "label2id": label2id,
            "id2label": id2label,
            "category_labels": category_labels,
            "category_to_slug": category_to_slug,
            "max_length": MAX_LENGTH,
            "standalone_ratio": STANDALONE_RATIO,
            "same_class_ratio": SAME_CLASS_RATIO,
            "mixed_class_ratio": MIXED_CLASS_RATIO,
            "balanced_coverage_ratio": BALANCED_COVERAGE_RATIO,
            "reuse_limit": REUSE_LIMIT,
            "train_examples": TRAIN_EXAMPLES,
            "validation_examples": VALIDATION_EXAMPLES,
            "test_examples": TEST_EXAMPLES,
            "split_summaries": split_summaries,
        },
    )

    print("Wrote Salad tokenized cache with splits:")
    for split_name, split in tokenized_splits.items():
        print(f"  {split_name}: {len(split):,} examples")


if __name__ == "__main__":
    main()
