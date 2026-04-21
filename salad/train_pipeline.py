from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import DatasetDict, concatenate_datasets
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from paths import path
from salad.cache import (
    ensure_clean_salad_cache,
    ensure_jailbreak_benign_cache,
    ensure_jailbreak_cache,
    ensure_openhermes_outside_cache,
)
from salad.data import build_tokenized_split
from salad.defaults import (
    BALANCED_COVERAGE_RATIO,
    CACHE_DIR,
    DATASET_NAME,
    CONTEXTUAL_MAX_SEGMENTS,
    CONTEXTUAL_MIN_SEGMENTS,
    CONTEXTUAL_PROBABILITY,
    JAILBREAK_CACHE_DIR,
    JAILBREAK_BENIGN_CACHE_DIR,
    JAILBREAK_DATASET_NAME,
    JAILBREAK_LABEL_COLUMN,
    JAILBREAK_MAX_SENTENCES,
    JAILBREAK_PROMPT_COLUMN,
    JAILBREAK_SPLIT,
    JAILBREAK_TARGET_LABEL,
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
    USE_SALAD_MUTATION,
)
from salad.labels import OUTSIDE_LABEL, load_label_map, slugify_label
from text_utils.mutations import TweetMutator


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
    jailbreak_benign_split, jailbreak_benign_meta = ensure_jailbreak_benign_cache(
        JAILBREAK_DATASET_NAME,
        split_name=JAILBREAK_SPLIT,
        prompt_column=JAILBREAK_PROMPT_COLUMN,
        label_column=JAILBREAK_LABEL_COLUMN,
        target_label="benign",
        max_sentences=JAILBREAK_MAX_SENTENCES,
        cache_dir=JAILBREAK_BENIGN_CACHE_DIR,
    )

    jailbreak_split, jailbreak_meta = ensure_jailbreak_cache(
        JAILBREAK_DATASET_NAME,
        split_name=JAILBREAK_SPLIT,
        prompt_column=JAILBREAK_PROMPT_COLUMN,
        label_column=JAILBREAK_LABEL_COLUMN,
        target_label=JAILBREAK_TARGET_LABEL,
        max_sentences=JAILBREAK_MAX_SENTENCES,
        cache_dir=JAILBREAK_CACHE_DIR,
    )
    jailbreak_label = "Jailbreak"

    label2id = load_label_map()
    id2label = {idx: label for label, idx in label2id.items()}
    category_labels = list(dict.fromkeys([*cache_meta["label_names"], jailbreak_label]))
    category_to_slug = {label: slugify_label(label) for label in category_labels}

    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    mutator = TweetMutator() if USE_SALAD_MUTATION else None

    combined_split = concatenate_datasets(
        [*unsafe_label_splits.values(), jailbreak_split, jailbreak_benign_split, outside_split]
    )

    if TOKENIZED_DATASET_DIR.exists():
        shutil.rmtree(TOKENIZED_DATASET_DIR)
    TOKENIZED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BUILDING SALAD TOKEN CLASSIFICATION CACHE")
    print("=" * 80)
    print(f"Unsafe dataset: {DATASET_NAME} / {SUBSET}")
    print(f"Label groups: {category_labels}")
    print(f"Jailbreak dataset: {JAILBREAK_DATASET_NAME} / {JAILBREAK_SPLIT}")
    print(f"Jailbreak cache: {JAILBREAK_CACHE_DIR}")
    print(f"Jailbreak benign cache: {JAILBREAK_BENIGN_CACHE_DIR}")
    print(f"Outside dataset: {NEUTRAL_DATASET_NAME} / {NEUTRAL_SPLIT}")
    print(f"Tokenizer: roberta-base")
    print(f"Output: {TOKENIZED_DATASET_DIR}")
    print(f"Label map: {path('salad', 'salad_label_map_file')}")
    print(f"Unsafe kept rows: {cache_meta['filter_stats']['kept']}")
    print(f"Jailbreak kept prompts: {jailbreak_meta['stats']['kept_prompts']}")
    print(f"Jailbreak generated chunks: {jailbreak_meta['stats']['generated_chunks']}")
    print(f"Jailbreak benign kept prompts: {jailbreak_benign_meta['stats']['kept_prompts']}")
    print(f"Jailbreak benign generated chunks: {jailbreak_benign_meta['stats']['generated_chunks']}")
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
            contextual_probability=CONTEXTUAL_PROBABILITY if split_name == "train" else 0.0,
            contextual_min_segments=CONTEXTUAL_MIN_SEGMENTS,
            contextual_max_segments=CONTEXTUAL_MAX_SEGMENTS,
            text_column="text",
            label_column="label",
            mutator=mutator if split_name == "train" else None,
            mutation_seed=42,
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
            "mutation": {"enabled": USE_SALAD_MUTATION},
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
