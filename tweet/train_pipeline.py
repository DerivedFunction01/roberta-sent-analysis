from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from paths import LABEL_MAP_FILE, PIPELINE_RESULTS_DIR, TOKENIZED_DATASET_DIR
from tweet.data import build_tokenized_split
from tweet.defaults import (
    DATASET_NAME,
    MAX_LENGTH,
    NORMALIZE_ALL_CAPS_DICTIONARY_WORDS,
    NORMALIZE_UNICODE_ESCAPES,
    REUSE_LIMIT,
    SAME_CLASS_RATIO,
    STRIP_QUOTE_ARTIFACTS,
    SUBSET,
    TEST_EXAMPLES,
    TRAIN_EXAMPLES,
    USE_TWEET_MUTATION,
    VALIDATION_EXAMPLES,
)
from tweet.labels import LABEL2ID
from text_utils.mutations import TweetMutator


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    label2id = LABEL2ID
    model_checkpoint = "roberta-base"
    seed = 42
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    dataset = load_dataset(DATASET_NAME, SUBSET)
    mutator = None
    if USE_TWEET_MUTATION:
        mutator = TweetMutator()

    if TOKENIZED_DATASET_DIR.exists():
        shutil.rmtree(TOKENIZED_DATASET_DIR)
    TOKENIZED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BUILDING TWEET SENTIMENT TOKENIZED CACHE")
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME} / {SUBSET}")
    print(f"Tokenizer: {model_checkpoint}")
    print(f"Output: {TOKENIZED_DATASET_DIR}")
    print(f"Label map: {label2id}")

    split_specs = [
        ("train", TRAIN_EXAMPLES),
        ("validation", VALIDATION_EXAMPLES),
        ("test", TEST_EXAMPLES),
    ]

    tokenized_splits = {}
    split_summaries = {}
    for split_name, num_examples in tqdm(split_specs, desc="Building splits"):
        split, summary = build_tokenized_split(
            dataset[split_name],
            num_examples=num_examples,
            same_class_ratio=SAME_CLASS_RATIO,
            reuse_limit=REUSE_LIMIT,
            seed=seed,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
            strip_quotes=STRIP_QUOTE_ARTIFACTS,
            normalize_escapes=NORMALIZE_UNICODE_ESCAPES,
            lowercase_dictionary_caps=NORMALIZE_ALL_CAPS_DICTIONARY_WORDS,
            mutator=mutator if split_name == "train" else None,
            mutation_seed=seed,
        )
        tokenized_splits[split_name] = split
        split_summaries[split_name] = summary
        save_json(TOKENIZED_DATASET_DIR / f"{split_name}_summary.json", summary)

    dataset_dict = DatasetDict(tokenized_splits)
    dataset_dict.save_to_disk(str(TOKENIZED_DATASET_DIR))
    save_json(LABEL_MAP_FILE, label2id)
    save_json(
        TOKENIZED_DATASET_DIR / "cache_meta.json",
        {
            "dataset_name": DATASET_NAME,
            "subset": SUBSET,
            "model_checkpoint": model_checkpoint,
            "max_length": MAX_LENGTH,
            "same_class_ratio": SAME_CLASS_RATIO,
            "reuse_limit": REUSE_LIMIT,
            "train_examples": TRAIN_EXAMPLES,
            "validation_examples": VALIDATION_EXAMPLES,
            "test_examples": TEST_EXAMPLES,
            "seed": seed,
            "label2id": label2id,
            "split_summaries": split_summaries,
            "tweet_mutation": {"enabled": USE_TWEET_MUTATION},
        },
    )

    print("Wrote tokenized cache with splits:")
    for split_name, split in tokenized_splits.items():
        print(f"  {split_name}: {len(split):,} examples")


if __name__ == "__main__":
    main()
