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
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from paths import path
from tweet.cache import ensure_clean_sentiment_cache
from tweet.data import build_tokenized_split
from tweet.defaults import (
    BALANCED_COVERAGE_RATIO,
    DATASET_NAME,
    MAX_LENGTH,
    NORMALIZE_ALL_CAPS_DICTIONARY_WORDS,
    NORMALIZE_UNICODE_ESCAPES,
    REUSE_LIMIT,
    STANDALONE_RATIO,
    SAME_CLASS_RATIO,
    MIXED_CLASS_RATIO,
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
    mutator = None
    if USE_TWEET_MUTATION:
        mutator = TweetMutator()

    cached_label_splits, cache_meta = ensure_clean_sentiment_cache(
        DATASET_NAME,
        SUBSET,
        strip_quotes=STRIP_QUOTE_ARTIFACTS,
        normalize_escapes=NORMALIZE_UNICODE_ESCAPES,
        lowercase_dictionary_caps=NORMALIZE_ALL_CAPS_DICTIONARY_WORDS,
        cache_dir=path("tweet", "sentiment_cache_dir"),
    )

    tokenized_dataset_dir = path("tweet", "tokenized_dataset_dir")
    pipeline_results_dir = path("tweet", "pipeline_results_dir")
    if tokenized_dataset_dir.exists():
        shutil.rmtree(tokenized_dataset_dir)
    tokenized_dataset_dir.mkdir(parents=True, exist_ok=True)
    pipeline_results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BUILDING TWEET SENTIMENT TOKENIZED CACHE")
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME} / {SUBSET}")
    print(f"Tokenizer: {model_checkpoint}")
    print(f"Output: {tokenized_dataset_dir}")
    print(f"Sentiment cache: {path('tweet', 'sentiment_cache_dir')}")
    print(f"Label map: {label2id}")
    print(f"Cached counts: {cache_meta['counts']}")

    split_specs = [
        ("train", TRAIN_EXAMPLES),
        ("validation", VALIDATION_EXAMPLES),
        ("test", TEST_EXAMPLES),
    ]

    tokenized_splits = {}
    split_summaries = {}
    for split_name, num_examples in tqdm(split_specs, desc="Building splits"):
        cached_split = concatenate_datasets(
            [
                cached_label_splits[label].filter(lambda row, split_name=split_name: row["split"] == split_name)
                for label in ("neg", "neu", "pos")
            ]
        )
        split, summary = build_tokenized_split(
            cached_split,
            num_examples=num_examples,
            standalone_ratio=STANDALONE_RATIO,
            same_class_ratio=SAME_CLASS_RATIO,
            mixed_class_ratio=MIXED_CLASS_RATIO,
            balanced_coverage_ratio=BALANCED_COVERAGE_RATIO,
            precleaned=True,
            reuse_limit=REUSE_LIMIT,
            seed=seed,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
            lang_column="lang",
            strip_quotes=STRIP_QUOTE_ARTIFACTS,
            normalize_escapes=NORMALIZE_UNICODE_ESCAPES,
            lowercase_dictionary_caps=NORMALIZE_ALL_CAPS_DICTIONARY_WORDS,
            mutator=mutator if split_name == "train" else None,
            mutation_seed=seed,
        )
        tokenized_splits[split_name] = split
        split_summaries[split_name] = summary
        save_json(tokenized_dataset_dir / f"{split_name}_summary.json", summary)

    dataset_dict = DatasetDict(tokenized_splits)
    dataset_dict.save_to_disk(str(tokenized_dataset_dir))
    save_json(
        path("tweet", "tokenized_dataset_meta"),
        {
            "dataset_name": DATASET_NAME,
            "subset": SUBSET,
            "model_checkpoint": model_checkpoint,
            "sentiment_cache_dir": str(path("tweet", "sentiment_cache_dir")),
            "max_length": MAX_LENGTH,
            "standalone_ratio": STANDALONE_RATIO,
            "same_class_ratio": SAME_CLASS_RATIO,
            "mixed_class_ratio": MIXED_CLASS_RATIO,
            "balanced_coverage_ratio": BALANCED_COVERAGE_RATIO,
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
