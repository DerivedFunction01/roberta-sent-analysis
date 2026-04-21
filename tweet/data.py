from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase
from tweet.preprocess import clean_tweet_text
from text_utils.mutations import TweetMutator
from tweet.labels import ID2LABEL, LABEL2ID, LABEL_NAMES


@dataclass(frozen=True)
class PairedTweetExample:
    text_a: str
    text_b: str
    label_a: str
    label_b: str
    pair_kind: str


class PoolSampler:
    """Sample tweet indices from per-label pools with bounded reuse."""

    def __init__(self, pools: dict[str, list[str]], *, reuse_limit: int, seed: int) -> None:
        self.pools = pools
        self.rng = random.Random(seed)
        self.max_uses = max(1, reuse_limit + 1)
        self.usage_counts = {label: [0] * len(texts) for label, texts in pools.items()}
        self.label_weights = {
            label: (1.0 / max(len(texts), 1))
            for label, texts in pools.items()
        }

    def _eligible_indices(self, label: str) -> list[int]:
        return [
            idx
            for idx, count in enumerate(self.usage_counts[label])
            if count < self.max_uses
        ]

    def sample_label(self, labels: list[str] | None = None) -> str:
        weights = [self.label_weights[label] for label in labels]
        return self.rng.choices(labels, weights=weights, k=1)[0]

    def sample_text(self, label: str) -> str:
        eligible = self._eligible_indices(label)
        if not eligible:
            raise RuntimeError(f"No reusable examples left in label pool '{label}'")
        index = self.rng.choice(eligible)
        self.usage_counts[label][index] += 1
        return self.pools[label][index]


def _label_name(value: Any) -> str:
    if isinstance(value, str):
        value = value.strip().lower()
        if value in LABEL2ID:
            return value
        raise ValueError(f"Unsupported label string: {value}")
    if isinstance(value, (int, np.integer)):
        label_id = int(value)
        if label_id not in ID2LABEL:
            raise ValueError(f"Unsupported label id: {label_id}")
        return ID2LABEL[label_id]
    raise TypeError(f"Unsupported label type: {type(value)!r}")


def build_sentiment_pools(
    split: Dataset,
    *,
    text_column: str = "text",
    label_column: str = "label",
    lang_column: str | None = None,
    strip_quotes: bool = True,
    normalize_escapes: bool = True,
    lowercase_dictionary_caps: bool = False,
    mutator: TweetMutator | None = None,
    mutation_seed: int = 42,
) -> dict[str, list[str]]:
    """Collect cleaned tweets into label pools."""
    pools = {label: [] for label in LABEL_NAMES}
    rng = random.Random(mutation_seed)
    for row in split:
        text = clean_tweet_text(
            str(row.get(text_column, "")),
            strip_quotes=strip_quotes,
            normalize_escapes=normalize_escapes,
            lowercase_dictionary_caps=lowercase_dictionary_caps,
        )
        if not text:
            continue
        label = _label_name(row[label_column])
        lang = str(row.get(lang_column, "")) if lang_column else None
        pools[label].append(text)
        if mutator is not None:
            for variant in mutator.augment(text, rng=rng, lang=lang):
                if variant != text:
                    pools[label].append(variant)
    return pools


def build_paired_examples(
    split: Dataset,
    *,
    num_examples: int,
    same_class_ratio: float,
    reuse_limit: int,
    seed: int,
    text_column: str = "text",
    label_column: str = "label",
    lang_column: str | None = None,
    strip_quotes: bool = True,
    normalize_escapes: bool = True,
    lowercase_dictionary_caps: bool = False,
    mutator: TweetMutator | None = None,
    mutation_seed: int = 42,
) -> tuple[Dataset, dict[str, Any]]:
    """Create paired token-classification examples from a tweet sentiment split."""
    pools = build_sentiment_pools(
        split,
        text_column=text_column,
        label_column=label_column,
        lang_column=lang_column,
        strip_quotes=strip_quotes,
        normalize_escapes=normalize_escapes,
        lowercase_dictionary_caps=lowercase_dictionary_caps,
        mutator=mutator,
        mutation_seed=mutation_seed,
    )
    for label, texts in pools.items():
        if not texts:
            raise RuntimeError(f"Pool '{label}' is empty; cannot build paired examples")

    sampler = PoolSampler(pools, reuse_limit=reuse_limit, seed=seed)
    rng = random.Random(seed)
    records: list[dict[str, str]] = []
    pair_counts = {label: 0 for label in LABEL_NAMES}
    pair_kind_counts = {"same": 0, "mixed": 0}

    for _ in tqdm(range(num_examples), desc="Building paired examples"):
        if rng.random() < same_class_ratio:
            label_a = sampler.sample_label()
            label_b = label_a
            pair_kind = "same"
        else:
            label_a = sampler.sample_label()
            other_labels = [label for label in LABEL_NAMES if label != label_a]
            label_b = sampler.sample_label(other_labels)
            pair_kind = "mixed"

        text_a = sampler.sample_text(label_a)
        text_b = sampler.sample_text(label_b)
        if pair_kind == "mixed" and rng.random() < 0.5:
            text_a, text_b = text_b, text_a
            label_a, label_b = label_b, label_a

        pair_counts[label_a] += 1
        pair_counts[label_b] += 1
        pair_kind_counts[pair_kind] += 1
        records.append(
            {
                "text_a": text_a,
                "text_b": text_b,
                "label_a": label_a,
                "label_b": label_b,
                "pair_kind": pair_kind,
            }
        )

    summary = {
        "pool_sizes": {label: len(texts) for label, texts in pools.items()},
        "pair_counts": pair_counts,
        "pair_kind_counts": pair_kind_counts,
        "num_examples": len(records),
    }
    return Dataset.from_list(records), summary


def tokenize_paired_examples(
    examples: dict[str, list[str]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, list[list[int]]]:
    """Tokenize paired tweet examples and assign per-token sentiment labels."""
    batch = tokenizer(
        examples["text_a"],
        examples["text_b"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    labels: list[list[int]] = []
    for index, (label_a, label_b) in enumerate(zip(examples["label_a"], examples["label_b"])):
        seq_ids = batch.sequence_ids(index)
        row_labels: list[int] = []
        for seq_id in seq_ids:
            if seq_id is None:
                row_labels.append(-100)
            elif seq_id == 0:
                row_labels.append(LABEL2ID[label_a])
            else:
                row_labels.append(LABEL2ID[label_b])
        labels.append(row_labels)

    batch["labels"] = labels
    return batch


def build_tokenized_split(
    split: Dataset,
    *,
    num_examples: int,
    same_class_ratio: float,
    reuse_limit: int,
    seed: int,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    text_column: str = "text",
    label_column: str = "label",
    lang_column: str | None = None,
    strip_quotes: bool = True,
    normalize_escapes: bool = True,
    lowercase_dictionary_caps: bool = False,
    mutator: TweetMutator | None = None,
    mutation_seed: int = 42,
) -> tuple[Dataset, dict[str, Any]]:
    """Build and tokenize a paired token-classification dataset from one split."""
    paired, summary = build_paired_examples(
        split,
        num_examples=num_examples,
        same_class_ratio=same_class_ratio,
        reuse_limit=reuse_limit,
        seed=seed,
        text_column=text_column,
        label_column=label_column,
        lang_column=lang_column,
        strip_quotes=strip_quotes,
        normalize_escapes=normalize_escapes,
        lowercase_dictionary_caps=lowercase_dictionary_caps,
        mutator=mutator,
        mutation_seed=mutation_seed,
    )
    tokenized = paired.map(
        lambda batch: tokenize_paired_examples(batch, tokenizer=tokenizer, max_length=max_length),
        batched=True,
        remove_columns=paired.column_names,
        desc="Tokenizing sentiment pairs",
    )
    return tokenized, summary
