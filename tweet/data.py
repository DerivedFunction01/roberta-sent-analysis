from __future__ import annotations

import random
import math
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase
from tweet.preprocess import clean_tweet_text
from text_utils.mutations import TweetMutator
from tweet.labels import (
    LABEL2ID,
    SENTIMENT_ID2LABEL,
    SENTIMENT_LABEL2ID,
    SENTIMENT_LABELS,
)


class PoolSampler:
    """Sample tweet indices from per-label pools with bounded reuse."""

    def __init__(self, pools: dict[str, list[str]], *, reuse_limit: int, seed: int) -> None:
        self.pools = pools
        self.rng = random.Random(seed)
        self.max_uses = max(1, reuse_limit + 1)
        self.usage_counts = {label: [0] * len(texts) for label, texts in pools.items()}
        self.label_weights = {
            label: float(len(texts))
            for label, texts in pools.items()
        }

    def _eligible_indices(self, label: str) -> list[int]:
        return [
            idx
            for idx, count in enumerate(self.usage_counts[label])
            if count < self.max_uses
        ]

    def sample_label(self, labels: list[str] | None = None) -> str:
        labels = labels or list(SENTIMENT_LABELS)
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
        if value in SENTIMENT_LABEL2ID:
            return value
        raise ValueError(f"Unsupported label string: {value}")
    if isinstance(value, (int, np.integer)):
        label_id = int(value)
        if label_id not in SENTIMENT_ID2LABEL:
            raise ValueError(f"Unsupported label id: {label_id}")
        return SENTIMENT_ID2LABEL[label_id]
    raise TypeError(f"Unsupported label type: {type(value)!r}")


def _allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    ratio_sum = sum(ratios.values())
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Ratios must sum to 1.0; got {ratio_sum}")

    raw_counts = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(math.floor(value)) for name, value in raw_counts.items()}
    remainder = total - sum(counts.values())
    if remainder > 0:
        fractional_parts = sorted(
            ((raw_counts[name] - counts[name], name) for name in ratios),
            reverse=True,
        )
        for _, name in fractional_parts[:remainder]:
            counts[name] += 1
    return counts


def _split_balanced_and_free(total: int, balanced_coverage_ratio: float) -> tuple[int, int]:
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if not 0.0 <= balanced_coverage_ratio <= 1.0:
        raise ValueError(f"balanced_coverage_ratio must be between 0 and 1, got {balanced_coverage_ratio}")
    balanced = int(math.ceil(total * balanced_coverage_ratio))
    balanced = min(total, balanced)
    return balanced, total - balanced


def _balanced_label_sequence(total: int) -> list[str]:
    if total <= 0:
        return []
    labels = list(SENTIMENT_LABELS)
    return [labels[index % len(labels)] for index in range(total)]


def _next_label(label: str) -> str:
    labels = list(SENTIMENT_LABELS)
    index = labels.index(label)
    return labels[(index + 1) % len(labels)]


def _token_label_ids_for_sentiment(sentiment: str) -> tuple[int, int]:
    if sentiment == "neu":
        return LABEL2ID["O"], LABEL2ID["O"]
    if sentiment == "pos":
        return LABEL2ID["B-POS"], LABEL2ID["I-POS"]
    if sentiment == "neg":
        return LABEL2ID["B-NEG"], LABEL2ID["I-NEG"]
    raise ValueError(f"Unsupported sentiment label: {sentiment}")


def _encode_token_labels(
    seq_ids: list[int | None],
    *,
    label_a: str,
    label_b: str | None = None,
) -> list[int]:
    labels: list[int] = []
    seen_a = False
    seen_b = False
    for seq_id in seq_ids:
        if seq_id is None:
            labels.append(-100)
            continue
        if seq_id == 0:
            first_id, other_id = _token_label_ids_for_sentiment(label_a)
            labels.append(first_id if not seen_a else other_id)
            seen_a = True
            continue
        if seq_id == 1 and label_b is not None:
            first_id, other_id = _token_label_ids_for_sentiment(label_b)
            labels.append(first_id if not seen_b else other_id)
            seen_b = True
            continue
        raise ValueError(f"Unexpected sequence id {seq_id!r} for sentiment labels")
    return labels


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
    pools = {label: [] for label in SENTIMENT_LABELS}
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


def build_standalone_examples(
    split: Dataset,
    *,
    num_examples: int,
    balanced_coverage_ratio: float,
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
    """Create standalone token-classification examples from a tweet sentiment split."""
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
            raise RuntimeError(f"Pool '{label}' is empty; cannot build standalone examples")

    sampler = PoolSampler(pools, reuse_limit=reuse_limit, seed=seed)
    records: list[dict[str, str]] = []
    label_counts = {label: 0 for label in SENTIMENT_LABELS}

    balanced_count, free_count = _split_balanced_and_free(num_examples, balanced_coverage_ratio)
    balanced_labels = _balanced_label_sequence(balanced_count)
    free_labels = [sampler.sample_label() for _ in range(free_count)]

    for label_a in tqdm(balanced_labels + free_labels, desc="Building standalone examples"):
        text_a = sampler.sample_text(label_a)
        label_counts[label_a] += 1
        records.append(
            {
                "text_a": text_a,
                "text_b": "",
                "label_a": label_a,
                "label_b": "",
                "example_kind": "standalone",
                "pair_kind": "",
            }
        )

    summary = {
        "pool_sizes": {label: len(texts) for label, texts in pools.items()},
        "label_counts": label_counts,
        "num_examples": len(records),
    }
    return Dataset.from_list(records), summary


def build_paired_examples(
    split: Dataset,
    *,
    num_examples: int,
    pair_kind: str,
    balanced_coverage_ratio: float,
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
    if pair_kind not in {"same", "mixed"}:
        raise ValueError(f"Unsupported pair kind: {pair_kind}")

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
    label_counts = {label: 0 for label in SENTIMENT_LABELS}

    balanced_count, free_count = _split_balanced_and_free(num_examples, balanced_coverage_ratio)
    balanced_labels = _balanced_label_sequence(balanced_count)
    free_labels = [sampler.sample_label() for _ in range(free_count)]

    for label_a in tqdm(balanced_labels + free_labels, desc=f"Building {pair_kind} pairs"):
        if pair_kind == "same":
            label_b = label_a
        else:
            label_b = _next_label(label_a)
            if label_b == label_a:
                other_labels = [label for label in SENTIMENT_LABELS if label != label_a]
                label_b = sampler.sample_label(other_labels)

        text_a = sampler.sample_text(label_a)
        text_b = sampler.sample_text(label_b)
        if pair_kind == "mixed" and rng.random() < 0.5:
            text_a, text_b = text_b, text_a
            label_a, label_b = label_b, label_a

        label_counts[label_a] += 1
        label_counts[label_b] += 1
        records.append(
            {
                "text_a": text_a,
                "text_b": text_b,
                "label_a": label_a,
                "label_b": label_b,
                "example_kind": "paired",
                "pair_kind": pair_kind,
            }
        )

    summary = {
        "pool_sizes": {label: len(texts) for label, texts in pools.items()},
        "label_counts": label_counts,
        "pair_kind": pair_kind,
        "balanced_coverage_ratio": balanced_coverage_ratio,
        "num_examples": len(records),
    }
    return Dataset.from_list(records), summary


def tokenize_standalone_examples(
    examples: dict[str, list[str]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, list[list[int]]]:
    """Tokenize standalone tweet examples and assign per-token sentiment labels."""
    batch = tokenizer(examples["text_a"], truncation=True, max_length=max_length, padding=False)
    labels: list[list[int]] = []
    for index, label_a in enumerate(examples["label_a"]):
        seq_ids = batch.sequence_ids(index)
        labels.append(_encode_token_labels(seq_ids, label_a=label_a))
    batch["labels"] = labels
    return batch


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
        labels.append(_encode_token_labels(seq_ids, label_a=label_a, label_b=label_b))

    batch["labels"] = labels
    return batch


def build_tokenized_split(
    split: Dataset,
    *,
    num_examples: int,
    standalone_ratio: float,
    same_class_ratio: float,
    mixed_class_ratio: float,
    balanced_coverage_ratio: float,
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
    """Build and tokenize a mixed token-classification dataset from one split."""
    counts = _allocate_counts(
        num_examples,
        {
            "standalone": standalone_ratio,
            "same": same_class_ratio,
            "mixed": mixed_class_ratio,
        },
    )

    standalone, standalone_summary = build_standalone_examples(
        split,
        num_examples=counts["standalone"],
        balanced_coverage_ratio=balanced_coverage_ratio,
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
    same_pairs, same_summary = build_paired_examples(
        split,
        num_examples=counts["same"],
        pair_kind="same",
        balanced_coverage_ratio=balanced_coverage_ratio,
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
    mixed_pairs, mixed_summary = build_paired_examples(
        split,
        num_examples=counts["mixed"],
        pair_kind="mixed",
        balanced_coverage_ratio=balanced_coverage_ratio,
        reuse_limit=reuse_limit,
        seed=seed + 1,
        text_column=text_column,
        label_column=label_column,
        lang_column=lang_column,
        strip_quotes=strip_quotes,
        normalize_escapes=normalize_escapes,
        lowercase_dictionary_caps=lowercase_dictionary_caps,
        mutator=mutator,
        mutation_seed=mutation_seed,
    )

    tokenized_parts = []
    if len(standalone):
        tokenized_parts.append(
            standalone.map(
                lambda batch: tokenize_standalone_examples(
                    batch, tokenizer=tokenizer, max_length=max_length
                ),
                batched=True,
                remove_columns=standalone.column_names,
                desc="Tokenizing standalone examples",
            )
        )
    if len(same_pairs):
        tokenized_parts.append(
            same_pairs.map(
                lambda batch: tokenize_paired_examples(batch, tokenizer=tokenizer, max_length=max_length),
                batched=True,
                remove_columns=same_pairs.column_names,
                desc="Tokenizing same pairs",
            )
        )
    if len(mixed_pairs):
        tokenized_parts.append(
            mixed_pairs.map(
                lambda batch: tokenize_paired_examples(batch, tokenizer=tokenizer, max_length=max_length),
                batched=True,
                remove_columns=mixed_pairs.column_names,
                desc="Tokenizing mixed pairs",
            )
        )

    if not tokenized_parts:
        raise RuntimeError("No tokenized parts were built")

    tokenized = tokenized_parts[0] if len(tokenized_parts) == 1 else concatenate_datasets(tokenized_parts)
    tokenized = tokenized.shuffle(seed=seed)
    summary = {
        "counts": counts,
        "balanced_coverage_ratio": balanced_coverage_ratio,
        "standalone": standalone_summary,
        "same_pairs": same_summary,
        "mixed_pairs": mixed_summary,
        "num_examples": len(tokenized),
    }
    return tokenized, summary
