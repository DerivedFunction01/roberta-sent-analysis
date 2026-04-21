from __future__ import annotations

import math
import random
from collections import deque
from typing import Any

import numpy as np
from datasets import Dataset, concatenate_datasets
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from salad.labels import OUTSIDE_LABEL, slugify_label
from text_utils.mutations import TweetMutator


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


def _normalize_label(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    raise TypeError(f"Unsupported label type: {type(value)!r}")


def _normalize_source_id(value: Any, fallback: int) -> int:
    if value is None:
        return int(fallback)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value)
    return int(fallback)


class PoolSampler:
    def __init__(self, pools: dict[str, list[dict[str, Any]]], *, reuse_limit: int, seed: int) -> None:
        self.pools = pools
        self.rng = random.Random(seed)
        self.max_uses = max(1, reuse_limit + 1)
        self.remaining_uses = {label: [self.max_uses] * len(texts) for label, texts in pools.items()}
        self.remaining_counts = {label: len(texts) * self.max_uses for label, texts in pools.items()}
        self.label_queues = {}
        for label, texts in pools.items():
            order = list(range(len(texts)))
            self.rng.shuffle(order)
            self.label_queues[label] = deque(order)

    def _active_labels(self, labels: list[str] | None = None) -> list[str]:
        labels = labels or list(self.pools)
        return [label for label in labels if self.remaining_counts[label] > 0]

    def active_labels(self, labels: list[str] | None = None) -> list[str]:
        return self._active_labels(labels)

    def sample_label(self, labels: list[str] | None = None) -> str:
        labels = self._active_labels(labels)
        if not labels:
            raise RuntimeError("No reusable examples left in any label pool")
        weights = [self.remaining_counts[label] for label in labels]
        return self.rng.choices(labels, weights=weights, k=1)[0]

    def sample_balanced_label(self, labels: list[str] | None = None) -> str:
        labels = labels or list(self.pools)
        active = self._active_labels(labels)
        if not active:
            raise RuntimeError("No reusable examples left in any label pool")
        start = self.rng.randrange(len(labels))
        for offset in range(len(labels)):
            candidate = labels[(start + offset) % len(labels)]
            if candidate in active:
                return candidate
        return active[0]

    def sample_record(self, label: str) -> dict[str, Any]:
        queue = self.label_queues[label]
        while queue:
            index = queue.popleft()
            remaining = self.remaining_uses[label][index]
            if remaining <= 0:
                continue
            remaining -= 1
            self.remaining_uses[label][index] = remaining
            self.remaining_counts[label] -= 1
            if remaining > 0:
                queue.append(index)
            return self.pools[label][index]
        raise RuntimeError(f"No reusable examples left in label pool '{label}'")


def _build_pools(
    split: Dataset,
    *,
    text_column: str = "text",
    label_column: str = "label",
    mutator: TweetMutator | None = None,
    mutation_seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = {}
    rng = random.Random(mutation_seed)
    for fallback_source_id, row in enumerate(split):
        text = str(row.get(text_column, "")).strip()
        if not text:
            continue
        label = _normalize_label(row[label_column])
        source_id = _normalize_source_id(row.get("source_id"), fallback_source_id)
        pools.setdefault(label, []).append(
            {
                "source_id": source_id,
                "text": text,
                "label": label,
            }
        )
        if mutator is not None:
            for variant in mutator.augment(text, rng=rng, lang=None):
                if variant != text:
                    pools[label].append(
                        {
                            "source_id": source_id,
                            "text": variant,
                            "label": label,
                        }
                    )
    return pools


def _token_label_ids(label: str, *, label2id: dict[str, int], category_to_slug: dict[str, str]) -> tuple[int, int]:
    if label == OUTSIDE_LABEL:
        return label2id["O"], label2id["O"]
    slug = category_to_slug[label]
    return label2id[f"B-{slug}"], label2id[f"I-{slug}"]


def _encode_token_labels(
    seq_ids: list[int | None],
    *,
    label_a: str,
    label_b: str | None = None,
    label2id: dict[str, int],
    category_to_slug: dict[str, str],
) -> list[int]:
    labels: list[int] = []
    seen_a = False
    seen_b = False
    for seq_id in seq_ids:
        if seq_id is None:
            labels.append(-100)
            continue
        if seq_id == 0:
            first_id, other_id = _token_label_ids(label_a, label2id=label2id, category_to_slug=category_to_slug)
            labels.append(first_id if not seen_a else other_id)
            seen_a = True
            continue
        if seq_id == 1 and label_b is not None:
            first_id, other_id = _token_label_ids(label_b, label2id=label2id, category_to_slug=category_to_slug)
            labels.append(first_id if not seen_b else other_id)
            seen_b = True
            continue
        raise ValueError(f"Unexpected sequence id {seq_id!r} for salad labels")
    return labels


def build_standalone_examples(
    split: Dataset,
    *,
    num_examples: int,
    balanced_coverage_ratio: float,
    precleaned: bool,
    reuse_limit: int,
    seed: int,
    text_column: str = "text",
    label_column: str = "label",
    mutator: TweetMutator | None = None,
    mutation_seed: int = 42,
) -> tuple[Dataset, dict[str, Any]]:
    pools = _build_pools(
        split,
        text_column=text_column,
        label_column=label_column,
        mutator=mutator,
        mutation_seed=mutation_seed,
    )
    for label, records in pools.items():
        if not records:
            raise RuntimeError(f"Pool '{label}' is empty; cannot build standalone examples")

    sampler = PoolSampler(pools, reuse_limit=reuse_limit, seed=seed)
    records: list[dict[str, Any]] = []
    label_counts = {label: 0 for label in pools}
    balanced_count, free_count = _split_balanced_and_free(num_examples, balanced_coverage_ratio)
    active_labels = list(pools)

    def _draw_record(picker) -> tuple[str, dict[str, Any]] | None:
        nonlocal active_labels
        while active_labels:
            labels = sampler.active_labels(active_labels)
            if not labels:
                return None
            label = picker(labels)
            try:
                return label, sampler.sample_record(label)
            except RuntimeError:
                if label in active_labels:
                    active_labels = [candidate for candidate in active_labels if candidate != label]
        return None

    for _ in tqdm(range(balanced_count), desc="Building balanced standalone examples"):
        drawn = _draw_record(sampler.sample_balanced_label)
        if drawn is None:
            break
        label, record = drawn
        label_counts[label] += 1
        records.append(
            {
                "text_a": str(record["text"]),
                "text_b": "",
                "label_a": label,
                "label_b": "",
                "source_id_a": record["source_id"],
                "source_id_b": -1,
                "example_kind": "standalone",
                "pair_kind": "",
            }
        )

    for _ in tqdm(range(free_count), desc="Building free standalone examples"):
        drawn = _draw_record(sampler.sample_label)
        if drawn is None:
            break
        label, record = drawn
        label_counts[label] += 1
        records.append(
            {
                "text_a": str(record["text"]),
                "text_b": "",
                "label_a": label,
                "label_b": "",
                "source_id_a": record["source_id"],
                "source_id_b": -1,
                "example_kind": "standalone",
                "pair_kind": "",
            }
        )

    summary = {"pool_sizes": {label: len(texts) for label, texts in pools.items()}, "label_counts": label_counts, "num_examples": len(records)}
    return Dataset.from_list(records), summary


def build_paired_examples(
    split: Dataset,
    *,
    num_examples: int,
    pair_kind: str,
    balanced_coverage_ratio: float,
    precleaned: bool,
    reuse_limit: int,
    seed: int,
    text_column: str = "text",
    label_column: str = "label",
    mutator: TweetMutator | None = None,
    mutation_seed: int = 42,
) -> tuple[Dataset, dict[str, Any]]:
    if pair_kind not in {"same", "mixed"}:
        raise ValueError(f"Unsupported pair kind: {pair_kind}")

    pools = _build_pools(
        split,
        text_column=text_column,
        label_column=label_column,
        mutator=mutator,
        mutation_seed=mutation_seed,
    )
    for label, records in pools.items():
        if not records:
            raise RuntimeError(f"Pool '{label}' is empty; cannot build paired examples")

    sampler = PoolSampler(pools, reuse_limit=reuse_limit, seed=seed)
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    label_counts = {label: 0 for label in pools}

    balanced_count, free_count = _split_balanced_and_free(num_examples, balanced_coverage_ratio)
    balanced_labels = [sampler.sample_balanced_label() for _ in range(balanced_count)]
    free_labels = [sampler.sample_label() for _ in range(free_count)]

    for label_a in tqdm(balanced_labels + free_labels, desc=f"Building {pair_kind} pairs"):
        if pair_kind == "same":
            label_b = label_a
        else:
            active_labels = sampler.active_labels()
            other_labels = [label for label in active_labels if label != label_a]
            if not other_labels:
                raise RuntimeError("Need at least two active labels to build a mixed pair")
            label_b = sampler.sample_label(other_labels)

        record_a = sampler.sample_record(label_a)
        record_b = sampler.sample_record(label_b)
        text_a = str(record_a["text"])
        text_b = str(record_b["text"])
        if pair_kind == "mixed" and rng.random() < 0.5:
            text_a, text_b = text_b, text_a
            label_a, label_b = label_b, label_a
            record_a, record_b = record_b, record_a

        label_counts[label_a] += 1
        label_counts[label_b] += 1
        records.append(
            {
                "text_a": text_a,
                "text_b": text_b,
                "label_a": label_a,
                "label_b": label_b,
                "source_id_a": record_a["source_id"],
                "source_id_b": record_b["source_id"],
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
    label2id: dict[str, int],
    category_to_slug: dict[str, str],
) -> dict[str, list[list[int]]]:
    batch = tokenizer(examples["text_a"], truncation=True, max_length=max_length, padding=False)
    labels: list[list[int]] = []
    for index, label_a in enumerate(examples["label_a"]):
        seq_ids = batch.sequence_ids(index)
        labels.append(
            _encode_token_labels(
                seq_ids,
                label_a=label_a,
                label2id=label2id,
                category_to_slug=category_to_slug,
            )
        )
    batch["labels"] = labels
    return batch


def tokenize_paired_examples(
    examples: dict[str, list[str]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    label2id: dict[str, int],
    category_to_slug: dict[str, str],
) -> dict[str, list[list[int]]]:
    batch = tokenizer(examples["text_a"], examples["text_b"], truncation=True, max_length=max_length, padding=False)
    labels: list[list[int]] = []
    for index, (label_a, label_b) in enumerate(zip(examples["label_a"], examples["label_b"])):
        seq_ids = batch.sequence_ids(index)
        labels.append(
            _encode_token_labels(
                seq_ids,
                label_a=label_a,
                label_b=label_b,
                label2id=label2id,
                category_to_slug=category_to_slug,
            )
        )
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
    label2id: dict[str, int],
    category_labels: list[str],
    text_column: str = "text",
    label_column: str = "label",
    mutator: TweetMutator | None = None,
    mutation_seed: int = 42,
) -> tuple[Dataset, dict[str, Any]]:
    counts = _allocate_counts(
        num_examples,
        {
            "standalone": standalone_ratio,
            "same": same_class_ratio,
            "mixed": mixed_class_ratio,
        },
    )
    category_to_slug = {label: slugify_label(label) for label in category_labels}

    standalone, standalone_summary = build_standalone_examples(
        split,
        num_examples=counts["standalone"],
        balanced_coverage_ratio=balanced_coverage_ratio,
        precleaned=True,
        reuse_limit=reuse_limit,
        seed=seed,
        text_column=text_column,
        label_column=label_column,
        mutator=mutator,
        mutation_seed=mutation_seed,
    )
    same, same_summary = build_paired_examples(
        split,
        num_examples=counts["same"],
        pair_kind="same",
        balanced_coverage_ratio=balanced_coverage_ratio,
        precleaned=True,
        reuse_limit=reuse_limit,
        seed=seed,
        text_column=text_column,
        label_column=label_column,
        mutator=mutator,
        mutation_seed=mutation_seed,
    )
    mixed, mixed_summary = build_paired_examples(
        split,
        num_examples=counts["mixed"],
        pair_kind="mixed",
        balanced_coverage_ratio=balanced_coverage_ratio,
        precleaned=True,
        reuse_limit=reuse_limit,
        seed=seed,
        text_column=text_column,
        label_column=label_column,
        mutator=mutator,
        mutation_seed=mutation_seed,
    )

    standalone_tokenized = standalone.map(
        lambda batch: tokenize_standalone_examples(
            batch,
            tokenizer=tokenizer,
            max_length=max_length,
            label2id=label2id,
            category_to_slug=category_to_slug,
        ),
        batched=True,
    )
    same_tokenized = same.map(
        lambda batch: tokenize_paired_examples(
            batch,
            tokenizer=tokenizer,
            max_length=max_length,
            label2id=label2id,
            category_to_slug=category_to_slug,
        ),
        batched=True,
    )
    mixed_tokenized = mixed.map(
        lambda batch: tokenize_paired_examples(
            batch,
            tokenizer=tokenizer,
            max_length=max_length,
            label2id=label2id,
            category_to_slug=category_to_slug,
        ),
        batched=True,
    )

    tokenized = concatenate_datasets([standalone_tokenized, same_tokenized, mixed_tokenized])
    summary = {
        "counts": counts,
        "standalone_summary": standalone_summary,
        "same_summary": same_summary,
        "mixed_summary": mixed_summary,
        "num_examples": len(tokenized),
    }
    return tokenized, summary
