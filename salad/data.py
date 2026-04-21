from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
from datasets import Dataset, concatenate_datasets
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from salad.labels import OUTSIDE_LABEL, slugify_label


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


class PoolSampler:
    def __init__(self, pools: dict[str, list[dict[str, Any]]], *, reuse_limit: int, seed: int) -> None:
        self.pools = pools
        self.rng = random.Random(seed)
        self.max_uses = max(1, reuse_limit + 1)
        self.usage_counts = {label: [0] * len(texts) for label, texts in pools.items()}
        self.current_cycle = 0
        self.label_order = {
            label: sorted(
                range(len(texts)),
                key=lambda idx: (int(texts[idx].get("source_id", idx)), idx),
            )
            for label, texts in pools.items()
        }
        self.label_positions = {
            label: (self.rng.randrange(len(texts)) if texts else 0)
            for label, texts in pools.items()
        }
        self.label_weights = {label: float(len(texts)) for label, texts in pools.items()}

    def _eligible_indices(self, label: str) -> list[int]:
        return [
            idx
            for idx, count in enumerate(self.usage_counts[label])
            if count == self.current_cycle
        ]

    def _advance_cycle(self) -> bool:
        next_cycle = self.current_cycle + 1
        if any(any(count == next_cycle for count in counts) for counts in self.usage_counts.values()):
            self.current_cycle = next_cycle
            return True
        return False

    def _active_labels(self, labels: list[str] | None = None) -> list[str]:
        labels = labels or list(self.pools)
        active = [label for label in labels if self._eligible_indices(label)]
        while not active and self._advance_cycle():
            active = [label for label in labels if self._eligible_indices(label)]
        return active

    def active_labels(self, labels: list[str] | None = None) -> list[str]:
        return self._active_labels(labels)

    def sample_label(self, labels: list[str] | None = None) -> str:
        labels = self._active_labels(labels)
        if not labels:
            raise RuntimeError("No reusable examples left in any label pool")
        weights = [self.label_weights[label] for label in labels]
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
        eligible = self._eligible_indices(label)
        if not eligible:
            while self._advance_cycle():
                eligible = self._eligible_indices(label)
                if eligible:
                    break
            if not eligible:
                raise RuntimeError(f"No reusable examples left in label pool '{label}'")
        order = self.label_order[label]
        start = self.label_positions[label] % len(order)
        index = None
        for offset in range(len(order)):
            candidate = order[(start + offset) % len(order)]
            if candidate in eligible:
                index = candidate
                self.label_positions[label] = (start + offset + 1) % len(order)
                break
        if index is None:
            index = eligible[0]
            self.label_positions[label] = (start + 1) % len(order)
        self.usage_counts[label][index] += 1
        return self.pools[label][index]


def _build_pools(split: Dataset, *, text_column: str = "text", label_column: str = "label") -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = {}
    for fallback_source_id, row in enumerate(split):
        text = str(row.get(text_column, "")).strip()
        if not text:
            continue
        label = _normalize_label(row[label_column])
        pools.setdefault(label, []).append(
            {
                "source_id": int(row.get("source_id", fallback_source_id)),
                "text": text,
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
) -> tuple[Dataset, dict[str, Any]]:
    pools = _build_pools(split, text_column=text_column, label_column=label_column)
    for label, records in pools.items():
        if not records:
            raise RuntimeError(f"Pool '{label}' is empty; cannot build standalone examples")

    sampler = PoolSampler(pools, reuse_limit=reuse_limit, seed=seed)
    records: list[dict[str, Any]] = []
    label_counts = {label: 0 for label in pools}
    balanced_count, free_count = _split_balanced_and_free(num_examples, balanced_coverage_ratio)
    balanced_labels = [sampler.sample_balanced_label() for _ in range(balanced_count)]
    free_labels = [sampler.sample_label() for _ in range(free_count)]

    for label in tqdm(balanced_labels + free_labels, desc="Building standalone examples"):
        record = sampler.sample_record(label)
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
) -> tuple[Dataset, dict[str, Any]]:
    if pair_kind not in {"same", "mixed"}:
        raise ValueError(f"Unsupported pair kind: {pair_kind}")

    pools = _build_pools(split, text_column=text_column, label_column=label_column)
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
