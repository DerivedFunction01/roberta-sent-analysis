from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import concatenate_datasets

from paths import path
from tweet.cache import ensure_clean_sentiment_cache
from tweet.data import (
    _allocate_counts,
    build_paired_examples,
    build_standalone_examples,
)
from tweet.defaults import (
    BALANCED_COVERAGE_RATIO,
    DATASET_NAME,
    MIXED_CLASS_RATIO,
    NORMALIZE_ALL_CAPS_DICTIONARY_WORDS,
    NORMALIZE_UNICODE_ESCAPES,
    REUSE_LIMIT,
    SAME_CLASS_RATIO,
    STANDALONE_RATIO,
    STRIP_QUOTE_ARTIFACTS,
    SUBSET,
)
from text_utils.mutations import TweetMutator


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate tweet sentiment dataset generation.")
    parser.add_argument(
        "--examples",
        nargs="+",
        type=int,
        default=[60_000],
        help="One or more target dataset sizes to simulate.",
    )
    return parser.parse_args()


def example_signature(example: dict[str, Any]) -> tuple[Any, ...]:
    if example["example_kind"] == "standalone":
        return (
            "standalone",
            example["text_a"],
            example["label_a"],
        )
    return (
        "paired",
        example["pair_kind"],
        example["text_a"],
        example["text_b"],
        example["label_a"],
        example["label_b"],
    )


def summarize_examples(dataset) -> dict[str, Any]:
    rows = [dict(row) for row in dataset]
    signatures = [example_signature(row) for row in rows]
    source_usage: Counter[int] = Counter()
    for row in rows:
        for key in ("source_id_a", "source_id_b"):
            source_id = int(row.get(key, -1))
            if source_id >= 0:
                source_usage[source_id] += 1

    unique_signatures = len(set(signatures))
    unique_sources = len(source_usage)
    total_source_uses = sum(source_usage.values())
    max_source_use = max(source_usage.values(), default=0)

    return {
        "num_examples": len(rows),
        "unique_examples": unique_signatures,
        "duplicate_examples": len(rows) - unique_signatures,
        "unique_source_ids_used": unique_sources,
        "total_source_uses": total_source_uses,
        "max_source_reuse": max_source_use,
        "mean_source_reuse": (total_source_uses / unique_sources) if unique_sources else 0.0,
        "source_reuse_histogram": dict(sorted(Counter(source_usage.values()).items())),
    }


def main() -> None:
    args = parse_args()
    mutator = TweetMutator()

    cached_label_splits, cache_meta = ensure_clean_sentiment_cache(
        DATASET_NAME,
        SUBSET,
        strip_quotes=STRIP_QUOTE_ARTIFACTS,
        normalize_escapes=NORMALIZE_UNICODE_ESCAPES,
        lowercase_dictionary_caps=NORMALIZE_ALL_CAPS_DICTIONARY_WORDS,
        cache_dir=path("tweet", "sentiment_cache_dir"),
    )
    cached_split = concatenate_datasets(
        [
            cached_label_splits[label].filter(lambda row: row["split"] == "train")
            for label in ("neg", "neu", "pos")
        ]
    )

    print("=" * 80)
    print("TWEET SENTIMENT SIMULATION")
    print("=" * 80)
    print(f"Cache counts: {cache_meta['counts']}")
    print(f"Balanced coverage ratio: {BALANCED_COVERAGE_RATIO}")
    print(f"Reuse limit: {REUSE_LIMIT}")

    reports: list[dict[str, Any]] = []
    for num_examples in args.examples:
        counts = _allocate_counts(
            num_examples,
            {
                "standalone": STANDALONE_RATIO,
                "same": SAME_CLASS_RATIO,
                "mixed": MIXED_CLASS_RATIO,
            },
        )
        standalone, standalone_summary = build_standalone_examples(
            cached_split,
            num_examples=counts["standalone"],
            balanced_coverage_ratio=BALANCED_COVERAGE_RATIO,
            precleaned=True,
            reuse_limit=REUSE_LIMIT,
            seed=42,
            lang_column="lang",
            mutator=mutator,
        )
        same_pairs, same_summary = build_paired_examples(
            cached_split,
            num_examples=counts["same"],
            pair_kind="same",
            balanced_coverage_ratio=BALANCED_COVERAGE_RATIO,
            precleaned=True,
            reuse_limit=REUSE_LIMIT,
            seed=42,
            lang_column="lang",
            mutator=mutator,
        )
        mixed_pairs, mixed_summary = build_paired_examples(
            cached_split,
            num_examples=counts["mixed"],
            pair_kind="mixed",
            balanced_coverage_ratio=BALANCED_COVERAGE_RATIO,
            precleaned=True,
            reuse_limit=REUSE_LIMIT,
            seed=43,
            lang_column="lang",
            mutator=mutator,
        )

        combined = concatenate_datasets([standalone, same_pairs, mixed_pairs]).shuffle(seed=42)
        summary = {
            "requested_examples": num_examples,
            "bucket_counts": counts,
            "standalone": summarize_examples(standalone),
            "same_pairs": summarize_examples(same_pairs),
            "mixed_pairs": summarize_examples(mixed_pairs),
            "combined": summarize_examples(combined),
            "bucket_summaries": {
                "standalone": standalone_summary,
                "same_pairs": same_summary,
                "mixed_pairs": mixed_summary,
            },
        }
        reports.append(summary)

        combined_summary = summary["combined"]
        reuse_ratio = (
            combined_summary["total_source_uses"] / combined_summary["unique_source_ids_used"]
            if combined_summary["unique_source_ids_used"]
            else 0.0
        )
        print(f"\nTarget rows: {num_examples:,}")
        print(f"  combined unique examples: {combined_summary['unique_examples']:,}")
        print(f"  combined duplicate examples: {combined_summary['duplicate_examples']:,}")
        print(f"  unique source ids used: {combined_summary['unique_source_ids_used']:,}")
        print(f"  total source uses: {combined_summary['total_source_uses']:,}")
        print(f"  mean source reuse: {combined_summary['mean_source_reuse']:.2f}")
        print(f"  max source reuse: {combined_summary['max_source_reuse']:,}")
        print(f"  reuse ratio: {reuse_ratio:.2f}")

    output_path = path("root", "results_dir") / "simulation_report.json"
    save_json(output_path, {"cache_meta": cache_meta, "reports": reports})
    print(f"\nSaved simulation report to {output_path}")


if __name__ == "__main__":
    main()
