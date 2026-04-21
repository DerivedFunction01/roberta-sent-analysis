from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm

from paths import path
from tweet.labels import SENTIMENT_ID2LABEL, SENTIMENT_LABEL2ID, SENTIMENT_LABELS
from tweet.preprocess import clean_tweet_text


_CACHE_FILES = {
    "neg": path("tweet", "sentiment_cache_neg_file").name,
    "neu": path("tweet", "sentiment_cache_neu_file").name,
    "pos": path("tweet", "sentiment_cache_pos_file").name,
}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _normalize_label(value: Any) -> str:
    if isinstance(value, str):
        value = value.strip().lower()
        if value in SENTIMENT_LABEL2ID:
            return value
        raise ValueError(f"Unsupported label string: {value}")
    if isinstance(value, int):
        if value not in SENTIMENT_ID2LABEL:
            raise ValueError(f"Unsupported label id: {value}")
        return SENTIMENT_ID2LABEL[value]
    raise TypeError(f"Unsupported label type: {type(value)!r}")


def load_clean_sentiment_cache(cache_dir: Path = path("tweet", "sentiment_cache_dir")) -> dict[str, Dataset]:
    cache_files = {label: cache_dir / file_name for label, file_name in _CACHE_FILES.items()}
    missing = [path for path in cache_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing sentiment cache files: {missing}")
    return {label: load_dataset("parquet", data_files=str(path), split="train") for label, path in cache_files.items()}


def build_clean_sentiment_cache(
    dataset_name: str,
    subset: str,
    *,
    text_column: str = "text",
    label_column: str = "label",
    lang_column: str | None = None,
    strip_quotes: bool = True,
    normalize_escapes: bool = True,
    lowercase_dictionary_caps: bool = False,
    cache_dir: Path = path("tweet", "sentiment_cache_dir"),
) -> tuple[dict[str, Dataset], dict[str, Any]]:
    raw = load_dataset(dataset_name, subset)
    if isinstance(raw, DatasetDict):
        splits = dict(raw.items())
    else:
        splits = {"train": raw}

    cache_dir.mkdir(parents=True, exist_ok=True)
    pools: dict[str, list[dict[str, Any]]] = {label: [] for label in SENTIMENT_LABELS}
    counts = {label: 0 for label in SENTIMENT_LABELS}
    split_counts: dict[str, dict[str, int]] = {}

    for split_name, split in splits.items():
        split_counts[split_name] = {label: 0 for label in SENTIMENT_LABELS}
        for source_id, row in enumerate(tqdm(split, desc=f"Cleaning {split_name} cache")):
            label = _normalize_label(row[label_column])
            text = clean_tweet_text(
                str(row.get(text_column, "")),
                strip_quotes=strip_quotes,
                normalize_escapes=normalize_escapes,
                lowercase_dictionary_caps=lowercase_dictionary_caps,
            )
            if not text:
                continue
            record = {
                "split": split_name,
                "source_id": source_id,
                "text": text,
                "label": label,
                "lang": str(row.get(lang_column, "")) if lang_column else "",
            }
            pools[label].append(record)
            counts[label] += 1
            split_counts[split_name][label] += 1

    label_datasets: dict[str, Dataset] = {}
    file_map = {label: cache_dir / file_name for label, file_name in _CACHE_FILES.items()}
    for label, records in pools.items():
        dataset = Dataset.from_list(records)
        dataset.to_parquet(str(file_map[label]))
        label_datasets[label] = dataset

    meta = {
        "dataset_name": dataset_name,
        "subset": subset,
        "text_column": text_column,
        "label_column": label_column,
        "lang_column": lang_column,
        "strip_quotes": strip_quotes,
        "normalize_escapes": normalize_escapes,
        "lowercase_dictionary_caps": lowercase_dictionary_caps,
        "counts": counts,
        "split_counts": split_counts,
        "total_rows": sum(counts.values()),
        "cache_files": {label: str(path) for label, path in file_map.items()},
    }
    save_json(path("tweet", "sentiment_cache_meta_file"), meta)
    return label_datasets, meta


def ensure_clean_sentiment_cache(
    dataset_name: str,
    subset: str,
    *,
    text_column: str = "text",
    label_column: str = "label",
    lang_column: str | None = None,
    strip_quotes: bool = True,
    normalize_escapes: bool = True,
    lowercase_dictionary_caps: bool = False,
    cache_dir: Path = path("tweet", "sentiment_cache_dir"),
) -> tuple[dict[str, Dataset], dict[str, Any]]:
    cache_files = {label: cache_dir / file_name for label, file_name in _CACHE_FILES.items()}
    meta_file = path("tweet", "sentiment_cache_meta_file")
    if meta_file.exists() and all(path.exists() for path in cache_files.values()):
        return load_clean_sentiment_cache(cache_dir=cache_dir), json.loads(meta_file.read_text(encoding="utf-8"))
    return build_clean_sentiment_cache(
        dataset_name,
        subset,
        text_column=text_column,
        label_column=label_column,
        lang_column=lang_column,
        strip_quotes=strip_quotes,
        normalize_escapes=normalize_escapes,
        lowercase_dictionary_caps=lowercase_dictionary_caps,
        cache_dir=cache_dir,
    )
