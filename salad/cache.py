from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm

from paths import path
from salad.defaults import DATASET_NAME, LABEL_COLUMN, MAX_SENTENCES, MIN_LATIN_RATIO, SUBSET, TEXT_COLUMN


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_label(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int,)):
        return str(value)
    raise TypeError(f"Unsupported label type: {type(value)!r}")


def sentence_count(text: str) -> int:
    sentences = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(text.strip()) if piece.strip()]
    return len(sentences)


def latin_ratio(text: str) -> float:
    letters = 0
    latin_letters = 0
    for char in text:
        if not unicodedata.category(char).startswith("L"):
            continue
        letters += 1
        if unicodedata.name(char, "").startswith("LATIN"):
            latin_letters += 1
    if letters == 0:
        return 0.0
    return latin_letters / letters


def is_majority_latin(text: str, *, min_ratio: float = MIN_LATIN_RATIO) -> bool:
    return latin_ratio(text) >= min_ratio


def _dataset_label_names(dataset: Dataset) -> list[str]:
    feature = dataset.features[LABEL_COLUMN]
    if hasattr(feature, "names"):
        return [str(name) for name in feature.names]

    labels: list[str] = []
    seen: set[str] = set()
    for row in dataset:
        label = normalize_label(row[LABEL_COLUMN])
        if label in seen:
            continue
        labels.append(label)
        seen.add(label)
    return labels


def _slugify_label(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    slug = slug.strip("_")
    return slug or "label"


def _load_split(dataset_name: str, subset: str, split_name: str) -> Dataset:
    loaded = load_dataset(dataset_name, subset, split=split_name)
    if not isinstance(loaded, Dataset):
        raise TypeError(f"Expected a Dataset for {dataset_name}/{subset}/{split_name}, got {type(loaded)!r}")
    return loaded


def _filter_split(
    split: Dataset,
    *,
    text_column: str,
    max_sentences: int,
    min_latin_ratio: float,
) -> tuple[Dataset, dict[str, int]]:
    kept_indices: list[int] = []
    stats = {
        "total": 0,
        "kept": 0,
        "dropped_empty": 0,
        "dropped_sentence_count": 0,
        "dropped_script": 0,
    }

    for idx, row in enumerate(tqdm(split, desc="Filtering Salad-Data")):
        stats["total"] += 1
        text = str(row.get(text_column, "")).strip()
        if not text:
            stats["dropped_empty"] += 1
            continue
        if sentence_count(text) > max_sentences:
            stats["dropped_sentence_count"] += 1
            continue
        if not is_majority_latin(text, min_ratio=min_latin_ratio):
            stats["dropped_script"] += 1
            continue
        kept_indices.append(idx)
        stats["kept"] += 1

    return split.select(kept_indices), stats


def load_clean_salad_cache(cache_dir: Path = path("salad", "salad_cache_dir")) -> dict[str, Dataset]:
    meta_file = path("salad", "salad_cache_meta_file")
    if not meta_file.exists():
        raise FileNotFoundError(f"Missing Salad-Data cache metadata: {meta_file}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    cache_files = meta.get("cache_files", {})
    if not isinstance(cache_files, dict):
        raise ValueError(f"Malformed cache metadata in {meta_file}")

    datasets_by_label: dict[str, Dataset] = {}
    for label, path_value in cache_files.items():
        path = Path(str(path_value))
        if not path.exists():
            raise FileNotFoundError(f"Missing Salad-Data cache file: {path}")
        datasets_by_label[str(label)] = load_dataset("parquet", data_files=str(path), split="train")
    return datasets_by_label


def build_clean_salad_cache(
    dataset_name: str = DATASET_NAME,
    subset: str = SUBSET,
    *,
    split_name: str = "train",
    text_column: str = TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
    max_sentences: int = MAX_SENTENCES,
    min_latin_ratio: float = MIN_LATIN_RATIO,
    cache_dir: Path = path("salad", "salad_cache_dir"),
) -> tuple[dict[str, Dataset], dict[str, Any]]:
    raw = _load_split(dataset_name, subset, split_name)
    filtered, filter_stats = _filter_split(
        raw,
        text_column=text_column,
        max_sentences=max_sentences,
        min_latin_ratio=min_latin_ratio,
    )

    label_names = _dataset_label_names(filtered)
    cache_dir.mkdir(parents=True, exist_ok=True)

    label_datasets: dict[str, Dataset] = {}
    cache_files: dict[str, str] = {}
    label_counts: dict[str, int] = {}
    for label_index, label_name in enumerate(label_names):
        label_slug = f"{label_index:02d}_{_slugify_label(label_name)}"
        records = [
            {
                "source_id": int(idx),
                "text": str(row.get(text_column, "")),
                "label": normalize_label(row[label_column]),
            }
            for idx, row in enumerate(filtered)
            if normalize_label(row[label_column]) == label_name
        ]
        dataset = Dataset.from_list(records)
        out_path = cache_dir / f"{label_slug}.parquet"
        dataset.to_parquet(str(out_path))
        label_datasets[label_name] = dataset
        cache_files[label_name] = str(out_path)
        label_counts[label_name] = len(records)

    meta = {
        "dataset_name": dataset_name,
        "subset": subset,
        "split_name": split_name,
        "text_column": text_column,
        "label_column": label_column,
        "max_sentences": max_sentences,
        "min_latin_ratio": min_latin_ratio,
        "filter_stats": filter_stats,
        "label_counts": label_counts,
        "label_names": label_names,
        "cache_files": cache_files,
        "total_rows": filter_stats["kept"],
    }
    save_json(path("salad", "salad_cache_meta_file"), meta)
    return label_datasets, meta


def ensure_clean_salad_cache(
    dataset_name: str = DATASET_NAME,
    subset: str = SUBSET,
    *,
    split_name: str = "train",
    text_column: str = TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
    max_sentences: int = MAX_SENTENCES,
    min_latin_ratio: float = MIN_LATIN_RATIO,
    cache_dir: Path = path("salad", "salad_cache_dir"),
) -> tuple[dict[str, Dataset], dict[str, Any]]:
    meta_file = path("salad", "salad_cache_meta_file")
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        cache_files = meta.get("cache_files", {})
        if isinstance(cache_files, dict) and all(Path(str(path)).exists() for path in cache_files.values()):
            return load_clean_salad_cache(cache_dir=cache_dir), meta
    return build_clean_salad_cache(
        dataset_name=dataset_name,
        subset=subset,
        split_name=split_name,
        text_column=text_column,
        label_column=label_column,
        max_sentences=max_sentences,
        min_latin_ratio=min_latin_ratio,
        cache_dir=cache_dir,
    )
