from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm

from paths import path
from salad.defaults import (
    CACHE_DIR,
    DATASET_NAME,
    JAILBREAK_CACHE_DIR,
    JAILBREAK_BENIGN_CACHE_DIR,
    JAILBREAK_BENIGN_LABEL,
    JAILBREAK_DATASET_NAME,
    JAILBREAK_LABEL_COLUMN,
    JAILBREAK_MAX_SENTENCES,
    JAILBREAK_PROMPT_COLUMN,
    JAILBREAK_SPLIT,
    JAILBREAK_TARGET_LABEL,
    LABEL_COLUMN,
    MAX_SENTENCES,
    MIN_LATIN_RATIO,
    NEUTRAL_CACHE_DIR,
    NEUTRAL_DATASET_NAME,
    NEUTRAL_MAX_SENTENCES,
    NEUTRAL_MIN_LATIN_RATIO,
    NEUTRAL_SAMPLE_FRACTION,
    NEUTRAL_SPLIT,
    NEUTRAL_TEXT_COLUMN,
    SUBSET,
    TEXT_COLUMN,
)


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
NEWLINE_SPLIT_RE = re.compile(r"(?:\r?\n){1,2}")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_local_parquet_dataset(parquet_path: Path) -> Dataset:
    table = pq.read_table(str(parquet_path))
    rows = table.to_pylist()
    for idx, row in enumerate(rows):
        if row.get("source_id") is None:
            row["source_id"] = idx
        row.pop("__index_level_0__", None)
    return Dataset.from_list(rows)


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


def _load_split(
    dataset_name: str,
    subset: str | None,
    split_name: str,
    *,
    sample_fraction: float | None = None,
) -> Dataset:
    split_spec = split_name
    if sample_fraction is not None:
        if not 0 < sample_fraction <= 1:
            raise ValueError("sample_fraction must be in the interval (0, 1]")
        if sample_fraction < 1:
            split_spec = f"{split_name}[:{sample_fraction * 100:g}%]"

    if subset is None:
        loaded = load_dataset(dataset_name, split=split_spec)
    else:
        loaded = load_dataset(dataset_name, subset, split=split_spec)
    if not isinstance(loaded, Dataset):
        label = f"{dataset_name}/{split_spec}" if subset is None else f"{dataset_name}/{subset}/{split_spec}"
        raise TypeError(f"Expected a Dataset for {label}, got {type(loaded)!r}")
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


def _first_human_turn(conversations: Any) -> str:
    if isinstance(conversations, str):
        return conversations.strip()
    if not isinstance(conversations, list):
        return ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("from", "")).strip().lower() != "human":
            continue
        return str(turn.get("value", "")).strip()
    return ""


def _split_jailbreak_segments(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    pieces = [piece.strip() for piece in NEWLINE_SPLIT_RE.split(normalized) if piece.strip()]
    segments: list[str] = []
    for piece in pieces:
        sentence_parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(piece) if part.strip()]
        if sentence_parts:
            segments.extend(sentence_parts)
        else:
            segments.append(piece)
    return segments


def _sliding_windows(items: list[str], window_size: int, stride: int = 1) -> list[list[str]]:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if not items:
        return []
    if len(items) <= window_size:
        return [items]
    windows: list[list[str]] = []
    for start in range(0, len(items) - window_size + 1, stride):
        windows.append(items[start : start + window_size])
    tail_start = len(items) - window_size
    if windows and windows[-1] != items[tail_start:]:
        windows.append(items[tail_start:])
    return windows


def _chunk_jailbreak_prompt(prompt: str, *, max_sentences: int = JAILBREAK_MAX_SENTENCES) -> list[str]:
    sentences = _split_jailbreak_segments(prompt)
    if not sentences:
        return []
    if len(sentences) <= max_sentences:
        return [" ".join(sentences)]
    return [" ".join(window) for window in _sliding_windows(sentences, max_sentences, stride=1)]


def _filter_openhermes_split(
    split: Dataset,
    *,
    conversations_column: str,
    max_sentences: int,
    min_latin_ratio: float,
) -> tuple[Dataset, dict[str, int]]:
    kept_indices: list[int] = []
    stats = {
        "total": 0,
        "kept": 0,
        "dropped_empty": 0,
        "dropped_no_human_turn": 0,
        "dropped_sentence_count": 0,
        "dropped_script": 0,
    }

    for idx, row in enumerate(tqdm(split, desc="Filtering OpenHermes")):
        stats["total"] += 1
        human_text = _first_human_turn(row.get(conversations_column, ""))
        if not human_text:
            stats["dropped_no_human_turn"] += 1
            continue
        if not human_text.strip():
            stats["dropped_empty"] += 1
            continue
        if sentence_count(human_text) > max_sentences:
            stats["dropped_sentence_count"] += 1
            continue
        if not is_majority_latin(human_text, min_ratio=min_latin_ratio):
            stats["dropped_script"] += 1
            continue
        kept_indices.append(idx)
        stats["kept"] += 1

    return split.select(kept_indices), stats


def load_clean_salad_cache(cache_dir: Path = CACHE_DIR) -> dict[str, Dataset]:
    meta_file = path("salad", "salad_cache_meta_file")
    if not meta_file.exists():
        raise FileNotFoundError(f"Missing Salad-Data cache metadata: {meta_file}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    cache_files = meta.get("cache_files", {})
    if not isinstance(cache_files, dict):
        raise ValueError(f"Malformed cache metadata in {meta_file}")

    datasets_by_label: dict[str, Dataset] = {}
    for label, path_value in cache_files.items():
        cache_path = Path(str(path_value))
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing Salad-Data cache file: {cache_path}")
        datasets_by_label[str(label)] = load_local_parquet_dataset(cache_path)
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
    cache_dir: Path = CACHE_DIR,
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
    cache_dir: Path = CACHE_DIR,
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


def load_openhermes_outside_cache(cache_dir: Path = NEUTRAL_CACHE_DIR) -> Dataset:
    meta_file = path("salad", "salad_outside_cache_meta_file")
    if not meta_file.exists():
        raise FileNotFoundError(f"Missing OpenHermes outside cache metadata: {meta_file}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    cache_file = meta.get("cache_file")
    if not isinstance(cache_file, str):
        raise ValueError(f"Malformed outside cache metadata in {meta_file}")
    out_path = Path(cache_file)
    if not out_path.exists():
        raise FileNotFoundError(f"Missing OpenHermes outside cache file: {out_path}")
    return load_local_parquet_dataset(out_path)


def load_jailbreak_cache(cache_dir: Path = JAILBREAK_CACHE_DIR) -> Dataset:
    meta_file = path("salad", "salad_jailbreak_cache_meta_file")
    if not meta_file.exists():
        raise FileNotFoundError(f"Missing jailbreak cache metadata: {meta_file}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    cache_file = meta.get("cache_file")
    if not isinstance(cache_file, str):
        raise ValueError(f"Malformed jailbreak cache metadata in {meta_file}")
    out_path = Path(cache_file)
    if not out_path.exists():
        raise FileNotFoundError(f"Missing jailbreak cache file: {out_path}")
    return load_local_parquet_dataset(out_path)


def _build_jackhhao_classification_cache(
    dataset_name: str = JAILBREAK_DATASET_NAME,
    *,
    split_name: str = JAILBREAK_SPLIT,
    prompt_column: str = JAILBREAK_PROMPT_COLUMN,
    label_column: str = JAILBREAK_LABEL_COLUMN,
    target_label: str = JAILBREAK_TARGET_LABEL,
    output_label: str = "Jailbreak",
    max_sentences: int = JAILBREAK_MAX_SENTENCES,
    cache_dir: Path = JAILBREAK_CACHE_DIR,
) -> tuple[Dataset, dict[str, Any]]:
    raw = _load_split(dataset_name, None, split_name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0,
        "kept_prompts": 0,
        "dropped_empty": 0,
        "dropped_label": 0,
        "dropped_no_chunks": 0,
        "generated_chunks": 0,
    }
    records: list[dict[str, Any]] = []
    for fallback_source_id, row in enumerate(tqdm(raw, desc="Filtering jailbreak prompts")):
        stats["total"] += 1
        label = str(row.get(label_column, "")).strip().lower()
        if label != target_label:
            stats["dropped_label"] += 1
            continue
        prompt = str(row.get(prompt_column, "")).strip()
        if not prompt:
            stats["dropped_empty"] += 1
            continue
        chunks = _chunk_jailbreak_prompt(prompt, max_sentences=max_sentences)
        if not chunks:
            stats["dropped_no_chunks"] += 1
            continue
        stats["kept_prompts"] += 1
        for chunk_index, chunk in enumerate(chunks):
            records.append(
                {
                    "source_id": fallback_source_id * 10_000 + chunk_index,
                    "text": chunk,
                    "label": output_label,
                    "prompt_source_id": fallback_source_id,
                    "chunk_index": chunk_index,
                }
            )
        stats["generated_chunks"] += len(chunks)

    dataset = Dataset.from_list(records)
    out_path = cache_dir / "jailbreak.parquet"
    dataset.to_parquet(str(out_path))
    meta = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "prompt_column": prompt_column,
        "label_column": label_column,
        "target_label": target_label,
        "output_label": output_label,
        "max_sentences": max_sentences,
        "stats": stats,
        "cache_file": str(out_path),
        "total_rows": len(dataset),
        "label": output_label,
    }
    return dataset, meta


def build_jailbreak_cache(
    dataset_name: str = JAILBREAK_DATASET_NAME,
    *,
    split_name: str = JAILBREAK_SPLIT,
    prompt_column: str = JAILBREAK_PROMPT_COLUMN,
    label_column: str = JAILBREAK_LABEL_COLUMN,
    target_label: str = JAILBREAK_TARGET_LABEL,
    max_sentences: int = JAILBREAK_MAX_SENTENCES,
    cache_dir: Path = JAILBREAK_CACHE_DIR,
) -> tuple[Dataset, dict[str, Any]]:
    dataset, meta = _build_jackhhao_classification_cache(
        dataset_name=dataset_name,
        split_name=split_name,
        prompt_column=prompt_column,
        label_column=label_column,
        target_label=target_label,
        output_label="Jailbreak",
        max_sentences=max_sentences,
        cache_dir=cache_dir,
    )
    save_json(path("salad", "salad_jailbreak_cache_meta_file"), meta)
    return dataset, meta


def build_jailbreak_benign_cache(
    dataset_name: str = JAILBREAK_DATASET_NAME,
    *,
    split_name: str = JAILBREAK_SPLIT,
    prompt_column: str = JAILBREAK_PROMPT_COLUMN,
    label_column: str = JAILBREAK_LABEL_COLUMN,
    target_label: str = "benign",
    max_sentences: int = JAILBREAK_MAX_SENTENCES,
    cache_dir: Path = JAILBREAK_BENIGN_CACHE_DIR,
) -> tuple[Dataset, dict[str, Any]]:
    dataset, meta = _build_jackhhao_classification_cache(
        dataset_name=dataset_name,
        split_name=split_name,
        prompt_column=prompt_column,
        label_column=label_column,
        target_label=target_label,
        output_label=JAILBREAK_BENIGN_LABEL,
        max_sentences=max_sentences,
        cache_dir=cache_dir,
    )
    save_json(path("salad", "salad_jailbreak_benign_cache_meta_file"), meta)
    return dataset, meta


def ensure_jailbreak_cache(
    dataset_name: str = JAILBREAK_DATASET_NAME,
    *,
    split_name: str = JAILBREAK_SPLIT,
    prompt_column: str = JAILBREAK_PROMPT_COLUMN,
    label_column: str = JAILBREAK_LABEL_COLUMN,
    target_label: str = JAILBREAK_TARGET_LABEL,
    max_sentences: int = JAILBREAK_MAX_SENTENCES,
    cache_dir: Path = JAILBREAK_CACHE_DIR,
) -> tuple[Dataset, dict[str, Any]]:
    meta_file = path("salad", "salad_jailbreak_cache_meta_file")
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        cache_file = meta.get("cache_file")
        if isinstance(cache_file, str) and Path(cache_file).exists():
            return load_jailbreak_cache(cache_dir=cache_dir), meta
    return build_jailbreak_cache(
        dataset_name=dataset_name,
        split_name=split_name,
        prompt_column=prompt_column,
        label_column=label_column,
        target_label=target_label,
        max_sentences=max_sentences,
        cache_dir=cache_dir,
    )


def load_jailbreak_benign_cache(cache_dir: Path = JAILBREAK_BENIGN_CACHE_DIR) -> Dataset:
    meta_file = path("salad", "salad_jailbreak_benign_cache_meta_file")
    if not meta_file.exists():
        raise FileNotFoundError(f"Missing jailbreak benign cache metadata: {meta_file}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    cache_file = meta.get("cache_file")
    if not isinstance(cache_file, str):
        raise ValueError(f"Malformed jailbreak benign cache metadata in {meta_file}")
    out_path = Path(cache_file)
    if not out_path.exists():
        raise FileNotFoundError(f"Missing jailbreak benign cache file: {out_path}")
    return load_local_parquet_dataset(out_path)


def ensure_jailbreak_benign_cache(
    dataset_name: str = JAILBREAK_DATASET_NAME,
    *,
    split_name: str = JAILBREAK_SPLIT,
    prompt_column: str = JAILBREAK_PROMPT_COLUMN,
    label_column: str = JAILBREAK_LABEL_COLUMN,
    target_label: str = "benign",
    max_sentences: int = JAILBREAK_MAX_SENTENCES,
    cache_dir: Path = JAILBREAK_BENIGN_CACHE_DIR,
) -> tuple[Dataset, dict[str, Any]]:
    meta_file = path("salad", "salad_jailbreak_benign_cache_meta_file")
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        cache_file = meta.get("cache_file")
        if isinstance(cache_file, str) and Path(cache_file).exists():
            return load_jailbreak_benign_cache(cache_dir=cache_dir), meta
    return build_jailbreak_benign_cache(
        dataset_name=dataset_name,
        split_name=split_name,
        prompt_column=prompt_column,
        label_column=label_column,
        target_label=target_label,
        max_sentences=max_sentences,
        cache_dir=cache_dir,
    )


def build_openhermes_outside_cache(
    dataset_name: str = NEUTRAL_DATASET_NAME,
    *,
    split_name: str = NEUTRAL_SPLIT,
    conversations_column: str = NEUTRAL_TEXT_COLUMN,
    max_sentences: int = NEUTRAL_MAX_SENTENCES,
    min_latin_ratio: float = NEUTRAL_MIN_LATIN_RATIO,
    sample_fraction: float = NEUTRAL_SAMPLE_FRACTION,
    cache_dir: Path = NEUTRAL_CACHE_DIR,
) -> tuple[Dataset, dict[str, Any]]:
    raw = _load_split(dataset_name, None, split_name, sample_fraction=sample_fraction)
    filtered, filter_stats = _filter_openhermes_split(
        raw,
        conversations_column=conversations_column,
        max_sentences=max_sentences,
        min_latin_ratio=min_latin_ratio,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "source_id": int(idx),
            "text": _first_human_turn(row.get(conversations_column, "")),
            "label": "outside",
        }
        for idx, row in enumerate(filtered)
    ]
    dataset = Dataset.from_list(records)
    out_path = cache_dir / "outside.parquet"
    dataset.to_parquet(str(out_path))

    meta = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "conversations_column": conversations_column,
        "max_sentences": max_sentences,
        "min_latin_ratio": min_latin_ratio,
        "sample_fraction": sample_fraction,
        "filter_stats": filter_stats,
        "cache_file": str(out_path),
        "total_rows": filter_stats["kept"],
        "label": "outside",
    }
    save_json(path("salad", "salad_outside_cache_meta_file"), meta)
    return dataset, meta


def ensure_openhermes_outside_cache(
    dataset_name: str = NEUTRAL_DATASET_NAME,
    *,
    split_name: str = NEUTRAL_SPLIT,
    conversations_column: str = NEUTRAL_TEXT_COLUMN,
    max_sentences: int = NEUTRAL_MAX_SENTENCES,
    min_latin_ratio: float = NEUTRAL_MIN_LATIN_RATIO,
    sample_fraction: float = NEUTRAL_SAMPLE_FRACTION,
    cache_dir: Path = NEUTRAL_CACHE_DIR,
) -> tuple[Dataset, dict[str, Any]]:
    meta_file = path("salad", "salad_outside_cache_meta_file")
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        cache_file = meta.get("cache_file")
        if isinstance(cache_file, str) and Path(cache_file).exists():
            return load_openhermes_outside_cache(cache_dir=cache_dir), meta
    return build_openhermes_outside_cache(
        dataset_name=dataset_name,
        split_name=split_name,
        conversations_column=conversations_column,
        max_sentences=max_sentences,
        min_latin_ratio=min_latin_ratio,
        sample_fraction=sample_fraction,
        cache_dir=cache_dir,
    )
