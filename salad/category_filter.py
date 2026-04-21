from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from paths import path
from salad.cache import ensure_openhermes_outside_cache
from salad.defaults import (
    DATASET_NAME,
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
    SALAD_CATEGORY_FILTER_DIR,
)
from salad.jailbreak_filter import clean_jailbreak_text
from salad.labels import slugify_label


SENTENCE_SPLIT_RE = __import__("re").compile(r"(?<=[.!?])\s+|\n+")
NEWLINE_SPLIT_RE = __import__("re").compile(r"(?:\r?\n){1,2}")

MODEL_ROOT = SALAD_CATEGORY_FILTER_DIR
META_FILE = path("salad", "salad_category_filter_meta_file")


@dataclass
class CategoryFilterMetrics:
    category: str
    total_rows: int
    positive_rows: int
    negative_rows: int
    positive_chunks: int
    negative_chunks: int
    train_rows: int
    test_rows: int
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    positive_label: str
    negative_label: str
    threshold: float
    ngram_max: int
    min_df: int
    max_features: int | None


def sentence_count(text: str) -> int:
    pieces = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(text.strip()) if piece.strip()]
    return len(pieces)


def latin_ratio(text: str) -> float:
    letters = 0
    latin_letters = 0
    for char in text:
        if not __import__("unicodedata").category(char).startswith("L"):
            continue
        letters += 1
        if __import__("unicodedata").name(char, "").startswith("LATIN"):
            latin_letters += 1
    if letters == 0:
        return 0.0
    return latin_letters / letters


def is_majority_latin(text: str, *, min_ratio: float = MIN_LATIN_RATIO) -> bool:
    return latin_ratio(text) >= min_ratio


def split_segments(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
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


def sliding_windows(items: list[str], window_size: int, stride: int = 1) -> list[list[str]]:
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


def chunk_text(text: str, *, max_sentences: int = MAX_SENTENCES) -> list[str]:
    segments = split_segments(text)
    if not segments:
        return []
    if len(segments) <= max_sentences:
        return [" ".join(segments)]
    return [" ".join(window) for window in sliding_windows(segments, max_sentences, stride=1)]


def resolve_label(value: Any, label_names: list[str]) -> str:
    if isinstance(value, (int, np.integer)):
        index = int(value)
        if 0 <= index < len(label_names):
            return label_names[index]
        return str(index)
    if isinstance(value, str):
        value = value.strip()
        if value.isdigit():
            index = int(value)
            if 0 <= index < len(label_names):
                return label_names[index]
        return value
    return str(value)


def load_salad_positive_chunks(
    *,
    dataset_name: str = DATASET_NAME,
    subset: str = "base_set",
    split_name: str = "train",
    label_column: str = LABEL_COLUMN,
    text_column: str = "question",
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    raw = load_dataset(dataset_name, subset, split=split_name)
    label_feature = raw.features[label_column]
    if hasattr(label_feature, "names"):
        label_names = [str(name) for name in label_feature.names]
    else:
        label_names = []
        for row in raw:
            label = str(row.get(label_column, "")).strip()
            if label and label not in label_names:
                label_names.append(label)
        if not label_names:
            raise TypeError(f"Could not determine label names for {label_column}")
        raw = load_dataset(dataset_name, subset, split=split_name)
    chunks_by_label = {label: [] for label in label_names}
    stats = {
        "total_rows": 0,
        "kept_rows": 0,
        "dropped_empty": 0,
        "dropped_script": 0,
        "chunked_rows": 0,
        "generated_chunks": 0,
    }

    for row in tqdm(raw, desc="Loading Salad-Data category rows"):
        stats["total_rows"] += 1
        label = resolve_label(row.get(label_column, ""), label_names)
        if label not in chunks_by_label:
            continue
        text = str(row.get(text_column, "")).strip()
        if not text:
            stats["dropped_empty"] += 1
            continue
        if not is_majority_latin(text, min_ratio=MIN_LATIN_RATIO):
            stats["dropped_script"] += 1
            continue
        chunks = chunk_text(text, max_sentences=MAX_SENTENCES)
        if not chunks:
            continue
        if len(chunks) > 1:
            stats["chunked_rows"] += 1
        chunks_by_label[label].extend(chunks)
        stats["generated_chunks"] += len(chunks)
        stats["kept_rows"] += 1

    return chunks_by_label, {"label_names": label_names, "stats": stats}


def load_neutral_chunks(
    *,
    dataset_name: str = NEUTRAL_DATASET_NAME,
    split_name: str = NEUTRAL_SPLIT,
    conversations_column: str = NEUTRAL_TEXT_COLUMN,
    sample_fraction: float = NEUTRAL_SAMPLE_FRACTION,
    max_sentences: int = NEUTRAL_MAX_SENTENCES,
    min_latin_ratio: float = NEUTRAL_MIN_LATIN_RATIO,
) -> tuple[list[str], dict[str, Any]]:
    outside_split, meta = ensure_openhermes_outside_cache(
        dataset_name=dataset_name,
        split_name=split_name,
        conversations_column=conversations_column,
        max_sentences=max_sentences,
        min_latin_ratio=min_latin_ratio,
        sample_fraction=sample_fraction,
        cache_dir=NEUTRAL_CACHE_DIR,
    )
    chunks: list[str] = []
    stats = {
        "rows": 0,
        "generated_chunks": 0,
    }
    for row in outside_split:
        stats["rows"] += 1
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        row_chunks = chunk_text(text, max_sentences=MAX_SENTENCES)
        chunks.extend(row_chunks)
        stats["generated_chunks"] += len(row_chunks)
    return chunks, {"cache_meta": meta, "stats": stats}


def build_pipeline(*, ngram_max: int, min_df: int, max_features: int | None) -> Pipeline:
    vectorizer = TfidfVectorizer(
        preprocessor=clean_jailbreak_text,
        lowercase=False,
        strip_accents="unicode",
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )
    classifier = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=2_000,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def evaluate_pipeline(
    pipeline: Pipeline,
    texts: list[str],
    labels: list[str],
    *,
    positive_label: str,
    negative_label: str,
    threshold: float,
) -> dict[str, Any]:
    probs = pipeline.predict_proba(texts)
    classes = list(pipeline.named_steps["clf"].classes_)
    if positive_label not in classes or negative_label not in classes:
        raise ValueError(f"Expected classes {positive_label!r}/{negative_label!r}, got {classes}")
    positive_index = classes.index(positive_label)
    scores = probs[:, positive_index]
    predictions = np.array(
        [positive_label if score >= threshold else negative_label for score in scores]
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        pos_label=positive_label,
        zero_division=0,
    )
    accuracy = float((predictions == np.array(labels)).mean())
    report = classification_report(labels, predictions, digits=4, zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=[negative_label, positive_label]).tolist()
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classification_report": report,
        "confusion_matrix": matrix,
        "scores": scores.tolist(),
        "predictions": predictions.tolist(),
    }


def top_features(pipeline: Pipeline, *, top_n: int = 30) -> dict[str, list[dict[str, float | str]]]:
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    classifier: LogisticRegression = pipeline.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    order_pos = np.argsort(coefficients)[::-1][:top_n]
    order_neg = np.argsort(coefficients)[:top_n]

    return {
        "positive": [
            {"feature": str(feature_names[idx]), "weight": float(coefficients[idx])}
            for idx in order_pos
        ],
        "negative": [
            {"feature": str(feature_names[idx]), "weight": float(coefficients[idx])}
            for idx in order_neg
        ],
    }


def fit_and_save(
    *,
    test_size: float,
    random_state: int,
    threshold: float,
    ngram_max: int,
    min_df: int,
    max_features: int | None,
    top_n: int,
) -> dict[str, Any]:
    positive_chunks_by_label, salad_meta = load_salad_positive_chunks()
    neutral_chunks, neutral_meta = load_neutral_chunks()
    if not neutral_chunks:
        raise RuntimeError("No neutral chunks available for category filter training")

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "dataset": {
            "salad": salad_meta,
            "neutral": neutral_meta,
        },
        "models": {},
    }

    rng = random.Random(random_state)
    for category_label, positive_chunks in positive_chunks_by_label.items():
        if not positive_chunks:
            continue

        target_negative_count = min(len(neutral_chunks), max(1, len(positive_chunks)))
        if target_negative_count < len(neutral_chunks):
            negative_chunks = rng.sample(neutral_chunks, target_negative_count)
        else:
            negative_chunks = list(neutral_chunks)

        texts = [*positive_chunks, *negative_chunks]
        labels = [category_label] * len(positive_chunks) + ["outside"] * len(negative_chunks)
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        pipeline = build_pipeline(ngram_max=ngram_max, min_df=min_df, max_features=max_features)
        pipeline.fit(x_train, y_train)
        eval_metrics = evaluate_pipeline(
            pipeline,
            x_test,
            y_test,
            positive_label=category_label,
            negative_label="outside",
            threshold=threshold,
        )
        feature_summary = top_features(pipeline, top_n=top_n)

        model_dir = MODEL_ROOT / slugify_label(category_label)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / "model.joblib"
        metrics_file = model_dir / "metrics.json"
        meta_file = model_dir / "meta.json"
        features_file = model_dir / "top_features.json"

        joblib.dump(pipeline, model_file)

        metrics_payload = {
            **eval_metrics,
            "positive_label": category_label,
            "negative_label": "outside",
        }
        meta_payload = asdict(
            CategoryFilterMetrics(
                category=category_label,
                total_rows=len(texts),
                positive_rows=len(positive_chunks),
                negative_rows=len(negative_chunks),
                positive_chunks=len(positive_chunks),
                negative_chunks=len(negative_chunks),
                train_rows=len(x_train),
                test_rows=len(x_test),
                test_accuracy=eval_metrics["accuracy"],
                test_precision=eval_metrics["precision"],
                test_recall=eval_metrics["recall"],
                test_f1=eval_metrics["f1"],
                positive_label=category_label,
                negative_label="outside",
                threshold=threshold,
                ngram_max=ngram_max,
                min_df=min_df,
                max_features=max_features,
            )
        )
        with metrics_file.open("w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
        with meta_file.open("w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)
        with features_file.open("w", encoding="utf-8") as f:
            json.dump(feature_summary, f, ensure_ascii=False, indent=2)

        results["models"][category_label] = {
            "model_file": str(model_file),
            "metrics_file": str(metrics_file),
            "meta_file": str(meta_file),
            "top_features_file": str(features_file),
            "metrics": metrics_payload,
            "meta": meta_payload,
        }

    with META_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Salad category chunk filters.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Train/test split seed.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for keeping chunks.")
    parser.add_argument("--ngram-max", type=int, default=2, help="Maximum TF-IDF n-gram size.")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for TF-IDF features.")
    parser.add_argument("--max-features", type=int, default=100_000, help="Maximum TF-IDF vocabulary size.")
    parser.add_argument("--top-n", type=int, default=30, help="Top positive/negative features to save.")
    args = parser.parse_args()

    results = fit_and_save(
        test_size=args.test_size,
        random_state=args.random_state,
        threshold=args.threshold,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_features=args.max_features,
        top_n=args.top_n,
    )

    print(f"Wrote Salad category filter metadata to {META_FILE}")
    for label, info in results["models"].items():
        print(f"{label}: {info['model_file']}")


if __name__ == "__main__":
    main()
