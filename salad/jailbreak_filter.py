from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

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

from salad.defaults import (
    JAILBREAK_DATASET_NAME,
    JAILBREAK_FILTER_DIR,
    JAILBREAK_LABEL_COLUMN,
    JAILBREAK_PROMPT_COLUMN,
    JAILBREAK_SPLIT,
)


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
WHITESPACE_RE = re.compile(r"\s+")

MODEL_FILE = JAILBREAK_FILTER_DIR / "model.joblib"
METRICS_FILE = JAILBREAK_FILTER_DIR / "metrics.json"
META_FILE = JAILBREAK_FILTER_DIR / "meta.json"
TOP_FEATURES_FILE = JAILBREAK_FILTER_DIR / "top_features.json"
HF_CACHE_DIR = JAILBREAK_FILTER_DIR / "hf_cache"


@dataclass
class FilterMetrics:
    total_rows: int
    kept_rows: int
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


def clean_jailbreak_text(text: str) -> str:
    cleaned = str(text or "").strip().lower()
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = NON_ALNUM_RE.sub(" ", cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def load_jailbreak_rows(
    *,
    dataset_name: str = JAILBREAK_DATASET_NAME,
    split_name: str = JAILBREAK_SPLIT,
    prompt_column: str = JAILBREAK_PROMPT_COLUMN,
    label_column: str = JAILBREAK_LABEL_COLUMN,
) -> tuple[list[str], list[str], dict[str, int]]:
    raw = load_dataset(dataset_name, split=split_name, cache_dir=str(HF_CACHE_DIR))
    texts: list[str] = []
    labels: list[str] = []
    stats = {"total": 0, "kept": 0, "dropped_empty": 0, "dropped_label": 0}
    for row in tqdm(raw, desc="Loading jailbreak dataset"):
        stats["total"] += 1
        label = str(row.get(label_column, "")).strip().lower()
        if label not in {"jailbreak", "benign"}:
            stats["dropped_label"] += 1
            continue
        text = str(row.get(prompt_column, "")).strip()
        if not text:
            stats["dropped_empty"] += 1
            continue
        texts.append(text)
        labels.append(label)
        stats["kept"] += 1
    return texts, labels, stats


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
    threshold: float,
) -> dict[str, Any]:
    probs = pipeline.predict_proba(texts)
    classes = list(pipeline.named_steps["clf"].classes_)
    if "jailbreak" not in classes or "benign" not in classes:
        raise ValueError(f"Expected binary classes benign/jailbreak, got {classes}")
    jailbreak_index = classes.index("jailbreak")
    jailbreak_scores = probs[:, jailbreak_index]
    predictions = np.array(["jailbreak" if score >= threshold else "benign" for score in jailbreak_scores])

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        pos_label="jailbreak",
        zero_division=0,
    )
    accuracy = float((predictions == np.array(labels)).mean())
    report = classification_report(labels, predictions, digits=4, zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=["benign", "jailbreak"]).tolist()
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classification_report": report,
        "confusion_matrix": matrix,
        "scores": jailbreak_scores.tolist(),
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
        "jailbreak": [
            {"feature": str(feature_names[idx]), "weight": float(coefficients[idx])}
            for idx in order_pos
        ],
        "benign": [
            {"feature": str(feature_names[idx]), "weight": float(coefficients[idx])}
            for idx in order_neg
        ],
    }


def fit_and_save(
    *,
    ngram_max: int,
    min_df: int,
    max_features: int | None,
    test_size: float,
    random_state: int,
    threshold: float,
    top_n: int,
) -> dict[str, Any]:
    texts, labels, load_stats = load_jailbreak_rows()
    if not texts:
        raise RuntimeError("No jailbreak/benign rows found in jackhhao/jailbreak-classification")

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    pipeline = build_pipeline(ngram_max=ngram_max, min_df=min_df, max_features=max_features)
    pipeline.fit(x_train, y_train)
    eval_metrics = evaluate_pipeline(pipeline, x_test, y_test, threshold=threshold)
    feature_summary = top_features(pipeline, top_n=top_n)

    JAILBREAK_FILTER_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)

    meta = FilterMetrics(
        total_rows=load_stats["total"],
        kept_rows=load_stats["kept"],
        train_rows=len(x_train),
        test_rows=len(x_test),
        test_accuracy=eval_metrics["accuracy"],
        test_precision=eval_metrics["precision"],
        test_recall=eval_metrics["recall"],
        test_f1=eval_metrics["f1"],
        positive_label="jailbreak",
        negative_label="benign",
        threshold=threshold,
        ngram_max=ngram_max,
        min_df=min_df,
        max_features=max_features,
    )

    save_payload = {
        "metrics": eval_metrics,
        "load_stats": load_stats,
        "meta": asdict(meta),
        "top_features": feature_summary,
    }
    with METRICS_FILE.open("w", encoding="utf-8") as f:
        json.dump(save_payload["metrics"], f, ensure_ascii=False, indent=2)
    with META_FILE.open("w", encoding="utf-8") as f:
        json.dump(save_payload["meta"], f, ensure_ascii=False, indent=2)
    with TOP_FEATURES_FILE.open("w", encoding="utf-8") as f:
        json.dump(save_payload["top_features"], f, ensure_ascii=False, indent=2)

    final_pipeline = build_pipeline(ngram_max=ngram_max, min_df=min_df, max_features=max_features)
    final_pipeline.fit(texts, labels)
    joblib.dump(final_pipeline, MODEL_FILE)

    return save_payload


def load_filter_model(model_file: Path = MODEL_FILE) -> Pipeline:
    if not model_file.exists():
        raise FileNotFoundError(f"Missing trained filter model: {model_file}")
    model = joblib.load(model_file)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline in {model_file}, got {type(model)!r}")
    return model


def score_texts(texts: Iterable[str], *, model_file: Path = MODEL_FILE) -> list[float]:
    model = load_filter_model(model_file)
    probs = model.predict_proba(list(texts))
    classes = list(model.named_steps["clf"].classes_)
    jailbreak_index = classes.index("jailbreak")
    return probs[:, jailbreak_index].tolist()


def keep_mask(texts: Iterable[str], *, threshold: float = 0.5, model_file: Path = MODEL_FILE) -> list[bool]:
    scores = score_texts(texts, model_file=model_file)
    return [score >= threshold for score in scores]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TF-IDF + logistic regression jailbreak filter.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows reserved for evaluation.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for the train/test split.")
    parser.add_argument("--ngram-max", type=int, default=2, help="Maximum TF-IDF n-gram size.")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for TF-IDF features.")
    parser.add_argument("--max-features", type=int, default=100_000, help="Maximum TF-IDF vocabulary size.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for labeling a chunk as jailbreak.",
    )
    parser.add_argument("--top-n", type=int, default=30, help="How many top features to save for each class.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_features = None if args.max_features <= 0 else args.max_features
    results = fit_and_save(
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_features=max_features,
        test_size=args.test_size,
        random_state=args.random_state,
        threshold=args.threshold,
        top_n=args.top_n,
    )

    print("=" * 80)
    print("JAILBREAK FILTER TRAINING")
    print("=" * 80)
    print(f"Model file: {MODEL_FILE}")
    print(f"Meta file: {META_FILE}")
    print(f"Metrics file: {METRICS_FILE}")
    print(f"Top features file: {TOP_FEATURES_FILE}")
    print()
    print(results["metrics"]["classification_report"])
    print("Confusion matrix [benign, jailbreak]:")
    print(results["metrics"]["confusion_matrix"])
    print()
    print("Top jailbreak features:")
    for item in results["top_features"]["jailbreak"][:10]:
        print(f"  {item['feature']}: {item['weight']:.4f}")
    print("Top benign features:")
    for item in results["top_features"]["benign"][:10]:
        print(f"  {item['feature']}: {item['weight']:.4f}")


if __name__ == "__main__":
    main()
