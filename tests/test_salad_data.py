#!/usr/bin/env python3
"""
Evaluate a safety classifier on OpenSafetyLab/Salad-Data base_set using the
`1-category` taxonomy.

The script filters to examples with at most `max_sentences` sentences and a
majority of Latin-script letters before scoring. It supports either a
sequence-classification model or a token-classification model that is collapsed
to one label per example.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, project_root)


DATASET_NAME = "OpenSafetyLab/Salad-Data"
DATASET_SUBSET = "base_set"
DEFAULT_TEXT_COLUMN = "question"
DEFAULT_LABEL_COLUMN = "1-category"
DEFAULT_MODEL_NAME = "roberta-base"
DEFAULT_TOKENIZER_NAME = "roberta-base"
DEFAULT_RESULTS_DIR = Path(project_root) / "results" / "salad_data_eval"
DEFAULT_MAX_SENTENCES = 3
DEFAULT_MIN_LATIN_RATIO = 0.5

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on OpenSafetyLab/Salad-Data base_set.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Model checkpoint or local path.")
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=DEFAULT_TOKENIZER_NAME,
        help="Tokenizer checkpoint or local path.",
    )
    parser.add_argument(
        "--task-type",
        choices=("sequence-classification", "token-classification"),
        default="sequence-classification",
        help="How to interpret model outputs.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--subset", type=str, default=DATASET_SUBSET, help="Dataset subset to use.")
    parser.add_argument("--text-column", type=str, default=DEFAULT_TEXT_COLUMN, help="Text column to score.")
    parser.add_argument("--label-column", type=str, default=DEFAULT_LABEL_COLUMN, help="Label column to score.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum tokenizer length.")
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=DEFAULT_MAX_SENTENCES,
        help="Keep only texts with at most this many sentences.",
    )
    parser.add_argument(
        "--min-latin-ratio",
        type=float,
        default=DEFAULT_MIN_LATIN_RATIO,
        help="Keep only texts with at least this fraction of Latin-script letters.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the filtered evaluation set.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Output directory.")
    return parser.parse_args()


def _load_dataset(split_name: str, subset: str):
    dataset = load_dataset(DATASET_NAME, name=subset, split=split_name)
    return dataset


def _sentence_count(text: str) -> int:
    sentences = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(text.strip()) if piece.strip()]
    return len(sentences)


def _latin_ratio(text: str) -> float:
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


def _is_majority_latin(text: str, *, min_ratio: float) -> bool:
    return _latin_ratio(text) >= min_ratio


def _normalize_label(label: str) -> str:
    label = label.strip()
    if label.lower().startswith(("b-", "i-")):
        label = label[2:]
    return label


def _label_names_from_dataset(dataset, label_column: str) -> list[str]:
    feature = dataset.features[label_column]
    if hasattr(feature, "names"):
        return [str(name) for name in feature.names]
    return sorted({str(row[label_column]) for row in dataset})


def _label_from_example(example: dict[str, Any], dataset, label_column: str, label_names: list[str]) -> str:
    value = example[label_column]
    feature = dataset.features[label_column]
    if hasattr(feature, "int2str") and isinstance(value, int):
        return _normalize_label(feature.int2str(value))
    if hasattr(feature, "names") and isinstance(value, int):
        return _normalize_label(feature.names[value])
    raw_label = _normalize_label(str(value))
    if raw_label in label_names:
        return raw_label
    return raw_label


def _dataset_label_order(dataset, label_column: str, label_names: list[str]) -> list[str]:
    return [_normalize_label(label) for label in label_names]


def _model_label_to_dataset_label(label_name: str, idx: int, dataset_labels: list[str]) -> str:
    normalized = _normalize_label(label_name)
    if normalized in dataset_labels:
        return normalized
    if normalized.startswith("label_") and idx < len(dataset_labels):
        return dataset_labels[idx]
    if idx < len(dataset_labels):
        return dataset_labels[idx]
    return normalized


def _batched(items: list[str], batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _filter_dataset(
    dataset,
    *,
    text_column: str,
    max_sentences: int,
    min_latin_ratio: float,
) -> tuple[Any, dict[str, int]]:
    kept_indices: list[int] = []
    stats = {
        "total": 0,
        "kept": 0,
        "dropped_empty": 0,
        "dropped_sentence_count": 0,
        "dropped_script": 0,
    }

    for idx, example in enumerate(tqdm(dataset, desc="Filtering examples")):
        stats["total"] += 1
        text = str(example.get(text_column, "")).strip()
        if not text:
            stats["dropped_empty"] += 1
            continue
        if _sentence_count(text) > max_sentences:
            stats["dropped_sentence_count"] += 1
            continue
        if not _is_majority_latin(text, min_ratio=min_latin_ratio):
            stats["dropped_script"] += 1
            continue
        kept_indices.append(idx)
        stats["kept"] += 1

    return dataset.select(kept_indices), stats


def _predict_sequence_classification(
    texts: list[str],
    *,
    model,
    tokenizer,
    batch_size: int,
    max_length: int,
    dataset_labels: list[str],
) -> list[dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    id2label = {int(idx): str(label) for idx, label in getattr(model.config, "id2label", {}).items()}
    total_batches = (len(texts) + batch_size - 1) // batch_size
    predictions: list[dict[str, Any]] = []

    for batch_texts in tqdm(_batched(texts, batch_size), total=total_batches, desc="Scoring"):
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        for row_probs in probabilities:
            label_scores: dict[str, float] = {}
            ranked = sorted(
                ((idx, float(score)) for idx, score in enumerate(row_probs)),
                key=lambda item: item[1],
                reverse=True,
            )
            for idx, score in ranked:
                label_name = id2label.get(idx, f"label_{idx}")
                dataset_label = _model_label_to_dataset_label(label_name, idx, dataset_labels)
                if dataset_label in label_scores:
                    continue
                label_scores[dataset_label] = score
            for label in dataset_labels:
                label_scores.setdefault(label, 0.0)
            predicted = max(dataset_labels, key=lambda label: label_scores[label])
            predictions.append({"predicted": predicted, "scores": label_scores})

    return predictions


def _predict_token_classification(
    texts: list[str],
    *,
    model,
    tokenizer,
    batch_size: int,
    max_length: int,
    dataset_labels: list[str],
) -> list[dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    id2label = {int(idx): str(label) for idx, label in getattr(model.config, "id2label", {}).items()}
    total_batches = (len(texts) + batch_size - 1) // batch_size
    predictions: list[dict[str, Any]] = []

    for batch_texts in tqdm(_batched(texts, batch_size), total=total_batches, desc="Scoring"):
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special_tokens_mask = encoded.pop("special_tokens_mask").bool()
        attention_mask = encoded["attention_mask"].bool()
        keep_mask = attention_mask & ~special_tokens_mask
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        keep_mask_np = keep_mask.detach().cpu().numpy()

        for row_probs, row_keep in zip(probabilities, keep_mask_np):
            token_count = int(row_keep.sum())
            if token_count == 0:
                predictions.append(
                    {
                        "predicted": dataset_labels[0],
                        "scores": {label: 0.0 for label in dataset_labels},
                        "token_count": 0,
                    }
                )
                continue

            score_sums = {label: 0.0 for label in dataset_labels}
            for token_probs, keep in zip(row_probs, row_keep):
                if not keep:
                    continue
                for idx, score in enumerate(token_probs):
                    label_name = id2label.get(idx, f"label_{idx}")
                    dataset_label = _model_label_to_dataset_label(label_name, idx, dataset_labels)
                    score_sums[dataset_label] += float(score)

            averaged_scores = {label: score / token_count for label, score in score_sums.items()}
            predicted = max(dataset_labels, key=lambda label: averaged_scores[label])
            predictions.append(
                {
                    "predicted": predicted,
                    "scores": averaged_scores,
                    "token_count": token_count,
                }
            )

    return predictions


def _compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion[label_to_index[true_label], label_to_index[pred_label]] += 1

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = float(correct / total) if total else 0.0

    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for label in labels:
        idx = label_to_index[label]
        tp = int(confusion[idx, idx])
        fp = int(confusion[:, idx].sum() - tp)
        fn = int(confusion[idx, :].sum() - tp)
        support = int(confusion[idx, :].sum())
        predicted_total = int(confusion[:, idx].sum())
        precision = float(tp / predicted_total) if predicted_total else 0.0
        recall = float(tp / support) if support else 0.0
        f1 = float((2 * precision * recall / (precision + recall))) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "total": total,
        "correct": correct,
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def main() -> None:
    args = _parse_args()

    print("# Salad-Data Safety Evaluation")
    print()
    print("## Setup")
    print(f"- Model: `{args.model_name}`")
    print(f"- Tokenizer: `{args.tokenizer_name}`")
    print(f"- Task type: `{args.task_type}`")
    print(f"- Dataset: `{DATASET_NAME}` / `{args.subset}` / `{args.split}`")

    dataset = _load_dataset(args.split, args.subset)
    label_names = _label_names_from_dataset(dataset, args.label_column)
    dataset_labels = _dataset_label_order(dataset, args.label_column, label_names)

    print()
    print("## Filters")
    print(f"- Max sentences: `{args.max_sentences}`")
    print(f"- Min Latin ratio: `{args.min_latin_ratio:.2f}`")
    print(f"- Dataset label count: `{len(dataset_labels)}`")
    print(f"- Labels: `{', '.join(label_names)}`")

    filtered_dataset, filter_stats = _filter_dataset(
        dataset,
        text_column=args.text_column,
        max_sentences=args.max_sentences,
        min_latin_ratio=args.min_latin_ratio,
    )

    if args.limit is not None:
        filtered_dataset = filtered_dataset.select(range(min(args.limit, len(filtered_dataset))))

    print(f"- Kept: `{len(filtered_dataset)}` of `{filter_stats['total']}`")
    print(f"- Dropped for empty text: `{filter_stats['dropped_empty']}`")
    print(f"- Dropped for sentence count: `{filter_stats['dropped_sentence_count']}`")
    print(f"- Dropped for script ratio: `{filter_stats['dropped_script']}`")

    if len(filtered_dataset) == 0:
        raise RuntimeError("No examples left after filtering.")

    texts = [str(example[args.text_column]) for example in filtered_dataset]

    print()
    print("## Inference")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if args.task_type == "sequence-classification":
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
        predictions = _predict_sequence_classification(
            texts,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            dataset_labels=dataset_labels,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(args.model_name)
        predictions = _predict_token_classification(
            texts,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            dataset_labels=dataset_labels,
        )

    y_true: list[str] = []
    y_pred: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for example, text, prediction in tqdm(
        zip(filtered_dataset, texts, predictions),
        total=len(filtered_dataset),
        desc="Collecting metrics",
    ):
        true_label = _label_from_example(example, filtered_dataset, args.label_column, label_names)
        predicted_label = str(prediction["predicted"])
        y_true.append(true_label)
        y_pred.append(predicted_label)
        if true_label != predicted_label:
            mismatch: dict[str, Any] = {
                "text_preview": text[:240],
                "true_label": true_label,
                "predicted": predicted_label,
                "scores": prediction["scores"],
            }
            if "token_count" in prediction:
                mismatch["token_count"] = int(prediction["token_count"])
            mismatches.append(mismatch)

    metrics = _compute_metrics(y_true, y_pred, dataset_labels)

    print()
    print("## Overall Results")
    print("| Model | Acc | Macro F1 | Macro P | Macro R |")
    print("|---|---:|---:|---:|---:|")
    print(
        f"| {args.task_type} | {_fmt_pct(metrics['accuracy'])} | {_fmt_pct(metrics['macro_f1'])} | "
        f"{_fmt_pct(metrics['macro_precision'])} | {_fmt_pct(metrics['macro_recall'])} |"
    )

    print()
    print("## Per-Class Breakdown")
    print("| Label | Precision | Recall | F1 | Support |")
    print("|---|---:|---:|---:|---:|")
    for label in dataset_labels:
        stats = metrics["per_class"][label]
        print(
            f"| {label} | {_fmt_pct(stats['precision'])} | {_fmt_pct(stats['recall'])} | "
            f"{_fmt_pct(stats['f1'])} | {stats['support']} |"
        )

    print()
    print("## Confusion Matrix")
    print("| True \\ Pred | " + " | ".join(dataset_labels) + " |")
    print("|---|" + "---:|" * len(dataset_labels))
    for label, row in zip(dataset_labels, metrics["confusion_matrix"]):
        print(f"| {label} | " + " | ".join(str(int(value)) for value in row) + " |")

    print()
    print("## Sample Errors")
    if mismatches:
        for sample_error in mismatches[:10]:
            print(f"- `{sample_error['true_label']}` -> `{sample_error['predicted']}`")
            print(f"  - Scores: `{sample_error['scores']}`")
            print(f"  - Preview: {sample_error['text_preview']}...")
    else:
        print("- No mismatches found.")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    results_output = args.results_dir / "results.json"
    mismatch_output = args.results_dir / "mismatches.jsonl"

    _save_json(
        results_output,
        {
            "dataset": {
                "name": DATASET_NAME,
                "subset": args.subset,
                "split": args.split,
                "text_column": args.text_column,
                "label_column": args.label_column,
                "total_rows": filter_stats["total"],
                "kept_rows": len(filtered_dataset),
                "filter_stats": filter_stats,
                "max_sentences": args.max_sentences,
                "min_latin_ratio": args.min_latin_ratio,
            },
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name,
            "task_type": args.task_type,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "label_names": dataset_labels,
            "metrics": metrics,
        },
    )
    with mismatch_output.open("w", encoding="utf-8") as f:
        for item in mismatches:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print()
    print("## Saved Output")
    print(f"- Results: `{results_output}`")
    print(f"- Mismatches: `{mismatch_output}`")


if __name__ == "__main__":
    main()
