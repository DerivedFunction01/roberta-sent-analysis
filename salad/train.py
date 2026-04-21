from __future__ import annotations

# %% [markdown]
# # Salad Token Classification
# Fine-tunes RoBERTa on the tokenized Salad censorship cache built by `train_pipeline.py`.

# %%
# --- Environment Setup ---
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from paths import path
from salad.labels import id2label as build_id2label, load_label_map

from huggingface_hub import login
with open("hf_token", "r", encoding="utf-8") as f:
    token = f.read().strip()
login(token=token)


# %%
# --- Constants ---
TOKENIZED_DATASET_DIR = path("salad", "salad_tokenized_dataset_dir")
TRAINING_OUTPUT_DIR = path("salad", "salad_pipeline_results_dir") / "salad-token-classifier"
MODEL_CHECKPOINT = "roberta-base"
SEED = 42

# %%
# --- Helpers ---
def get_workers(split: int = 2) -> int:
    return max(1, mp.cpu_count() // split)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_tokenized_cache(data_dir: Path) -> dict[str, Dataset]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Tokenized cache not found: {data_dir}")

    loaded = load_from_disk(str(data_dir))
    if isinstance(loaded, DatasetDict):
        return {name: split for name, split in loaded.items()}
    if isinstance(loaded, Dataset):
        return {"train": loaded}
    raise TypeError(f"Unsupported dataset cache type: {type(loaded)!r}")


def choose_split(splits: dict[str, Dataset], *names: str, required: bool = True) -> Dataset | None:
    for name in names:
        if name in splits:
            return splits[name]
    if required:
        raise KeyError(f"Could not find any of these splits: {names}")
    return None


def make_compute_metrics(id2label: dict[int, str]):
    label_ids = sorted(id2label.keys())

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predicted_ids = np.argmax(predictions, axis=-1)
        mask = labels != -100
        true_ids = labels[mask]
        pred_ids = predicted_ids[mask]

        if true_ids.size == 0:
            return {
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
            }

        accuracy = float(np.mean(true_ids == pred_ids))
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        for label_id in label_ids:
            tp = float(np.sum((pred_ids == label_id) & (true_ids == label_id)))
            fp = float(np.sum((pred_ids == label_id) & (true_ids != label_id)))
            fn = float(np.sum((pred_ids != label_id) & (true_ids == label_id)))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return {
            "accuracy": accuracy,
            "macro_precision": float(np.mean(precisions)),
            "macro_recall": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1s)),
        }

    return compute_metrics


def make_training_args(
    output_dir: str,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    eval_steps: int,
    save_steps: int,
    epochs: float,
    gradient_accumulation_steps: int,
):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=5e-5,
        weight_decay=0.01,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        logging_steps=max(1, min(eval_steps, save_steps) // 2),
        save_total_limit=2,
        report_to="tensorboard",
        seed=SEED,
        dataloader_num_workers=get_workers(),
        push_to_hub=True,
    )


def make_trainer(
    *,
    model,
    train_dataset,
    eval_dataset,
    data_collator,
    compute_metrics,
    output_dir: str,
    epochs: float = 2,
    eval_steps: int = 1000,
    save_steps: int = 1000,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
):
    return Trainer(
        model=model,
        args=make_training_args(
            output_dir,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            eval_steps=eval_steps,
            save_steps=save_steps,
            epochs=epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


# %%
# --- Reproducibility ---
torch.manual_seed(SEED)
np.random.seed(SEED)


# %%
# --- Label Map ---
label2id = load_label_map()
id2label = build_id2label(label2id)


# %%
# --- Tokenized Cache ---
splits = load_tokenized_cache(TOKENIZED_DATASET_DIR)

train_dataset = choose_split(splits, "train")
validation_dataset = choose_split(splits, "validation", "eval", "dev")
test_dataset = choose_split(splits, "test", "validation", "eval", "dev", required=False)

if train_dataset is None or validation_dataset is None:
    raise RuntimeError("Missing required train or validation split in tokenized cache")

print(f"Train: {len(train_dataset)} | Validation: {len(validation_dataset)}")
if test_dataset is not None:
    print(f"Test: {len(test_dataset)}")


# %%
# --- Tokenizer and Model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)


# %%
# --- Sample Inspection ---
first_example = train_dataset[0]
tokens = tokenizer.convert_ids_to_tokens(first_example["input_ids"])
labels = [id2label[label] for label in first_example["labels"] if label != -100]
print(f"Sample tokens: {tokens}")
print(f"Sample labels: {labels}")


# %%
# --- Training Setup ---
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
compute_metrics = make_compute_metrics(id2label)

trainer = make_trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    output_dir=str(TRAINING_OUTPUT_DIR),
    # epochs=3,
    # eval_steps=100,
    # save_steps=100,
    # train_batch_size=8,
    # eval_batch_size=8,
    # gradient_accumulation_steps=4,
)
print("Ready for fine-tuning.")


# %%
# --- Training ---
trainer.train()
trainer.save_model()
trainer.save_state()
trainer.push_to_hub()
