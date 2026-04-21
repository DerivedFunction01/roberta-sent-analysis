from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

LABEL_MAP_FILE = PROJECT_ROOT / "label2id.json"
TOKENIZED_DATASET_DIR = PROJECT_ROOT / "tokenized_dataset"
TOKENIZED_DATASET_META = TOKENIZED_DATASET_DIR / "cache_meta.json"

RESULTS_DIR = PROJECT_ROOT / "results"
STANDALONE_RESULTS_DIR = RESULTS_DIR / "tweet_eval_standalone"

PIPELINE_RESULTS_DIR = RESULTS_DIR / "tweet_eval_pipeline"

DEFAULT_CONFIG_FILE = PROJECT_ROOT / "evaluation_config.json"
