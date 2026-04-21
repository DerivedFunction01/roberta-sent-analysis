from __future__ import annotations

from paths import path


DATASET_NAME = "OpenSafetyLab/Salad-Data"
SUBSET = "base_set"
TEXT_COLUMN = "question"
LABEL_COLUMN = "1-category"
MAX_SENTENCES = 3
MIN_LATIN_RATIO = 0.5
CACHE_DIR = path("salad", "salad_cache_dir")
NEUTRAL_DATASET_NAME = "teknium/OpenHermes-2.5"
NEUTRAL_SPLIT = "train"
NEUTRAL_TEXT_COLUMN = "conversations"
NEUTRAL_SAMPLE_FRACTION = 0.30
NEUTRAL_MAX_SENTENCES = 3
NEUTRAL_MIN_LATIN_RATIO = 0.5
NEUTRAL_CACHE_DIR = path("salad", "salad_outside_cache_dir")
JAILBREAK_DATASET_NAME = "jackhhao/jailbreak-classification"
JAILBREAK_SPLIT = "train"
JAILBREAK_PROMPT_COLUMN = "prompt"
JAILBREAK_LABEL_COLUMN = "type"
JAILBREAK_TARGET_LABEL = "jailbreak"
JAILBREAK_MAX_SENTENCES = 3
JAILBREAK_CACHE_DIR = path("salad", "salad_jailbreak_cache_dir")
JAILBREAK_BENIGN_LABEL = "outside"
JAILBREAK_BENIGN_CACHE_DIR = path("salad", "salad_jailbreak_benign_cache_dir")
JAILBREAK_FILTER_DIR = path("salad", "salad_jailbreak_filter_dir")
JAILBREAK_FILTER_META_FILE = path("salad", "salad_jailbreak_filter_meta_file")
JAILBREAK_FILTER_MODEL_FILE = JAILBREAK_FILTER_DIR / "model.joblib"
JAILBREAK_FILTER_THRESHOLD = 0.5
TOKENIZED_DATASET_DIR = path("salad", "salad_tokenized_dataset_dir")
TOKENIZED_DATASET_META = path("salad", "salad_tokenized_dataset_meta")
PIPELINE_RESULTS_DIR = path("salad", "salad_pipeline_results_dir")

MAX_LENGTH = 512
STANDALONE_RATIO = 0.5
SAME_CLASS_RATIO = 0.25
MIXED_CLASS_RATIO = 0.25
BALANCED_COVERAGE_RATIO = 0.25
REUSE_LIMIT = 5
TRAIN_EXAMPLES =   100_000
VALIDATION_EXAMPLES = 2_000
TEST_EXAMPLES = 2_000
USE_SALAD_MUTATION = True
CONTEXTUAL_PROBABILITY = 0.5
CONTEXTUAL_MIN_SEGMENTS = 2
CONTEXTUAL_MAX_SEGMENTS = 5
