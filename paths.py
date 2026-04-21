from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

PATHS = {
    "root": {
        "label_map_file": PROJECT_ROOT / "label2id.json",
        "results_dir": PROJECT_ROOT / "results",
        "default_config_file": PROJECT_ROOT / "evaluation_config.json",
    },
    "tweet": {
        "sentiment_cache_dir": PROJECT_ROOT / "sentiment_cache",
        "sentiment_cache_meta_file": PROJECT_ROOT / "sentiment_cache" / "meta.json",
        "sentiment_cache_pos_file": PROJECT_ROOT / "sentiment_cache" / "pos.parquet",
        "sentiment_cache_neg_file": PROJECT_ROOT / "sentiment_cache" / "neg.parquet",
        "sentiment_cache_neu_file": PROJECT_ROOT / "sentiment_cache" / "neu.parquet",
        "tokenized_dataset_dir": PROJECT_ROOT / "tokenized_dataset",
        "tokenized_dataset_meta": PROJECT_ROOT / "tokenized_dataset" / "cache_meta.json",
        "standalone_results_dir": PROJECT_ROOT / "results" / "tweet_eval_standalone",
        "pipeline_results_dir": PROJECT_ROOT / "results" / "tweet_eval_pipeline",
    },
    "salad": {
        "salad_cache_dir": PROJECT_ROOT / "salad_cache",
        "salad_cache_meta_file": PROJECT_ROOT / "salad_cache" / "meta.json",
        "salad_outside_cache_dir": PROJECT_ROOT / "salad_outside_cache",
        "salad_outside_cache_meta_file": PROJECT_ROOT / "salad_outside_cache" / "meta.json",
        "salad_jailbreak_cache_dir": PROJECT_ROOT / "salad_jailbreak_cache",
        "salad_jailbreak_cache_meta_file": PROJECT_ROOT / "salad_jailbreak_cache" / "meta.json",
        "salad_jailbreak_benign_cache_dir": PROJECT_ROOT / "salad_jailbreak_benign_cache",
        "salad_jailbreak_benign_cache_meta_file": PROJECT_ROOT / "salad_jailbreak_benign_cache" / "meta.json",
        "salad_jailbreak_filter_dir": PROJECT_ROOT / "results" / "salad_jailbreak_filter",
        "salad_jailbreak_filter_meta_file": PROJECT_ROOT / "results" / "salad_jailbreak_filter" / "meta.json",
        "salad_label_map_file": PROJECT_ROOT / "salad_label2id.json",
        "salad_tokenized_dataset_dir": PROJECT_ROOT / "salad_tokenized_dataset",
        "salad_tokenized_dataset_meta": PROJECT_ROOT / "salad_tokenized_dataset" / "cache_meta.json",
        "salad_pipeline_results_dir": PROJECT_ROOT / "results" / "salad_pipeline",
        "salad_results_dir": PROJECT_ROOT / "results" / "salad_data",
    },
}


def path(category: str, name: str) -> Path:
    try:
        value = PATHS[category][name]
    except KeyError as exc:
        raise KeyError(f"Unknown path key: {category}.{name}") from exc
    return Path(value)
