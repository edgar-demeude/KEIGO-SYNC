"""
Metrics runner utilities.

This module exposes:
    - compute_metrics_batches(df_responses: pd.DataFrame) -> pd.DataFrame

It also provides a main() that computes metrics for all models
listed in config.GEN_MODELS, using per-model checkpoints.
"""

import json
from pathlib import Path

import pandas as pd

import metrics_calculators
import config


def compute_metrics_batches(df_responses: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metrics for response batches.

    Groups responses by (question_id, model, num_batch) and computes metrics
    for each group together (enables relative comparisons across language variants).

    Requires 'response_embedding' to be present in df_responses.
    """
    all_scored_groups = []
    groups = df_responses.groupby(["question_id", "model", "num_batch"])

    print(f"Computing metrics for {len(groups)} batches...")

    for _, group in groups:
        scored_group = metrics_calculators.compute_metrics_for_batch(group)
        all_scored_groups.append(scored_group)

    return pd.concat(all_scored_groups, ignore_index=True)


def main() -> None:
    """
    Standalone entry point to compute metrics for existing JSON files.

    For each model in config.GEN_MODELS:
      - reads ../data/<model>_x<GEN_NB_ITER>_embeddings.json
      - writes ../data/<model>_x<GEN_NB_ITER>_metrics.json
    """
    models_list = config.GEN_MODELS
    nb_iter = config.GEN_NB_ITER

    for model_name in models_list:
        base_name = f"{model_name}_x{nb_iter}"
        input_path = Path(f"../data/{base_name}_embeddings.json")
        output_path = Path(f"../data/{base_name}_final.json")

        print("\n===============================================")
        print(f"[METRICS] Model: {model_name} ({base_name})")
        print("===============================================")

        if not input_path.exists():
            print(f"[WARN] Input JSON not found, skipping: {input_path}")
            continue

        print(f"[LOADING] Reading input JSON: {input_path}")
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Input JSON must be a list of objects")

        df = pd.DataFrame(data)

        required_cols = [
            "question_id",
            "model",
            "num_batch",
            "response_text",
            "language_variant",
            "response_embedding",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in input JSON: {missing}")

        print(f"[INFO] Loaded {len(df)} responses")
        print("[INFO] Computing metrics...")

        df_with_metrics = compute_metrics_batches(df)

        print("[INFO] Metrics computed, writing output JSON...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_with_metrics.to_json(
            output_path,
            orient="records",
            force_ascii=False,
            indent=2,
        )

        print(f"[DONE] Saved JSON with metrics to: {output_path}")


if __name__ == "__main__":
    main()
