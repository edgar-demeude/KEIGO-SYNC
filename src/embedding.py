"""
Embedding utilities.

This module exposes:
    - compute_embeddings_batch(df_input: pd.DataFrame) -> pd.DataFrame

It also provides a main() that recomputes embeddings for all models
listed in config.GEN_MODELS, using per-model checkpoints.
"""

import json
from pathlib import Path

import pandas as pd

import llm_clients
import config


def compute_embeddings_batch(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Compute semantic embeddings for all responses.

    Uses the embedding API to convert text responses to vector representations,
    used later for cosine similarity and semantic analysis.

    Args:
        df_input: DataFrame with response text in 'response_text' column.

    Returns:
        A copy of df_input with an added 'response_embedding' column (list of floats).
    """
    df_output = df_input.copy()
    df_output["response_embedding"] = df_output["response_text"].apply(
        llm_clients.call_embedding_model
    )
    return df_output


def main() -> None:
    """
    Standalone entry point to compute embeddings for existing JSON files.

    For each model in config.GEN_MODELS:
      - reads ../data/<model>_x<GEN_NB_ITER>_responses.json
      - writes ../data/<model>_x<GEN_NB_ITER>_embeddings.json
    """
    models_list = config.GEN_MODELS
    nb_iter = config.GEN_NB_ITER

    for model_name in models_list:
        base_name = f"{model_name}_x{nb_iter}"
        input_path = Path(f"../data/{base_name}_responses.json")
        output_path = Path(f"../data/{base_name}_embeddings.json")

        print("\n===============================================")
        print(f"[EMBEDDING] Model: {model_name} ({base_name})")
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
        if "response_text" not in df.columns:
            raise ValueError("Input JSON objects must contain a 'response_text' field")

        print(f"[INFO] Loaded {len(df)} responses")
        print("[INFO] Computing embeddings...")

        df_with_embeddings = compute_embeddings_batch(df)

        print("[INFO] Embeddings computed, writing output JSON...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_with_embeddings.to_json(
            output_path,
            orient="records",
            force_ascii=False,
            indent=2,
        )

        print(f"[DONE] Saved JSON with embeddings to: {output_path}")


if __name__ == "__main__":
    main()
