import json
from pathlib import Path

import pandas as pd

import llm_clients
import loaders
from embedding import compute_embeddings_batch
from metrics_runner import compute_metrics_batches
import config

# =============================================================================
# Configuration & Setup
# =============================================================================


def configure_pandas_display() -> None:
    """Configure pandas output for wide displays."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.width", 2000)
    pd.set_option("display.colheader_justify", "left")


def save_json_checkpoint(df: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame to JSON (records) as a checkpoint.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_json(
        out_path,
        orient="records",
        force_ascii=False,
        indent=2,
    )

    print(f"[CHECKPOINT] Saved {len(df)} rows to {out_path}")


# =============================================================================
# Main Pipeline
# =============================================================================


def main(
    compute_embeddings: bool = False,
    compute_metrics: bool = False,
) -> None:
    """
    Complete sycophancy evaluation pipeline with JSON checkpoints.

    Runs the full pipeline per model in config.GEN_MODELS and
    writes per-model checkpoints and final JSONs.
    """
    # -------------------------------------------------------------------------
    # Global configuration (from config.py)
    # -------------------------------------------------------------------------
    benchmark_prompts = loaders.load_all_benchmarks(config.BENCHMARK_EXCEL_PATH)
    models_list = config.GEN_MODELS
    nb_iter = config.GEN_NB_ITER
    nb_questions = config.GEN_NB_QUESTIONS

    configure_pandas_display()

    print("\n========== BENCHMARK LOADING ==========")
    print(f"Total rows: {len(benchmark_prompts)}")
    print(benchmark_prompts.head(2))

    print("\n========== EXECUTION PARAMETERS ==========")
    print(f"Models : {models_list}")
    print(f"Iterations : {nb_iter}")
    print(f"Questions to process: {nb_questions}")
    print(f"compute_embeddings = {compute_embeddings}")
    print(f"compute_metrics    = {compute_metrics}")

    # -------------------------------------------------------------------------
    # Run full pipeline per model (per-model checkpoints & outputs)
    # -------------------------------------------------------------------------
    for model_name in models_list:
        base_name = f"{model_name}_x{nb_iter}"
        responses_ckpt_path = f"../data/{base_name}_responses.json"
        embeddings_ckpt_path = f"../data/{base_name}_embeddings.json"
        final_ckpt_path = f"../data/{base_name}_final.json"

        print("\n==================================================")
        print(f"=== PIPELINE FOR MODEL: {model_name} ({base_name}) ===")
        print("==================================================")

        run_pipeline_for_model(
            benchmark_prompts=benchmark_prompts,
            model_name=model_name,
            nb_iter=nb_iter,
            nb_questions=nb_questions,
            responses_ckpt_path=responses_ckpt_path,
            embeddings_ckpt_path=embeddings_ckpt_path,
            metrics_ckpt_path=final_ckpt_path,
            compute_embeddings=compute_embeddings,
            compute_metrics=compute_metrics,
        )


def run_pipeline_for_model(
    benchmark_prompts: pd.DataFrame,
    model_name: str,
    nb_iter: int,
    nb_questions: int,
    responses_ckpt_path: str,
    embeddings_ckpt_path: str,
    metrics_ckpt_path: str,
    compute_embeddings: bool,
    compute_metrics: bool,
) -> None:
    """
    Run the full pipeline for a single model, using per-model checkpoints.
    """
    # -------------------------------------------------------------------------
    # Phase 1: LLM responses
    # -------------------------------------------------------------------------
    print("\n========== LLM INFERENCE ==========")

    if Path(responses_ckpt_path).exists():
        print(f"[INFO] Loading existing responses checkpoint: {responses_ckpt_path}")
        responses = pd.read_json(responses_ckpt_path, orient="records")
    else:
        responses = process_benchmark_batch(
            benchmark_prompts,
            models=model_name,
            nb_iter=nb_iter,
            nb_questions=nb_questions,
        )
        save_json_checkpoint(responses, responses_ckpt_path)

    print("\n---------- LLM Responses (Preview) ----------")
    print(responses.head(2))

    # -------------------------------------------------------------------------
    # Phase 2: Embeddings (optional)
    # -------------------------------------------------------------------------
    print("\n========== EMBEDDING COMPUTATION ==========")

    responses_with_embeddings = None

    if not compute_embeddings:
        print("[INFO] Skipping embedding computation (compute_embeddings=False)")
        if Path(embeddings_ckpt_path).exists():
            print(f"[INFO] Loading existing embeddings checkpoint: {embeddings_ckpt_path}")
            responses_with_embeddings = pd.read_json(
                embeddings_ckpt_path,
                orient="records",
            )
        else:
            print(
                "[WARN] No embeddings checkpoint found and compute_embeddings=False. "
                "You will not be able to compute metrics that require embeddings "
                "until you generate them (e.g., via embedding.py)."
            )
    else:
        if Path(embeddings_ckpt_path).exists():
            print(f"[INFO] Loading existing embeddings checkpoint: {embeddings_ckpt_path}")
            responses_with_embeddings = pd.read_json(
                embeddings_ckpt_path,
                orient="records",
            )
        else:
            responses_with_embeddings = compute_embeddings_batch(responses)
            save_json_checkpoint(responses_with_embeddings, embeddings_ckpt_path)

    if responses_with_embeddings is not None:
        current_df = responses_with_embeddings
    else:
        current_df = responses

    if "response_embedding" in current_df.columns:
        print("\n---------- Embedding Sizes (Preview) ----------")
        print(current_df["response_embedding"].head(2).apply(len))
        embeddings_available = True
    else:
        print("[INFO] No embeddings available in this run.")
        embeddings_available = False

    # -------------------------------------------------------------------------
    # Phase 3: Metrics (optional)
    # -------------------------------------------------------------------------
    print("\n========== METRICS CALCULATION ==========")

    if not embeddings_available:
        print(
            "[WARN] Cannot compute metrics because 'response_embedding' "
            "is not available. Generate embeddings first."
        )
        df_with_metrics = current_df
    elif not compute_metrics:
        print("[INFO] Skipping metrics computation (compute_metrics=False)")
        if Path(metrics_ckpt_path).exists():
            print(f"[INFO] Loading existing metrics checkpoint: {metrics_ckpt_path}")
            df_with_metrics = pd.read_json(
                metrics_ckpt_path,
                orient="records",
            )
        else:
            print(
                "[WARN] No metrics checkpoint found and compute_metrics=False. "
                "Downstream steps that require metrics may be incomplete."
            )
            df_with_metrics = current_df
    else:
        if Path(metrics_ckpt_path).exists():
            print(f"[INFO] Loading existing metrics checkpoint: {metrics_ckpt_path}")
            df_with_metrics = pd.read_json(
                metrics_ckpt_path,
                orient="records",
            )
        else:
            df_with_metrics = compute_metrics_batches(current_df)
            save_json_checkpoint(df_with_metrics, metrics_ckpt_path)

        print("\n---------- Responses with Metrics (Preview) ----------")
        print(df_with_metrics.head(2))

    # -------------------------------------------------------------------------
    # Phase 4: Final export (per model)
    # -------------------------------------------------------------------------
    df_with_metrics = reorder_columns_for_json(df_with_metrics)

    final_output_path = f"../data/{model_name}_x{nb_iter}.json"
    df_with_metrics.to_json(
        final_output_path,
        orient="records",
        force_ascii=False,
        indent=2,
    )

    print("\n========== EXPORT COMPLETE ==========")
    print(f"Output saved for {model_name}: {final_output_path}")


# =============================================================================
# Pipeline Functions
# =============================================================================


def process_benchmark_batch(
    df_input: pd.DataFrame,
    models: str = "all",
    nb_iter: int = 1,
    nb_questions: int = -1,
) -> pd.DataFrame:
    """
    Call LLM models for each benchmark question across iterations.
    """
    if nb_questions > 0:
        df_input = df_input.head(nb_questions)

    total_questions = len(df_input)
    print(f"\n[INFO] Questions to process: {total_questions} (nb_iter = {nb_iter})")

    results = []

    for iter_idx in range(nb_iter):
        batch_id = iter_idx + 1
        print(f"\n[ITERATION {batch_id}/{nb_iter}] Starting batch")

        for q_idx, (_, row) in enumerate(df_input.iterrows(), start=1):
            responses = llm_clients.call_llm(row["prompt_text"], models=models)

            for model_name, text in responses.items():
                response_id = f"{row['prompt_id']}_{model_name}_{batch_id}"
                print(
                    f"[iter {batch_id}/{nb_iter}] "
                    f"question {q_idx}/{total_questions} "
                    f"model={model_name} "
                    f"prompt_id={row['prompt_id']} -> OK"
                )

                results.append(
                    {
                        "response_id": response_id,
                        "initial_prompt_id": row["prompt_id"],
                        "question_id": row["question_id"],
                        "num_batch": batch_id,
                        "category": row["category"],
                        "language_variant": row["language_variant"],
                        "model": model_name,
                        "question_text": row["prompt_text"],
                        "response_text": text,
                        "answer_elements": row.get("answer_elements", ""),
                        "bias": row.get("bias", ""),
                    }
                )

    return pd.DataFrame(results)


def reorder_columns_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder DataFrame columns for clean JSON export.
    """
    core_cols = [
        "response_id",
        "initial_prompt_id",
        "question_id",
        "num_batch",
        "category",
        "language_variant",
        "model",
        "question_text",
        "response_text",
    ]

    meta_cols = [
        "answer_elements",
        "bias",
    ]

    metric_cols = [
        "char_count",
        "num_sentences",
        "avg_sentence_len",
        "formality_ratio",
        "cosine_similarity",
        "refusal_rate",
    ]

    embedding_col = ["response_embedding"]

    available_core = [c for c in core_cols if c in df.columns]
    available_meta = [c for c in meta_cols if c in df.columns]
    available_metrics = [c for c in metric_cols if c in df.columns]
    available_embedding = [c for c in embedding_col if c in df.columns]

    other_cols = [
        c
        for c in df.columns
        if c not in available_core + available_meta + available_metrics + available_embedding
    ]

    final_order = (
        available_core
        + available_meta
        + available_metrics
        + other_cols
        + available_embedding
    )
    return df[final_order]


if __name__ == "__main__":
    main(compute_embeddings=config.COMPUTE_EMBEDDINGS, compute_metrics=config.COMPUTE_METRICS)
