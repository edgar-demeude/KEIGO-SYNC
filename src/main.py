import pandas as pd

import llm_clients
import loaders
import metrics_calculators

# =============================================================================
# Configuration & Setup
# =============================================================================

# Pandas display options for better output readability
def configure_pandas_display():
    """Configure pandas output for wide displays."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.width", 2000)
    pd.set_option("display.colheader_justify", "left")

# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """
    Complete sycophancy evaluation pipeline:
    1. Load benchmark questions (multiple language variants)
    2. Call LLM models to generate responses
    3. Compute embeddings for semantic similarity
    4. Calculate quality metrics (text, formality, judge scores)
    5. Export results as JSON
    """
    # Load benchmark data
    benchmark_prompts = loaders.load_all_benchmarks("../data/Benchmark_Questions.xlsx")

    # Select model(s) to evaluate
    models_list = "glm"  # ou liste: ["mistral", "ministral-3b", "gemma", "glm", "qwen"]

    # FIXED number of iterations for all models
    nb_iter = 3

    # Limit number of questions for testing (-1 = all)
    nb_questions = -1

    # Setup display
    configure_pandas_display()

    # =========== Execution Logs ===========
    print("\n========== BENCHMARK LOADING ==========")
    print(f"Total rows: {len(benchmark_prompts)}")
    print(benchmark_prompts.head(2))

    print("\n========== EXECUTION PARAMETERS ==========")
    print(f"Models : {models_list}")
    print(f"Iterations : {nb_iter}")
    print(f"Questions to process: {nb_questions}")

    # Phase 1: Generate LLM responses
    print("\n========== LLM INFERENCE ==========")
    responses = process_benchmark_batch(
        benchmark_prompts,
        models=models_list,
        nb_iter=nb_iter,
        nb_questions=nb_questions,
    )

    print("\n---------- LLM Responses (Preview) ----------")
    print(responses.head(2))

    # Phase 2: Compute semantic embeddings
    print("\n========== EMBEDDING COMPUTATION ==========")
    responses_with_embeddings = compute_embeddings_batch(responses)
    print("\n---------- Embedding Sizes (Preview) ----------")
    print(responses_with_embeddings["response_embedding"].head(2).apply(len))

    # Phase 3: Compute quality metrics
    print("\n========== METRICS CALCULATION ==========")
    responses_with_metrics = process_metrics_batches(responses_with_embeddings)
    print("\n---------- Responses with Metrics (Preview) ----------")
    print(responses_with_metrics.head(2))

    # Phase 4: Export results
    output_path = f"../data/{models_list}_x{nb_iter}.json"

    # Reorder columns for cleaner JSON structure
    responses_with_metrics = reorder_columns_for_json(responses_with_metrics)

    responses_with_metrics.to_json(
        output_path,
        orient="records",
        force_ascii=False,
        indent=2,
    )

    print("\n========== EXPORT COMPLETE ==========")
    print(f"Output saved: {output_path}")


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

    For each iteration, call every model on each question variant,
    collecting responses with metadata (question ID, language, model name).

    Args:
        df_input: DataFrame with prompt variants (from loaders)
        models: Single model name (str) or list of models
        nb_iter: Number of iterations (deterministic models = 1)
        nb_questions: Limit to first N questions (-1 = all)

    Returns:
        DataFrame with columns: response_id, initial_prompt_id, question_id,
        num_batch, category, language_variant, model, question_text, response_text
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

                results.append({
                    "response_id": response_id,
                    "initial_prompt_id": row["prompt_id"],
                    "question_id": row["question_id"],
                    "num_batch": batch_id,
                    "category": row["category"],
                    "language_variant": row["language_variant"],
                    "model": model_name,
                    "question_text": row["prompt_text"],
                    "response_text": text,
                })

    return pd.DataFrame(results)


def compute_embeddings_batch(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Compute semantic embeddings for all responses.

    Uses Gemini Embedding API to convert text responses to vector representations.
    Used for cosine similarity and semantic analysis.

    Args:
        df_input: DataFrame with response text in 'response_text' column

    Returns:
        Same DataFrame with added 'response_embedding' column (list of floats)
    """
    df_input["response_embedding"] = df_input["response_text"].apply(
        llm_clients.call_embedding_model
    )
    return df_input


def process_metrics_batches(df_responses: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metrics for response batches.

    Groups responses by (question, model, iteration) and computes metrics
    for each group together (enables relative comparisons across language variants).

    Metrics computed:
    - Text properties: char count, sentence count, avg sentence length
    - Formality: language-specific politeness markers
    - Semantic: cosine similarity to English reference
    - Judgment: LLM-as-a-judge sycophancy evaluation

    Args:
        df_responses: DataFrame with responses and embeddings

    Returns:
        Same DataFrame with added metric columns
    """
    all_scored_groups = []
    groups = df_responses.groupby(["question_id", "model", "num_batch"])
    print(f"Computing metrics for {len(groups)} batches...")

    for _, group in groups:
        scored_group = metrics_calculators.compute_metrics_for_batch(group)
        all_scored_groups.append(scored_group)

    return pd.concat(all_scored_groups, ignore_index=True)


def reorder_columns_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder DataFrame columns for clean JSON export.

    Structure:
    - Core IDs: response_id, initial_prompt_id, question_id
    - Metadata: num_batch, category, language_variant, model
    - Content: question_text, response_text
    - Metrics: All computed metrics (char_count, formality_ratio, etc.)
    - Embeddings: response_embedding (at the very end)

    Args:
        df: DataFrame with all columns

    Returns:
        Same DataFrame with reordered columns
    """
    # Core columns in desired order
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

    # Metrics columns (numeric and analysis-related)
    metric_cols = [
        "char_count",
        "num_sentences",
        "avg_sentence_len",
        "formality_ratio",
        "cosine_similarity",
        "refusal_rate",
    ]

    # Embedding column (always at the end)
    embedding_col = ["response_embedding"]

    # Get all columns that exist in the DataFrame
    available_core = [c for c in core_cols if c in df.columns]
    available_metrics = [c for c in metric_cols if c in df.columns]
    available_embedding = [c for c in embedding_col if c in df.columns]

    # Any other columns not explicitly listed (in case there are extras)
    other_cols = [
        c
        for c in df.columns
        if c not in available_core + available_metrics + available_embedding
    ]

    # Final column order
    final_order = available_core + available_metrics + other_cols + available_embedding

    return df[final_order]


if __name__ == "__main__":
    main()
