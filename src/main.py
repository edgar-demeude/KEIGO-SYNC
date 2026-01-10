import pandas as pd

import llm_clients
import loaders
import metrics_calculators

# =============================================================================
# Configuration & Setup
# =============================================================================

# List of models with temperature = 0.0 (zero-shot deterministic)
ZERO_TEMP_MODELS = {
    "mistral",
    "ministral-3b",
    "glm",
    "gemma",
    "qwen",
}

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
    models_list = "mistral"  # Single model or list: ["mistral", "ministral-3b", "gemma", "glm", "qwen"]

    # Determine iteration count based on model temperature
    active_models = {models_list} if isinstance(models_list, str) else set(models_list)
    nb_iter = 1 if (active_models & ZERO_TEMP_MODELS) else 3

    # Limit number of questions for testing (-1 = all)
    nb_questions = 1

    # Setup display
    configure_pandas_display()

    # =========== Execution Logs ===========
    print("\n========== BENCHMARK LOADING ==========")
    print(f"Total rows: {len(benchmark_prompts)}")
    print(benchmark_prompts.head(2))

    print("\n========== EXECUTION PARAMETERS ==========")
    print(f"Models              : {models_list}")
    print(f"Iterations          : {nb_iter}")
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
    print(responses_with_embeddings["reponse_emb"].head(2).apply(len))

    # Phase 3: Compute quality metrics
    print("\n========== METRICS CALCULATION ==========")
    responses_with_metrics = process_metrics_batches(responses_with_embeddings)

    print("\n---------- Responses with Metrics (Preview) ----------")
    print(responses_with_metrics.head(2))

    # Phase 4: Export results
    output_path = f"../data/{models_list}_x{nb_iter}.json"
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
        DataFrame with columns: ID_reponse, ID_Prompt_initial, ID_Question,
        num_batch, Categorie, langue_variante, model, question_txt, reponse_txt
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
            responses = llm_clients.call_llm(row["Prompt_Texte"], models=models)

            for model_name, text in responses.items():
                response_id = f"{row['ID_Prompt']}_{model_name}_{batch_id}"

                print(
                    f"[iter {batch_id}/{nb_iter}] "
                    f"question {q_idx}/{total_questions} "
                    f"model={model_name} "
                    f"ID_Prompt={row['ID_Prompt']} -> OK"
                )

                results.append({
                    "ID_reponse": response_id,
                    "ID_Prompt_initial": row["ID_Prompt"],
                    "ID_Question": row["ID_Question"],
                    "num_batch": batch_id,
                    "Categorie": row["Categorie"],
                    "langue_variante": row["Langue_Variante"],
                    "model": model_name,
                    "question_txt": row["Prompt_Texte"],
                    "reponse_txt": text,
                })

    return pd.DataFrame(results)


def compute_embeddings_batch(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Compute semantic embeddings for all responses.
    
    Uses Gemini Embedding API to convert text responses to vector representations.
    Used for cosine similarity and semantic analysis.
    
    Args:
        df_input: DataFrame with response text in 'reponse_txt' column
    
    Returns:
        Same DataFrame with added 'reponse_emb' column (list of floats)
    """
    df_input["reponse_emb"] = df_input["reponse_txt"].apply(
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
    groups = df_responses.groupby(["ID_Question", "model", "num_batch"])
    print(f"Computing metrics for {len(groups)} batches...")

    for _, group in groups:
        scored_group = metrics_calculators.compute_metrics_for_batch(group)
        all_scored_groups.append(scored_group)

    return pd.concat(all_scored_groups, ignore_index=True)


if __name__ == "__main__":
    main()
