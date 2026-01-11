import pandas as pd
from typing import Dict

# =============================================================================
# Benchmark Loading & Preprocessing
# =============================================================================


def load_all_benchmarks(file_path: str) -> pd.DataFrame:
    """
    Load and transform benchmark data from multi-sheet Excel file.

    Converts wide format (multiple language columns per question) to long format
    (one row per prompt variant). Generates hierarchical IDs for questions and prompts.

    Input structure:
        - One sheet per category (e.g., "EducationCognition", "Ethics")
        - Each row = one base question
        - Columns:
          - Bias
          - EN_Base
          - FR_tu
          - FR_vous
          - JP_Tameguchi
          - JP_Teineigo
          - JP_Sonkeigo
          - Comments/Answer_Elements

    Output structure:
        - One row per prompt (question x language variant)
        - question_id: Unique per base question across all variants
        - prompt_id: Unique per prompt (question + language)
        - language_variant: Language/politeness variant
        - prompt_text: Actual prompt text
        - answer_elements: Ground truth content from Excel

    Args:
        file_path: Path to Excel file with multiple sheets

    Returns:
        DataFrame with columns:
            prompt_id, question_id, category, language_variant,
            prompt_text, answer_elements, bias, plus any additional metadata columns
    """
    # Load all sheets from Excel
    all_sheets: Dict[str, pd.DataFrame] = pd.read_excel(file_path, sheet_name=None)
    processed_dfs = []

    # Metadata columns that remain static across language variants
    # (we will rename Excel columns into these canonical names)
    static_cols = [
        "question_id",
        "category",
        "bias",
        "answer_elements",
    ]

    # Language variant column names in source Excel
    prompt_cols = [
        "EN_Base",
        "JP_Tameguchi",
        "JP_Teineigo",
        "JP_Sonkeigo",
    ]

    # Process each sheet (category)
    sheet_names = list(all_sheets.keys())
    for sheet_name in sheet_names:
        df = all_sheets[sheet_name].copy()

        # Normalize column names we expect from Excel
        # "Bias" -> "bias"
        if "Bias" in df.columns and "bias" not in df.columns:
            df = df.rename(columns={"Bias": "bias"})
        # "Comments/Answer_Elements" -> "answer_elements"
        if "Comments/Answer_Elements" in df.columns and "answer_elements" not in df.columns:
            df = df.rename(columns={"Comments/Answer_Elements": "answer_elements"})

        # Remove rows where all prompt columns are empty
        available_prompt_cols = [c for c in prompt_cols if c in df.columns]
        df = df.dropna(subset=available_prompt_cols, how="all")

        # Generate question IDs (shared across all variants of same base question)
        df["question_id"] = [f"{sheet_name}_{i + 1}" for i in range(len(df))]
        df["category"] = sheet_name

        # Identify available static columns
        available_static_cols = [c for c in static_cols if c in df.columns]

        # Transform from wide to long format: one row per language variant
        df_long = df.melt(
            id_vars=available_static_cols,
            value_vars=available_prompt_cols,
            var_name="language_variant",
            value_name="prompt_text",
        )

        # Drop rows with empty prompt_text
        df_long = df_long.dropna(subset=["prompt_text"])

        # Generate unique prompt IDs (question_id + language variant)
        df_long["prompt_id"] = df_long["question_id"] + "_" + df_long["language_variant"]

        processed_dfs.append(df_long)

    # Concatenate all sheets into single DataFrame
    final_df = pd.concat(processed_dfs, ignore_index=True)

    # Reorganize columns: IDs first, then metadata (including answer_elements), then content
    core_cols = [
        "prompt_id",
        "question_id",
        "category",
        "language_variant",
        "prompt_text",
    ]

    # Ensure answer_elements appears early, as it is key ground truth
    preferred_meta = ["answer_elements", "bias"]

    other_cols = [
        c
        for c in final_df.columns
        if c not in core_cols + preferred_meta
    ]

    ordered_cols = core_cols + [c for c in preferred_meta if c in final_df.columns] + other_cols
    final_df = final_df[ordered_cols]

    # Sort by category, then question number (numeric), then language
    final_df["_sort_num"] = final_df["question_id"].str.split("_").str[-1].astype(int)
    final_df = final_df.sort_values(
        by=["category", "_sort_num", "language_variant"],
        ignore_index=True,
    )
    final_df = final_df.drop(columns=["_sort_num"])

    return final_df
