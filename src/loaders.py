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
        - Columns: JP_Tameguchi, JP_Teineigo, JP_Sonkeigo, EN_Base (language variants)
    
    Output structure:
        - One row per prompt (question * language variant)
        - ID_Question: Unique per base question across all variants
        - ID_Prompt: Unique per prompt (question + language)
        - Langue_Variante: Language/politeness variant
        - Prompt_Texte: Actual prompt text
    
    Args:
        file_path: Path to Excel file with multiple sheets
    
    Returns:
        DataFrame with columns: ID_Prompt, ID_Question, Categorie, Langue_Variante, 
        Prompt_Texte, plus any additional metadata columns
    """
    # Load all sheets from Excel
    all_sheets = pd.read_excel(file_path, sheet_name=None)
    processed_dfs = []

    # Metadata columns that remain static across language variants
    static_cols = [
        "ID_Question",
        "Categorie",
        "Biais",
        "Comments/Answer_Elements",
    ]

    # Language variant column names in source Excel
    prompt_cols = [
        "JP_Tameguchi",
        "JP_Teineigo",
        "JP_Sonkeigo",
        "EN_Base",
    ]

    # Process each sheet (category)
    sheet_names = list(all_sheets.keys())

    for sheet_name in sheet_names:
        df = all_sheets[sheet_name].copy()

        # Remove rows where all prompt columns are empty
        available_prompt_cols = [c for c in prompt_cols if c in df.columns]
        df = df.dropna(subset=available_prompt_cols, how="all")

        # Generate question IDs (shared across all variants of same base question)
        df["ID_Question"] = [f"{sheet_name}_{i + 1}" for i in range(len(df))]
        df["Categorie"] = sheet_name

        # Identify available static columns
        available_static_cols = [c for c in static_cols if c in df.columns]

        # Transform from wide to long format
        # Each row becomes N rows (one per language variant)
        df_long = df.melt(
            id_vars=available_static_cols,
            value_vars=available_prompt_cols,
            var_name="Langue_Variante",
            value_name="Prompt_Texte",
        )

        # Generate unique prompt IDs (question_id + language variant)
        df_long["ID_Prompt"] = (
            df_long["ID_Question"] + "_" + df_long["Langue_Variante"]
        )

        processed_dfs.append(df_long)

    # Concatenate all sheets into single DataFrame
    final_df = pd.concat(processed_dfs, ignore_index=True)

    # Reorganize columns: IDs first, then metadata, then content
    core_cols = ["ID_Prompt", "ID_Question", "Categorie", "Langue_Variante", "Prompt_Texte"]
    other_cols = [c for c in final_df.columns if c not in core_cols]
    final_df = final_df[core_cols + other_cols]

    # Sort by category, then question number (numeric), then language
    # Extract numeric question ID for proper numeric sorting (e.g., 2 before 10)
    final_df["_sort_num"] = (
        final_df["ID_Question"].str.split("_").str[-1].astype(int)
    )
    final_df = final_df.sort_values(
        by=["Categorie", "_sort_num", "Langue_Variante"], ignore_index=True
    )
    final_df = final_df.drop(columns=["_sort_num"])

    return final_df
