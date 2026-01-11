"""
judge_analysis.py

Utility module to inspect and analyze sycophancy judge results.

Main features:
- Load enriched benchmark JSON (output of judge_benchmark_json).
- Safely explode per-judge metrics into a flat table.
- Identify missing or invalid metrics (None / NaN) per judge and response.
- Compute summary statistics by category, language_variant, model, judge_model.
- Provide convenient views for manual debugging and re-running problematic cases.
"""

import json
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# =====================================================================
# I/O helpers
# =====================================================================


def load_benchmark(path: str) -> pd.DataFrame:
    """
    Load the enriched benchmark JSON (list of dicts) into a pandas DataFrame.

    Expected top-level keys per entry:
      - response_id, initial_prompt_id, question_id, num_batch,
        category, language_variant, model, question_text, response_text,
        char_count, num_sentences, avg_sentence_len, formality_ratio,
        cosine_similarity, judges, judges_average, response_embedding, ...
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Benchmark JSON must be a list of objects")

    df = pd.DataFrame.from_records(data)
    return df


# =====================================================================
# Flatten / explode judges
# =====================================================================


def safe_float(x: Any) -> float:
    """
    Convert a value to float, returning np.nan if the value is None or invalid.
    Avoids errors when metrics are None in JSON.
    """
    if x is None:
        return np.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def explode_judges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the 'judges' list into one row per judge per response.

    Output columns include:
      - response-level columns: response_id, question_id, category, language_variant,
        model, char_count, num_sentences, avg_sentence_len, formality_ratio,
        cosine_similarity, etc.
      - judge-level columns: judge_model, regressive, validation, framing, overall.
    """
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        judges = row.get("judges", [])
        if not isinstance(judges, list):
            continue

        base = {
            "response_id": row.get("response_id"),
            "initial_prompt_id": row.get("initial_prompt_id"),
            "question_id": row.get("question_id"),
            "num_batch": row.get("num_batch"),
            "category": row.get("category"),
            "language_variant": row.get("language_variant"),
            "model": row.get("model"),
            "char_count": row.get("char_count"),
            "num_sentences": row.get("num_sentences"),
            "avg_sentence_len": row.get("avg_sentence_len"),
            "formality_ratio": row.get("formality_ratio"),
            "cosine_similarity": row.get("cosine_similarity"),
        }

        for j in judges:
            if not isinstance(j, dict):
                continue
            rec = base.copy()
            rec["judge_model"] = j.get("judge_model")

            # Raw values from JSON (may be None)
            rec["regressive_raw"] = j.get("regressive")
            rec["validation_raw"] = j.get("validation")
            rec["framing_raw"] = j.get("framing")
            rec["overall_raw"] = j.get("overall")

            # Safe numeric casting
            rec["regressive"] = safe_float(j.get("regressive"))
            rec["validation"] = safe_float(j.get("validation"))
            rec["framing"] = safe_float(j.get("framing"))
            rec["overall"] = safe_float(j.get("overall"))

            # Keep commentary / parser_raw for debugging
            rec["commentary"] = j.get("commentary")
            rec["parser_raw"] = j.get("parser_raw")

            records.append(rec)

    if not records:
        return pd.DataFrame(
            columns=[
                "response_id",
                "judge_model",
                "regressive",
                "validation",
                "framing",
                "overall",
                "commentary",
                "parser_raw",
            ]
        )

    judges_df = pd.DataFrame.from_records(records)
    return judges_df


# =====================================================================
# Missing / invalid metrics inspection
# =====================================================================


def summarize_missing(judges_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Build several DataFrames summarizing where metrics are missing (NaN)
    or invalid, grouped by response, judge_model, and category.

    Returns a dict of named DataFrames:
      - "missing_any": rows where at least one metric is NaN
      - "missing_by_response": counts per response_id
      - "missing_by_judge_model": counts per judge_model
      - "missing_by_category": counts per (category, language_variant, model)
    """
    metric_cols = ["regressive", "validation", "framing", "overall"]

    # Flag rows with at least one missing metric
    judges_df["missing_any_metric"] = judges_df[metric_cols].isna().any(axis=1)

    missing_any = judges_df[judges_df["missing_any_metric"]].copy()

    # Counts by response_id
    missing_by_response = (
        missing_any.groupby("response_id")[metric_cols]
        .apply(lambda g: g.isna().sum())
        .reset_index()
    )

    # Counts by judge_model
    missing_by_judge_model = (
        missing_any.groupby("judge_model")[metric_cols]
        .apply(lambda g: g.isna().sum())
        .reset_index()
    )

    # Counts by (category, language_variant, model)
    missing_by_category = (
        missing_any.groupby(["category", "language_variant", "model"])[metric_cols]
        .apply(lambda g: g.isna().sum())
        .reset_index()
    )

    return {
        "missing_any": missing_any,
        "missing_by_response": missing_by_response,
        "missing_by_judge_model": missing_by_judge_model,
        "missing_by_category": missing_by_category,
    }


def list_problematic_entries(judges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a compact table listing all response_id / judge_model pairs
    where at least one metric is NaN, plus commentary and parser_raw.

    This is the table you can use to decide what to re-run manually.
    """
    metric_cols = ["regressive", "validation", "framing", "overall"]
    mask = judges_df[metric_cols].isna().any(axis=1)
    cols = [
        "response_id",
        "initial_prompt_id",
        "question_id",
        "category",
        "language_variant",
        "model",
        "judge_model",
        "regressive_raw",
        "validation_raw",
        "framing_raw",
        "overall_raw",
        "regressive",
        "validation",
        "framing",
        "overall",
        "commentary",
        "parser_raw",
    ]
    for c in cols:
        if c not in judges_df.columns:
            judges_df[c] = np.nan
    return judges_df.loc[mask, cols].sort_values(
        ["response_id", "judge_model"], ignore_index=True
    )


# =====================================================================
# Per-response averages
# =====================================================================


def compute_response_averages(judges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-response averages across judges, similar to judges_average
    but using the exploded table (so you can check for consistency).

    Output columns:
      - response_id, category, language_variant, model
      - avg_regressive, avg_validation, avg_framing, avg_overall
      - count_regressive, count_validation, count_framing, count_overall
    """
    metric_cols = ["regressive", "validation", "framing", "overall"]

    grouped = (
        judges_df.groupby(
            ["response_id", "category", "language_variant", "model"], dropna=False
        )[metric_cols]
        .agg(["mean", "count"])
        .reset_index()
    )

    rows = []
    for _, row in grouped.iterrows():
        rec = {
            "response_id": row["response_id"],
            "category": row["category"],
            "language_variant": row["language_variant"],
            "model": row["model"],
        }
        for metric in metric_cols:
            rec[f"avg_{metric}"] = row[(metric, "mean")]
            rec[f"count_{metric}"] = int(row[(metric, "count")])
        rows.append(rec)

    out = pd.DataFrame.from_records(rows)
    return out


# =====================================================================
# Aggregated statistics (groupby helpers)
# =====================================================================

AGG_METRICS = ["regressive", "validation", "framing", "overall"]


def _aggregate_generic(judges_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Generic helper: compute mean, std, count per metric for a given grouping.

    Example:
      _aggregate_generic(judges_df, ["category"])
      _aggregate_generic(judges_df, ["language_variant", "model"])
    """
    grouped = judges_df.groupby(group_cols)[AGG_METRICS].agg(["mean", "std", "count"])
    return grouped.sort_index()


def aggregate_by_model(judges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average regressive/validation/framing/overall per generation model.
    """
    return _aggregate_generic(judges_df, ["model"])


def aggregate_by_category(judges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average metrics per question category (theme).
    """
    return _aggregate_generic(judges_df, ["category"])


def aggregate_by_language_variant(judges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average metrics per language_variant (e.g. EN_Base, JP_Sonkeigo, ...).
    """
    return _aggregate_generic(judges_df, ["language_variant"])


def aggregate_by_model_and_category(judges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average metrics per (model, category).
    """
    return _aggregate_generic(judges_df, ["model", "category"])


def aggregate_by_model_and_language(judges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average metrics per (model, language_variant).
    """
    return _aggregate_generic(judges_df, ["model", "language_variant"])


def aggregate_by_category_and_language(judges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average metrics per (category, language_variant).
    """
    return _aggregate_generic(judges_df, ["category", "language_variant"])


def aggregate_by_judge_model(judges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average metrics per judge_model (to see judge bias / calibration).
    """
    return _aggregate_generic(judges_df, ["judge_model"])


def aggregate_by_model_category_language(judges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average metrics per (model, category, language_variant).
    This is the most granular aggregate at the response-group level.
    """
    return _aggregate_generic(judges_df, ["model", "category", "language_variant"])


# =====================================================================
# High-level analysis pipeline
# =====================================================================


def run_full_analysis(json_path: str) -> Dict[str, Any]:
    """
    Run the full analysis pipeline on the given enriched benchmark JSON.

    Returns a dict of DataFrames:
      - df: raw benchmark DataFrame
      - judges_df: exploded per-judge DataFrame
      - missing_summaries: dict of missing/invalid summaries
      - problematic_entries: table to inspect/re-run
      - response_averages: per-response averages
      - by_category, by_language_variant, by_model, by_judge_model, etc.
    """
    df = load_benchmark(json_path)
    judges_df = explode_judges(df)

    missing_summaries = summarize_missing(judges_df)
    problematic_entries = list_problematic_entries(judges_df)
    response_averages = compute_response_averages(judges_df)

    by_category = aggregate_by_category(judges_df)
    by_language_variant = aggregate_by_language_variant(judges_df)
    by_model = aggregate_by_model(judges_df)
    by_judge_model = aggregate_by_judge_model(judges_df)
    by_model_cat = aggregate_by_model_and_category(judges_df)
    by_model_lang = aggregate_by_model_and_language(judges_df)
    by_cat_lang = aggregate_by_category_and_language(judges_df)
    by_model_cat_lang = aggregate_by_model_category_language(judges_df)

    return {
        "df": df,
        "judges_df": judges_df,
        "missing_summaries": missing_summaries,
        "problematic_entries": problematic_entries,
        "response_averages": response_averages,
        "by_category": by_category,
        "by_language_variant": by_language_variant,
        "by_model": by_model,
        "by_judge_model": by_judge_model,
        "by_model_category": by_model_cat,
        "by_model_language": by_model_lang,
        "by_category_language": by_cat_lang,
        "by_model_category_language": by_model_cat_lang,
    }


# =====================================================================
# CLI usage example
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze sycophancy judge results (enriched benchmark JSON)."
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to enriched benchmark JSON (output of judge_benchmark_json).",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="If set, save key tables as CSV files with this prefix.",
    )

    args = parser.parse_args()

    results = run_full_analysis(args.json_path)

    judges_df = results["judges_df"]
    problematic_entries = results["problematic_entries"]

    print("\n[INFO] Total judge rows:", len(judges_df))
    print(
        "[INFO] Rows with at least one missing metric:",
        len(results["missing_summaries"]["missing_any"]),
    )
    print(
        "[INFO] Unique (response_id, judge_model) pairs with missing metrics:",
        problematic_entries[["response_id", "judge_model"]]
        .drop_duplicates()
        .shape[0],
    )

    if args.save_prefix:
        judges_df.to_csv(f"{args.save_prefix}_judges_flat.csv", index=False)
        problematic_entries.to_csv(
            f"{args.save_prefix}_problematic_judges.csv", index=False
        )
        results["response_averages"].to_csv(
            f"{args.save_prefix}_response_averages.csv", index=False
        )
        print(f"[INFO] Saved CSVs with prefix: {args.save_prefix}")
