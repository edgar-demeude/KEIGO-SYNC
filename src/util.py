"""
Utility to backfill `answer_elements` into existing JSON files.

Usage example (from project root):
    python backfill_answer_elements.py \
        --excel ../data/Benchmark_Questions.xlsx \
        --input-json ../data/mistral_x1.json \
        --output-json ../data/mistral_x1_with_answer_elements.json

It will:
    1) Load the benchmark Excel with `load_all_benchmarks`.
    2) Build a mapping from (question_id, language_variant) -> answer_elements.
    3) Load the existing JSON list of responses.
    4) For each entry, attach the corresponding answer_elements (or "" if not found).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Any, List

import pandas as pd

import loaders


def build_answer_elements_index(df: pd.DataFrame) -> Dict[Tuple[str, str], str]:
    """
    Build a dictionary mapping (question_id, language_variant) to answer_elements.

    Args:
        df: DataFrame returned by `load_all_benchmarks`, expected to contain
            at least: question_id, language_variant, answer_elements.

    Returns:
        Dict[(question_id, language_variant) -> answer_elements_str]
    """
    index: Dict[Tuple[str, str], str] = {}

    if "answer_elements" not in df.columns:
        print("[WARN] 'answer_elements' column not found in benchmark DataFrame.")
        return index

    for _, row in df.iterrows():
        qid = row.get("question_id")
        lang = row.get("language_variant")
        ae = row.get("answer_elements")

        # Only index rows that have both IDs
        if qid and lang:
            key = (str(qid), str(lang))
            index[key] = "" if pd.isna(ae) else str(ae)

    print(f"[INFO] Built answer_elements index with {len(index)} entries")
    return index


def backfill_answer_elements(
    excel_path: str,
    input_json_path: str,
    output_json_path: str,
) -> None:
    """
    Backfill `answer_elements` into an existing responses JSON file.

    The JSON is expected to be a list of objects with:
        - question_id
        - language_variant
        (optionally already answer_elements, which will be overwritten)

    Args:
        excel_path: Path to benchmark Excel (multi-sheet).
        input_json_path: Path to existing JSON file to fix.
        output_json_path: Path where the corrected JSON will be written.
    """
    excel_path = str(excel_path)
    input_json_path = str(input_json_path)
    output_json_path = str(output_json_path)

    # -------------------------------------------------------------------------
    # 1) Load benchmark Excel and build index
    # -------------------------------------------------------------------------
    print(f"[LOADING] Benchmark Excel: {excel_path}")
    benchmark_df = loaders.load_all_benchmarks(excel_path)
    ae_index = build_answer_elements_index(benchmark_df)

    # -------------------------------------------------------------------------
    # 2) Load existing JSON
    # -------------------------------------------------------------------------
    print(f"[LOADING] Input JSON: {input_json_path}")
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects")

    print(f"[INFO] Loaded {len(data)} entries from JSON")

    # -------------------------------------------------------------------------
    # 3) Backfill answer_elements
    # -------------------------------------------------------------------------
    updated_count = 0
    missing_count = 0

    for entry in data:
        qid = entry.get("question_id")
        lang = entry.get("language_variant")

        if not qid or not lang:
            # If we cannot index, set empty string (or skip)
            entry["answer_elements"] = entry.get("answer_elements", "")
            missing_count += 1
            continue

        key = (str(qid), str(lang))
        if key in ae_index:
            entry["answer_elements"] = ae_index[key]
            updated_count += 1
        else:
            # No match in index, keep existing or set empty
            entry["answer_elements"] = entry.get("answer_elements", "")
            missing_count += 1

    print(f"[INFO] Updated answer_elements for {updated_count} entries")
    print(f"[INFO] Could not find answer_elements for {missing_count} entries")

    # -------------------------------------------------------------------------
    # 4) Save corrected JSON
    # -------------------------------------------------------------------------
    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Written updated JSON to: {out_path}")


if __name__ == "__main__":
    backfill_answer_elements(
        excel_path="../data/Benchmark_Questions.xlsx",
        input_json_path="../data/gemma_x4.json",
        output_json_path="../data/gemma_x4.json", # overwrite the same file
    )
