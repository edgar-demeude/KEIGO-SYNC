"""
Global configuration for the project.

All core parameters (models, iterations, file names, judge setup)
are defined here and imported by other modules.
"""

# ---------------------------------------------------------------------------
# Generation / benchmark configuration
# ---------------------------------------------------------------------------

# Base model(s) used to generate answers.
# Can be a single string or a list of model names.
GEN_MODELS = ["mistral"] # e.g. ["mistral", "llama2-7b"]

# Number of iterations (batches) for each question.
GEN_NB_ITER = 3

# Limit on number of questions (-1 = all).
GEN_NB_QUESTIONS = 1

# Whether to compute embeddings and metrics after generation.
COMPUTE_EMBEDDINGS = True
COMPUTE_METRICS = True

# Excel benchmark file.
BENCHMARK_EXCEL_PATH = "../data/Benchmark_Questions.xlsx"

# ---------------------------------------------------------------------------
# Judge configuration
# ---------------------------------------------------------------------------

# Name of the tested model answers file (without .json) used by judge.py.
# By default, you can manually set one of the per-model base names here, e.g. "mistral_x3".
TESTED_MODEL_ANSWERS = "qwen_x3"  # or "llama2-7b_x3", etc.

# Judge models to use.
JUDGE_MODELS = [
    "qwen2_5-7b-instruct",
    "llama3_2-3b-instruct",
    # "ministral-3b",
]

# Judge input/output paths.
JUDGE_INPUT_JSON_PATH = f"../data/{TESTED_MODEL_ANSWERS}_final.json"
JUDGE_OUTPUT_JSON_PATH = f"../data/judge/{TESTED_MODEL_ANSWERS}_judged.json"

# Max entries for judge (-1 = all).
JUDGE_MAX_ENTRIES = -1

# Whether to print raw judge responses.
JUDGE_VERBOSE = True
