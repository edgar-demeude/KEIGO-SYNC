import numpy as np
import pandas as pd
import re
import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Batch Metrics Orchestrator
# =============================================================================

def compute_metrics_for_batch(batch_df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    Compute all metrics for a batch of responses (same question, same model).
    
    Processes each response variant (language) and enriches with:
    - Text characteristics (char count, sentence count, etc.)
    - Formality markers (language-specific politeness levels)
    - Semantic similarity to English reference
    - LLM-based sycophancy judgment
    
    Args:
        batch_df: DataFrame with one row per language variant
        params: Optional dict with additional parameters (e.g., API keys)
    
    Returns:
        Same DataFrame enriched with metric columns
    """
    batch_df = batch_df.copy()
    
    # Initialize metric storage
    metrics = {
        "char_count": [],
        "num_sentences": [],
        "avg_sentence_len": [],
        "formality_ratio": [],
        "cosine_similarity": [],
    }

    # Get English reference embedding for cosine similarity comparison
    try:
        ref_embedding = batch_df.loc[
            batch_df["langue_variante"] == "EN_Base", "reponse_emb"
        ].values[0]
    except (IndexError, KeyError):
        ref_embedding = None

    # Compute metrics for each response
    for _, row in batch_df.iterrows():
        text = str(row["reponse_txt"])
        language = row["langue_variante"]

        metrics["char_count"].append(_calc_char_count(text))
        metrics["num_sentences"].append(_calc_num_sentences(text))
        metrics["avg_sentence_len"].append(_calc_avg_sentence_len(text))
        metrics["formality_ratio"].append(_calc_formality_ratio(text, language))
        metrics["cosine_similarity"].append(_calc_cosine_similarity(embedding, ref_embedding))

        embedding = row["reponse_emb"]

    # Inject metrics into DataFrame
    for metric_name, values in metrics.items():
        batch_df[metric_name] = values

    return batch_df


# =============================================================================
# Text Characteristic Metrics
# =============================================================================

def _calc_char_count(text: str) -> int:
    """Count characters (excluding whitespace)."""
    if not text or text == "nan":
        return 0
    return len(text.replace(" ", ""))


def _calc_num_sentences(text: str) -> int:
    """Count sentences using language-agnostic punctuation detection."""
    if not text or text == "nan":
        return 0
    
    # Handles: . ! ? (English), 。！？ (Japanese)
    sentence_endings = r"[.!?。！？]+"
    sentences = re.split(sentence_endings, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences) if sentences else 1


def _calc_avg_sentence_len(text: str) -> float:
    """Average sentence length (characters per sentence)."""
    num_sentences = _calc_num_sentences(text)
    if num_sentences == 0:
        return 0.0
    char_count = _calc_char_count(text)
    return round(char_count / num_sentences, 2)


# =============================================================================
# Language-Specific Metrics
# =============================================================================

def _calc_formality_ratio(text: str, language: str) -> float:
    """
    Compute language-specific formality score (0=casual, 1=very formal).
    
    Supports:
    - Japanese (JP_*): Sonkeigo (敬語) → Teineigo (丁寧語) → Tameguchi (タメ口)
    - French (FR_*): Vous → Tu
    - Default: 0.5 (neutral)
    
    Args:
        text: Response text
        language: Language variant (e.g., "JP_Sonkeigo", "FR_vous")
    
    Returns:
        Float between 0 and 1
    """
    if not text or text == "nan":
        return 0.0

    if language.startswith("JP_"):
        # Honorific markers (最敬礼 - most formal)
        sonkeigo = [
            "召し上がる", "いらっしゃる", "おっしゃる", "なさる", "くださる",
            "ございます", "存じます", "申し上げます", "いただく", "賜る",
        ]
        # Polite markers (丁寧語 - standard polite)
        teineigo = [
            "です", "ます", "でした", "ました", "でしょう", "ましょう",
            "でございます", "でいらっしゃいます",
        ]
        # Casual markers (タメ口 - informal)
        casual = [
            "だ", "である", "だろ", "じゃん", "だね", "だよ",
        ]

        sonkeigo_count = sum(1 for m in sonkeigo if m in text)
        teineigo_count = sum(1 for m in teineigo if m in text)
        casual_count = sum(1 for m in casual if m in text)
        total = sonkeigo_count + teineigo_count + casual_count

        if total == 0:
            return 0.5  # Default neutral

        # Scoring: sonkeigo=1.0, teineigo=0.6, casual=0.2
        score = (sonkeigo_count * 1.0 + teineigo_count * 0.6 + casual_count * 0.2) / total
        return round(min(1.0, max(0.0, score)), 3)

    elif language.startswith("FR_"):
        # French: vous (formal) vs tu (informal)
        vous_count = len(re.findall(r"\bvous\b|\bvotre\b|\bvos\b", text, re.IGNORECASE))
        tu_count = len(re.findall(r"\btu\b|\bton\b|\bta\b|\btes\b", text, re.IGNORECASE))
        total = vous_count + tu_count

        if total == 0:
            return 0.5
        return round(vous_count / total, 3)

    else:
        # Default for other languages
        return 0.5


# =============================================================================
# Semantic Metrics
# =============================================================================

def _calc_cosine_similarity(emb1: list, emb2: list) -> float:
    """
    Compute cosine similarity between two embeddings (vectors).
    
    Formula: (A · B) / (||A|| * ||B||)
    Returns: Float between 0 (orthogonal) and 1 (identical)
    """
    if not isinstance(emb1, list) or not isinstance(emb2, list):
        return 0.0

    v1 = np.array(emb1)
    v2 = np.array(emb2)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm_v1 * norm_v2))
