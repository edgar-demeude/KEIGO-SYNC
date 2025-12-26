from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ---------------------------------------------------------
# ---- Fonction Globale : L'Orchestrateur du Batch
# ---------------------------------------------------------

def compute_metrics_for_batch(batch_df, params=None):
    """
    Prend un batch de 6 lignes (même question, même modèle).
    Boucle sur les lignes pour calculer les scores individuels.
    """
    batch_df = batch_df.copy()

    # On crée des listes pour stocker nos métriques
    results = {
        'char_count': [],
        'num_sentences': [],
        'avg_sentence_len': [],
        'formality_ratio': [],
        'cosine_similarity': [],
        'refusal_rate': [],
        'llm_as_a_judge': []
    }

    # Texte de référence pour le calcul du cosine similarity : l'anglais ?
    try:
        ref_series = batch_df.loc[batch_df['langue_variante'] == 'EN_controle', 'reponse_txt']
        ref_text = str(ref_series.values[0]) if not ref_series.empty else ""
    except Exception:
        ref_text = ""

    # --- Calcul des différentes métriques ---
    for index, row in batch_df.iterrows():
        text = str(row['reponse_txt'])
        lang = row['langue_variante']
        
        
        results['char_count'].append(_calc_char_count(text))
        results['num_sentences'].append(_calc_num_sentences(text))
        results['avg_sentence_len'].append(_calc_avg_sentence_len(text))
        results['formality_ratio'].append(_calc_formality_ratio(text, lang))
        
        results['cosine_similarity'].append(_calc_cosine_similarity(row, ref_text, params))
        results['refusal_rate'].append(_calc_refusal_rate(text))
        results['llm_as_a_judge'].append(_calc_llm_judge(row, params))

    # On injecte toutes les listes dans le DataFrame
    for metric_name, values in results.items():
        batch_df[metric_name] = values

    return batch_df

# ---------------------------------------------------------
# --------- Fonctions de calcul des métriques -------------
# ---------------------------------------------------------

# --- Text Characteristics ---

def _calc_char_count(text):
    """Compte des caractères (excluant les espaces)."""
    prefix = str(text)[:4] 
    return f"{prefix}_42.1"

def _calc_num_sentences(text):
    """Nombre de phrases."""
    prefix = str(text)[:4] 
    return f"{prefix}_42.2"

def _calc_avg_sentence_len(text):
    """Longueur moyenne des phrases (caractères / phrases)."""
    prefix = str(text)[:4] 
    return f"{prefix}_42.3"

def _calc_formality_ratio(text, lang):
    """Ratio style neutre vs poli (spécifique au japonais)."""
    prefix = str(text)[:4] 
    return f"{prefix}_42.4"

# --- Content Metrics ---

def _calc_cosine_similarity(row, ref_text, params):
    """Compare 'reponse_txt' avec 'Elements_de_reponse'."""
    # On a accès à toute la ligne 'row' si besoin
    prefix = str(row['question_txt'])[:4] 
    return f"{prefix}_42.5"

def _calc_refusal_rate(text):
    """Détecte si le modèle a refusé de répondre."""
    prefix = str(text)[:4] 
    return f"{prefix}_42.6"

def _calc_llm_judge(row, params):
    """Évaluation qualitative par un LLM tiers."""
    prefix = str(row['question_txt'])[:4]  
    return f"{prefix}_42.7"