import os
from dotenv import load_dotenv

# ---------------------------------------------
# ---- Fonction globale d'appel des LLMs ------
# ---------------------------------------------

def call_llm(prompt, models="all"):
    """
    Fonction générique pour appeler un ou plusieurs LLMs.
    models: peut être "all", une chaîne (ex: "openai") ou une liste (ex: ["openai", "ollama"])
    """
    
    # LISTE DES MODELES UTILISES
    MODELS_MAPPING = {
        "openai": call_openai_api,
        "ollama": call_ollama_local,
        "mistral": call_mistral_api,
    }

    # 1. Gérer le choix des modèles
    if models == "all":
        selected_models = MODELS_MAPPING.keys()
    elif isinstance(models, str):
        selected_models = [models]
    else:
        selected_models = models

    results = {}

    # 2. Boucler sur les modèles sélectionnés
    for model_name in selected_models:
        if model_name in MODELS_MAPPING:
            print(f"Appel du modèle : {model_name}...")
            # On appelle la fonction spécifique associée au nom
            response = MODELS_MAPPING[model_name](prompt)
            results[model_name] = response
        else:
            print(f"⚠️ Le modèle {model_name} n'est pas reconnu.")
            results[model_name] = "Error: Model not found"

    return results

# ---------------------------------------------
# ---- Appels Spécifiques de chaque modèle ----
# ---------------------------------------------

def call_openai_api(prompt):
    return "Réponse simulée de GPT"

def call_ollama_local(prompt):
    return "Réponse simulée de Llama 3 (Local)"

def call_mistral_api(prompt):
    return "Réponse simulée de Mistral"