import os
from dotenv import load_dotenv
import google.generativeai as genai
import time
from google.api_core import exceptions


# ---------------------------------------------------------
# ---------------- Gestion des clefs APIs -----------------
# ---------------------------------------------------------
load_dotenv()

if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            print(m.name)

# ---------------------------------------------------------
# -- Fonction Globale qui appelle les différents modèles --
# ---------------------------------------------------------

def call_llm(prompt, models="all"):
    """
    Fonction générique pour appeler un ou plusieurs LLMs.
    models: peut être "all", une chaîne (ex: "openai") ou une liste (ex: ["openai", "ollama"])
    """
    load_dotenv()
    
    # LISTE DES MODELES UTILISES
    MODELS_MAPPING = {
        "openai": call_openai_api,
        "gemini": call_gemini_api,
        "ollama": call_ollama_local,
        "mistral": call_mistral_api,
    }

    # 1. Gérer le choix des modèles
    if models == "all":
        selected_models = list(MODELS_MAPPING.keys())
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
    return "GPT réponse simulée"

def call_ollama_local(prompt):
    return "Llama 3 réponse simulée (Local)"

def call_mistral_api(prompt):
    return "Mistral réponse simulée"

def call_gemini_api(prompt):
    """
    Appelle Gemma_3_27b-it avec gestion des limites de requêtes.
    """
    try:
        model = genai.GenerativeModel('models/gemma-3-27b-it')
        response = model.generate_content(prompt)
        
        if response.candidates:
            return response.text
        else:
            return "Error: No response candidate (potentially blocked by safety filters)"

    except exceptions.ResourceExhausted:
        # Si tu es en gratuit, tu as droit à 15 requêtes/min.
        # Si on dépasse, on attend et on réessaye une fois.
        print("⏳ Quota Gemini atteint (15 RPM), pause de 10s...")
        time.sleep(10)
        return call_gemini_api(prompt) # Tentative de récursion
        
    except exceptions.InvalidArgument:
        return "Error: Invalid arguments (check API Key or Model Name)"
        
    except Exception as e:
        return f"Gemini Error: {str(e)}"

def call_embedding_model(answer):
    """
    Appelle gemini-embedding-1.000, limite 1000 prompts/jour
    """
    model_name = "models/gemini-embedding-001"

    print(f"Appel du modèle : {model_name}...")

    try:
        result = genai.embed_content(
            model=model_name,
            content=answer,
            task_type="SEMANTIC_SIMILARITY"
        )

        return result['embedding']

    except exceptions.ResourceExhausted:
        print("⏳ Quota Embedding atteint, pause de 10s...")
        time.sleep(10)
        return call_embedding_model(answer)

    except exceptions.InvalidArgument as e:
        return f"Error: Invalid arguments ({str(e)})"

    except Exception as e:
        return f"Embedding Error: {str(e)}"

