import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
import time
from google.api_core import exceptions
import requests
from datetime import datetime
from zai import ZaiClient


# ---------------------------------------------------------
# ---------------- Gestion des clefs APIs -----------------
# ---------------------------------------------------------
load_dotenv()

ZAI_API_KEY = os.getenv("ZAI_API_KEY")  # Make sure your .env has this key
client_zai = ZaiClient(api_key=ZAI_API_KEY)
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
        "gemma": call_gemma_api,
        "mistral": call_mistral_local,
        "glm": call_glm46_sdk,
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

def call_glm46_sdk(prompt: str) -> str:

    while True:
        try:
            response = client_zai.chat.completions.create(
                model="glm-4.6v-flash",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.0
            )
            return response.choices[0].message.content

        except Exception as e:
            # Catch 429 / Rate limit or other errors
            print(f"[GLM Error] {e}. Retrying in 60 seconds...")
            time.sleep(60)
            continue


def call_openai_api(prompt):
    return "GPT réponse simulée"

def call_mistral_local(prompt: str) -> str:
    """
    Appelle Mistral 7B local via Ollama.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral",      # ou mistral:instruct
        "prompt": prompt,
        "stream": False,
        "temperature": 0.0,   # / déterministe
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # On normalise ici : on ne renvoie que le texte
        return data.get("response", "")
    except Exception as e:
        return f"Mistral local error: {e}"
    
def call_ministral_local(prompt: str) -> str:
    """
    Appelle Ministral-3:8b local via Ollama.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "ministral-3:8b",      # ou mistral:instruct
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # On normalise ici : on ne renvoie que le texte
        return data.get("response", "")
    except Exception as e:
        return f"Mistral local error: {e}"

def call_gemma_api(prompt):
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
        print("⏳ Quota Gemma atteint (15 RPM), pause de 10s...")
        time.sleep(10)
        return call_gemma_api(prompt) # Tentative de récursion
        
    except exceptions.InvalidArgument:
        return "Error: Invalid arguments (check API Key or Model Name)"
        
    except Exception as e:
        return f"Gemma Error: {str(e)}"

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

