import os
import time
from datetime import datetime
from typing import Dict, Optional

from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions
import requests

try:
    from zai import ZaiClient
    HAS_ZAI = True
except ImportError:
    ZaiClient = None
    HAS_ZAI = False

# =============================================================================
# Configuration & API Setup
# =============================================================================

load_dotenv()

# Google Generative AI (Gemini/Gemma)
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    for m in genai.list_models():
        if "embedContent" in m.supported_generation_methods:
            print(m.name)

# ZAI (for GLM models)
if os.getenv("ZAI_API_KEY") and HAS_ZAI:
    ZAI_API_KEY = os.getenv("ZAI_API_KEY")
    client_zai = ZaiClient(api_key=ZAI_API_KEY)

# Local Ollama API
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 120

# =============================================================================
# Helper: Call local Ollama models (via REST API)
# =============================================================================

def call_ollama_model(
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    timeout: int = OLLAMA_TIMEOUT,
) -> str:
    """
    Generic helper to call any Ollama model via /api/generate endpoint.
    
    Args:
        model_name: Ollama model identifier (e.g., "mistral", "ministral-3:3b")
        prompt: Input text prompt
        temperature: Generation temperature (0.0 = deterministic)
        timeout: Request timeout in seconds
    
    Returns:
        Generated text or error message
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
    }

    try:
        response = requests.post(OLLAMA_BASE_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except Exception as e:
        return f"{model_name} error: {str(e)}"


# =============================================================================
# Model-specific implementations
# =============================================================================

def call_mistral_local(prompt: str) -> str:
    """Mistral 7B via Ollama (zero-shot, temp=0)."""
    return call_ollama_model("mistral", prompt, temperature=0.5)

def call_ministral3b_local(prompt: str) -> str:
    """Ministral 3B via Ollama (zero-shot, temp=0)."""
    return call_ollama_model("ministral-3:3b", prompt, temperature=0.5)

def call_ministral8b_local(prompt: str) -> str:
    """Ministral 8B via Ollama (zero-shot, temp=0)."""
    return call_ollama_model("ministral-3:8b", prompt, temperature=0.5)

def call_qwen_local(prompt: str) -> str:
    """Qwen 2.5 Coder 7B via Ollama (zero-shot, temp=0)."""
    return call_ollama_model("qwen2.5-coder:7b", prompt, temperature=0.5)

def call_qwen2_5_local(prompt: str) -> str:
    """Qwen 2.5 Instruct 7B via Ollama (zero-shot, temp=0)."""
    return call_ollama_model("qwen2.5:7b-instruct", prompt, temperature=0.5)

def call_llama3_2_3b_local(prompt: str) -> str:
    """Llama 3.2 3B via Ollama (zero-shot, temp=0)."""
    return call_ollama_model("llama3.2:3b", prompt, temperature=0.5)

def call_deepseek_r1_7b_local(prompt: str) -> str:
    """Deepseek R1 7B via Ollama (zero-shot, temp=0)."""
    return call_ollama_model("deepseek-r1:7b", prompt, temperature=0.5)

def call_gemma_api(prompt: str) -> str:
    """
    Gemma 3 27B (Google Generative AI API).
    Handles rate limiting (15 RPM free tier).
    """
    try:
        model = genai.GenerativeModel(
            "models/gemma-3-27b-it",
            generation_config={"temperature": 0.5},
        )
        response = model.generate_content(prompt)
        if response.candidates:
            return response.text
        return "Error: No response candidate (safety filter)"
    except exceptions.ResourceExhausted:
        print("Rate limit reached (15 RPM). Waiting 10s...")
        time.sleep(10)
        return call_gemma_api(prompt)  # Retry once
    except exceptions.InvalidArgument:
        return "Error: Invalid arguments (check API key or model name)"
    except Exception as e:
        return f"Gemma error: {str(e)}"


def call_glm46_sdk(prompt: str) -> str:
    """
    GLM-4.6 via ZAI SDK (with rate limit handling).
    Retries indefinitely on error, waiting 60s between attempts.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client_zai.chat.completions.create(
                model="glm-4.6v-flash",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0.5,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"GLM error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in 60s...")
                time.sleep(60)
            else:
                return f"GLM error after {max_retries} attempts: {str(e)}"


def call_openai_api(prompt: str) -> str:
    """Placeholder for OpenAI API."""
    return "OpenAI simulated response"


def call_embedding_model(text: str) -> list:
    """
    Gemini Embedding 001 (Google Generative AI).
    Used for semantic similarity tasks. Rate limited to 1000 requests/day.
    """
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="SEMANTIC_SIMILARITY",
        )
        return result["embedding"]
    except exceptions.ResourceExhausted:
        print("Embedding quota reached. Waiting 10s...")
        time.sleep(10)
        return call_embedding_model(text)  # Retry once
    except exceptions.InvalidArgument as e:
        return f"Error: Invalid arguments ({str(e)})"
    except Exception as e:
        return f"Embedding error: {str(e)}"


# =============================================================================
# Main dispatcher: call_llm
# =============================================================================

def call_llm(prompt: str, models: str = "all") -> Dict[str, str]:
    """
    Generic dispatcher to call one or multiple LLM models.
    
    Args:
        prompt: Input text prompt
        models: Model selector
            - "all" → calls all registered models
            - "model_name" (str) → calls single model
            - ["model1", "model2"] (list) → calls specified models
    
    Returns:
        Dict {model_name: response_text} where response is either:
        - Generated text (success)
        - Error message (failure)
    """
    load_dotenv()

    # Model registry: maps display name to callable
    MODELS_MAPPING = {
        "openai": call_openai_api,
        "gemma": call_gemma_api,
        "mistral": call_mistral_local,
        "ministral-3b": call_ministral3b_local,
        "ministral-8b": call_ministral8b_local,
        "qwen2_5-7b-instruct": call_qwen2_5_local,
        "llama3_2-3b": call_llama3_2_3b_local,
        "deepseek-r1-7b": call_deepseek_r1_7b_local,
        "qwen": call_qwen_local,
        "glm": call_glm46_sdk,
    }

    # Resolve model selection
    if models == "all":
        selected_models = list(MODELS_MAPPING.keys())
    elif isinstance(models, str):
        selected_models = [models]
    else:
        selected_models = list(models)

    results = {}

    # Call each selected model
    for model_name in selected_models:
        if model_name in MODELS_MAPPING:
            print(f"Calling model: {model_name}...")
            response = MODELS_MAPPING[model_name](prompt)
            results[model_name] = response
        else:
            print(f"Warning: model '{model_name}' not found in registry.")
            results[model_name] = "Error: Model not found"

    return results
