import os
import time
from typing import Dict, Callable, Any, Optional

from dotenv import load_dotenv
import requests
import google.generativeai as genai
from google.api_core import exceptions

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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# ZAI (for GLM models)
ZAI_API_KEY = os.getenv("ZAI_API_KEY")
client_zai: Optional[ZaiClient] = ZaiClient(api_key=ZAI_API_KEY) if (ZAI_API_KEY and HAS_ZAI) else None

# Local Ollama API
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 120


# =============================================================================
# Generic helpers
# =============================================================================

def call_ollama_model(
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    timeout: int = OLLAMA_TIMEOUT,
) -> str:
    """
    Call any Ollama model via /api/generate endpoint.
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


def _retry_with_delay(func: Callable[[], str], max_retries: int, delay_s: int, label: str) -> str:
    """
    Generic retry helper with fixed delay.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"{label} error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay_s}s...")
                time.sleep(delay_s)
            else:
                return f"{label} error after {max_retries} attempts: {str(e)}"


# =============================================================================
# Model-specific implementations
# =============================================================================

def _call_gemma(prompt: str, temperature: float = 0.5) -> str:
    """
    Gemma 3 27B (Google Generative AI API).
    Handles rate limiting (15 RPM free tier) with a single retry.
    """

    def _once() -> str:
        model = genai.GenerativeModel(
            "models/gemma-3-27b-it",
            generation_config={"temperature": temperature},
        )
        response = model.generate_content(prompt)
        if response.candidates:
            return response.text
        return "Error: No response candidate (safety filter)"

    try:
        return _once()
    except exceptions.ResourceExhausted:
        print("Rate limit reached (15 RPM). Waiting 10s...")
        time.sleep(10)
        try:
            return _once()
        except Exception as e:
            return f"Gemma error after retry: {str(e)}"
    except exceptions.InvalidArgument:
        return "Error: Invalid arguments (check API key or model name)"
    except Exception as e:
        return f"Gemma error: {str(e)}"


def _call_glm(prompt: str, temperature: float = 0.5) -> str:
    """
    GLM-4.6 via ZAI SDK (with retry logic).
    """
    if client_zai is None:
        return "GLM error: ZaiClient not available or API key missing"

    def _once() -> str:
        response = client_zai.chat.completions.create(
            model="glm-4.6v-flash",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=temperature,
        )
        return response.choices[0].message.content

    return _retry_with_delay(_once, max_retries=3, delay_s=60, label="GLM")


def call_openai_api(prompt: str) -> str:
    """
    Placeholder for OpenAI API.
    """
    return "OpenAI simulated response"


def call_embedding_model(text: str) -> list:
    """
    Gemini Embedding 001 (Google Generative AI).
    Used for semantic similarity tasks. Rate limited to 1000 requests/day.
    """

    def _once():
        return genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="SEMANTIC_SIMILARITY",
        )

    try:
        result = _once()
        return result["embedding"]
    except exceptions.ResourceExhausted:
        print("Embedding quota reached. Waiting 10s...")
        time.sleep(10)
        try:
            result = _once()
            return result["embedding"]
        except Exception as e:
            return f"Embedding error after retry: {str(e)}"
    except exceptions.InvalidArgument as e:
        return f"Error: Invalid arguments ({str(e)})"
    except Exception as e:
        return f"Embedding error: {str(e)}"


# =============================================================================
# Model registry
# =============================================================================

# Each entry describes how to call the model.
# For ollama models, we only store the ollama model name and temperature.

def _ollama_fn(ollama_name: str, temperature: float = 0.5) -> Callable[[str], str]:
    def _wrapped(prompt: str) -> str:
        return call_ollama_model(ollama_name, prompt, temperature=temperature)
    return _wrapped


MODELS_MAPPING: Dict[str, Callable[[str], str]] = {
    # Cloud / API models
    "openai": call_openai_api,
    "gemma": _call_gemma,
    "glm": _call_glm,

    # Local Ollama models
    "mistral": _ollama_fn("mistral", temperature=0.5),
    "ministral-3b": _ollama_fn("ministral-3:3b", temperature=0.5),
    "ministral-8b": _ollama_fn("ministral-3:8b", temperature=0.5),
    "qwen2_5-7b-instruct": _ollama_fn("qwen2.5:7b-instruct", temperature=0.5),
    "qwen": _ollama_fn("qwen2.5-coder:7b", temperature=0.5),
    "llama3_2-3b-instruct": _ollama_fn("llama3.2:3b-instruct-fp16", temperature=0.5),
    "llama2-7b": _ollama_fn("llama2:7b", temperature=0.5),
    "deepseek-r1-7b": _ollama_fn("deepseek-r1:7b", temperature=0.5),
}


# =============================================================================
# Main dispatcher: call_llm
# =============================================================================

def call_llm(prompt: str, models: Any = "all") -> Dict[str, str]:
    """
    Generic dispatcher to call one or multiple LLM models.

    Args:
        prompt: Input text prompt.
        models:
            - "all"       → calls all registered models
            - "model_name" (str) → calls single model
            - ["model1", "model2"] (list) → calls specified models

    Returns:
        Dict {model_name: response_text} where response is either:
          - Generated text (success)
          - Error message (failure)
    """
    # Resolve model selection
    if models == "all":
        selected_models = list(MODELS_MAPPING.keys())
    elif isinstance(models, str):
        selected_models = [models]
    else:
        selected_models = list(models)

    results: Dict[str, str] = {}

    for model_name in selected_models:
        fn = MODELS_MAPPING.get(model_name)
        if fn is None:
            print(f"Warning: model '{model_name}' not found in registry.")
            results[model_name] = "Error: Model not found"
            continue

        print(f"Calling model: {model_name}...")
        try:
            results[model_name] = fn(prompt)
        except Exception as e:
            results[model_name] = f"{model_name} error: {str(e)}"

    return results
