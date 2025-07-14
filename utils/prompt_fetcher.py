import requests

MIDDLEWARE_URL = "http://localhost:8000"  # Change if your FastAPI runs elsewhere

def fetch_classifier_prompt(specialty_name: str, doctor_id: int) -> str:
    """Fetch the classifier prompt for a given specialty and doctor from the middleware API."""
    url = f"{MIDDLEWARE_URL}/classifier-prompts/{specialty_name}/{doctor_id}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data.get("prompt_text", "")
    except Exception as e:
        print(f"[PromptFetcher] Error fetching classifier prompt: {e}")
        return ""

def fetch_questioner_prompt(prompt_key: str) -> str:
    """Fetch the questioner prompt for a given key from the middleware API."""
    url = f"{MIDDLEWARE_URL}/questioner-prompts/{prompt_key}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data.get("prompt_text", "")
    except Exception as e:
        print(f"[PromptFetcher] Error fetching questioner prompt: {e}")
        return "" 