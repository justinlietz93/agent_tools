"""
Gemini: get models

Behavior
- Attempts to fetch model listings via Google Generative AI SDK (google-generativeai).
- Persists to JSON at: src/providers/gemini/gemini-models.json
- If API key or SDK is unavailable, falls back to cached JSON (no network).

Entry points recognized by the ModelRegistryRepository:
- run()  (preferred)
- get_models()/fetch_models()/update_models()/refresh_models() also provided for convenience
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai  # pip install google-generativeai
except Exception:
    genai = None  # type: ignore

from src.infrastructure.providers.base.get_models_base import save_provider_models, load_cached_models
from src.infrastructure.providers.base.repositories.keys import KeysRepository


PROVIDER = "gemini"


def _fetch_via_sdk(api_key: str) -> List[Any]:
    """
    Fetch model listings using Google Generative AI SDK. Returns raw items (SDK objects or dicts).
    """
    if genai is None:
        raise RuntimeError("google-generativeai SDK not available")

    # Configure API key
    genai.configure(api_key=api_key)

    # list_models returns iterable of Model objects with attributes:
    # - name (e.g., 'models/gemini-1.5-pro')
    # - supported_generation_methods
    # - input_token_limit, output_token_limit (on some versions)
    models = list(genai.list_models())

    return models


def _resolve_key() -> Optional[str]:
    # GEMINI_API_KEY via KeysRepository
    return KeysRepository().get_api_key(PROVIDER)


def run() -> List[Dict[str, Any]]:
    """
    Preferred entrypoint. Attempts online refresh; falls back to cached snapshot.

    Returns a list of dicts (models) for convenience; ModelRegistryRepository can
    also parse and persist this return value.
    """
    key = _resolve_key()
    if key:
        try:
            items = _fetch_via_sdk(key)
            # Persist normalized snapshot
            save_provider_models(PROVIDER, items, fetched_via="api", metadata={"source": "google_generativeai"})
            # Return lightweight list of dicts
            out: List[Dict[str, Any]] = []
            for it in items:
                # Try to extract useful fields defensively
                name = getattr(it, "name", None) or str(it)
                # Some SDK versions expose token limits
                input_limit = getattr(it, "input_token_limit", None)
                output_limit = getattr(it, "output_token_limit", None)
                out.append(
                    {
                        "id": name,
                        "name": name,
                        "input_token_limit": input_limit,
                        "output_token_limit": output_limit,
                    }
                )
            return out
        except Exception:
            # Fall through to cached
            pass

    snap = load_cached_models(PROVIDER)
    return [m.to_dict() for m in snap.models]


# Aliases for repository compatibility
def get_models() -> List[Dict[str, Any]]:
    return run()


def fetch_models() -> List[Dict[str, Any]]:
    return run()


def update_models() -> List[Dict[str, Any]]:
    return run()


def refresh_models() -> List[Dict[str, Any]]:
    return run()


if __name__ == "__main__":
    models = run()
    print(f"[gemini] loaded {len(models)} models")