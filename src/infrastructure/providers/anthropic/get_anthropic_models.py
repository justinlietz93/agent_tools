"""
Anthropic: get models

Behavior
- Attempts to fetch model listings via Anthropic SDK.
- Persists to JSON at: src/providers/anthropic/anthropic-models.json
- If API key or SDK is unavailable, falls back to cached JSON (no network).

Entry points recognized by the ModelRegistryRepository:
- run()  (preferred)
- get_models()/fetch_models()/update_models()/refresh_models() also provided for convenience
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import anthropic  # anthropic>=0.49.0 recommended
except Exception:
    anthropic = None  # type: ignore

from src.infrastructure.providers.base.get_models_base import save_provider_models, load_cached_models
from src.infrastructure.providers.base.repositories.keys import KeysRepository


PROVIDER = "anthropic"


def _fetch_via_sdk(api_key: str) -> List[Any]:
    """
    Fetch model listings using Anthropic SDK. Returns raw items (SDK objects or dicts).
    """
    if anthropic is None:
        raise RuntimeError("anthropic SDK not available")

    # Prefer modern initialization
    client = getattr(anthropic, "Anthropic", None)
    if client is None:
        raise RuntimeError("Anthropic.Anthropic class not found in SDK")

    client = client(api_key=api_key)  # type: ignore[call-arg]

    # Modern clients expose models.list()
    models_obj = getattr(getattr(client, "models", None), "list", None)
    if callable(models_obj):
        resp = models_obj()
        data = getattr(resp, "data", None)
        if data is None and isinstance(resp, dict):
            data = resp.get("data", [])
        return list(data or [])

    # Fallback: try attribute access commonly used in older SDKs
    list_fn = getattr(client, "models", None)
    if callable(list_fn):
        try:
            resp = list_fn()  # type: ignore[call-arg]
            data = getattr(resp, "data", None)
            if data is None and isinstance(resp, dict):
                data = resp.get("data", [])
            return list(data or [])
        except Exception:
            pass

    raise RuntimeError("Anthropic SDK does not expose a models listing in this version")


def _resolve_key() -> Optional[str]:
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
            save_provider_models(PROVIDER, items, fetched_via="api", metadata={"source": "anthropic_sdk"})
            out: List[Dict[str, Any]] = []
            for it in items:
                mid = getattr(it, "id", None) or getattr(it, "name", None) or str(it)
                name = getattr(it, "name", None) or mid
                out.append({"id": mid, "name": name})
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
    print(f"[anthropic] loaded {len(models)} models")