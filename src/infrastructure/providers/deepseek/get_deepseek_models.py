"""
DeepSeek: get models

Behavior
- Attempts to fetch model listings from DeepSeek's OpenAI-compatible HTTP endpoint:
  GET {DEEPSEEK_BASE_URL or https://api.deepseek.com/v1}/models
- Persists to JSON at: src/providers/deepseek/deepseek-models.json
- If API key or HTTP client is unavailable or fails, falls back to cached JSON (no network).

Entry points recognized by the ModelRegistryRepository:
- run()  (preferred)
- get_models()/fetch_models()/update_models()/refresh_models() also provided for convenience
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore

from src.providers.base.get_models_base import save_provider_models, load_cached_models
from src.providers.base.repositories.keys import KeysRepository

PROVIDER = "deepseek"


def _resolve_key() -> Optional[str]:
    return KeysRepository().get_api_key(PROVIDER)


def _resolve_base_url() -> str:
    # Default to the documented base; allow override via env
    return os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")


def _fetch_via_http(api_key: str, base_url: str) -> List[Dict[str, Any]]:
    """
    Fetch model listings using OpenAI-compatible HTTP endpoint.

    Returns a list of dicts with at least {'id', 'name'} keys.
    """
    if requests is None:
        raise RuntimeError("requests library not available")

    url = base_url.rstrip("/") + "/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # Accept either a plain list or {"data": [...]}
    raw = data.get("data", data) if isinstance(data, dict) else data
    items: List[Dict[str, Any]] = []
    for it in raw or []:
        if isinstance(it, dict):
            # Normalize id/name
            mid = it.get("id") or it.get("model") or it.get("name") or str(it)
            name = it.get("name") or it.get("id") or str(it)
            row = {"id": str(mid), "name": str(name)}
            # Passthrough useful fields when present
            for k in ("created", "modalities", "context_length", "max_context", "capabilities"):
                if k in it:
                    row[k] = it[k]
            items.append(row)
        else:
            items.append({"id": str(it), "name": str(it)})
    return items


def run() -> List[Dict[str, Any]]:
    """
    Preferred entrypoint. Attempts online refresh; falls back to cached snapshot.

    Returns a list of dicts (models) for convenience; ModelRegistryRepository can
    also parse and persist this return value.
    """
    key = _resolve_key()
    if key:
        try:
            base = _resolve_base_url()
            items = _fetch_via_http(key, base)
            if items:
                save_provider_models(PROVIDER, items, fetched_via="api", metadata={"source": "deepseek_http_models"})
                return items
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
    print(f"[deepseek] loaded {len(models)} models")