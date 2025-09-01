"""
OpenAI: get models

Behavior
- Attempts to fetch model listings via OpenAI SDK.
- Persists to JSON at: src/providers/openai/openai-models.json
- If API key or SDK is unavailable, falls back to cached JSON (no network).

Entry points recognized by the ModelRegistryRepository:
- run()  (preferred)
- get_models()/fetch_models()/update_models()/refresh_models() also provided for convenience
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI  # openai>=1.0.0
except Exception:
    OpenAI = None  # type: ignore

from src.providers.base.get_models_base import save_provider_models, load_cached_models
from src.providers.base.repositories.keys import KeysRepository


PROVIDER = "openai"


def _fetch_via_sdk(api_key: str) -> List[Any]:
    """
    Fetch model listings using OpenAI SDK. Returns raw items (SDK objects or dicts).
    """
    if not OpenAI:
        raise RuntimeError("openai SDK not available")
    client = OpenAI(api_key=api_key)
    resp = client.models.list()
    # Support both object and dict
    data = getattr(resp, "data", None)
    if data is None and isinstance(resp, dict):
        data = resp.get("data", [])
    return list(data or [])


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
            # Initial list call
            items = _fetch_via_sdk(key)

            # Enrich with per-model metadata where available
            # Note: client.models.retrieve(id) often includes fields like `created`,
            # `modalities`, and token limits; we map these through to normalization.
            client = OpenAI(api_key=key)

            enriched: List[Dict[str, Any]] = []
            for it in items:
                mid = getattr(it, "id", None) or getattr(it, "model", None) or getattr(it, "name", None) or str(it)
                name = getattr(it, "name", None) or getattr(it, "id", None) or str(it)

                # Best-effort details fetch; tolerate failures
                det = None
                try:
                    det = client.models.retrieve(str(mid))
                except Exception:
                    det = None  # keep minimal

                # Gather possible fields from either list item or retrieved detail
                def g(obj: Any, attr: str):
                    return getattr(obj, attr, None) if obj is not None else None

                modalities = g(det, "modalities") or g(it, "modalities")
                input_token_limit = g(det, "input_token_limit") or g(it, "input_token_limit") or g(det, "context_window")
                created = g(det, "created") or g(it, "created")
                context_length = g(det, "context_length") or g(it, "context_length") or g(it, "max_context")

                # Build capabilities using modalities and id heuristics
                caps: Dict[str, Any] = {}
                if isinstance(modalities, (list, tuple)):
                    for m in modalities:
                        mstr = str(m).lower()
                        caps[mstr] = True
                        if mstr in ("image", "vision"):
                            caps["vision"] = True

                lower = str(mid).lower()
                if lower.startswith(("o1", "o3")):
                    caps["reasoning"] = True
                    caps["responses_api"] = True
                if "gpt-4o" in lower or "omni" in lower or "vision" in lower:
                    caps["vision"] = True
                if "embedding" in lower or lower.startswith("text-embedding"):
                    caps["embedding"] = True
                if "search" in lower:
                    caps["search"] = True
                # JSON structured outputs generally supported by major chat families
                caps.setdefault("json_output", True)

                row: Dict[str, Any] = {"id": mid, "name": name}

                # Prefer numeric context length from explicit fields
                if context_length is None and input_token_limit is not None:
                    context_length = input_token_limit
                if context_length is not None:
                    try:
                        row["context_length"] = int(context_length)
                    except Exception:
                        pass

                if isinstance(modalities, (list, tuple)):
                    row["modalities"] = list(modalities)

                if caps:
                    row["capabilities"] = caps

                if isinstance(created, (int, float)):
                    # Normalizer will convert created epoch to updated_at date
                    row["created"] = int(created)

                enriched.append(row)

            # Persist enriched snapshot; base normalizer will handle family/updated_at inference
            save_provider_models(PROVIDER, enriched, fetched_via="api", metadata={"source": "openai_sdk_enriched"})
            return [{"id": it["id"], "name": it["name"]} for it in enriched]
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
    print(f"[openai] loaded {len(models)} models")