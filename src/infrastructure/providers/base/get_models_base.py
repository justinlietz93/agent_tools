"""
Utilities for provider model registry fetchers.

This module provides helpers shared by provider-specific "get models" scripts:
- Normalization of raw API results into ModelInfo DTOs
- Saving snapshots to canonical JSON files under src/providers/{provider}/{provider}-models.json
- Loading cached snapshots when online refresh fails

Intended usage (inside provider script):
    from src.infrastructure.providers.base.get_models_base import save_provider_models, load_cached_models
    items = fetch_from_api()  # list of dicts/SDK objects
    save_provider_models("openai", items, fetched_via="api", metadata={"source": "openai_api"})
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import re

from .models import ModelInfo, ModelRegistrySnapshot
from .repositories.model_registry import ModelRegistryRepository


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict(obj: Any) -> Dict[str, Any]:
    """
    Best-effort conversion of SDK objects to dicts by probing common attributes.
    """
    if isinstance(obj, dict):
        return obj
    d: Dict[str, Any] = {}
    # Probe common attributes found on SDK objects; tolerate absence.
    for attr in (
        "id",
        "name",
        "slug",
        "model",
        "display_name",
        "family",
        "series",
        "context_length",
        "max_context",
        # Additional candidates that may exist on some SDK objects
        "created",              # epoch seconds
        "modalities",           # e.g., ["text","vision","audio"]
        "input_token_limit",    # sometimes exposed as per-model token limit
        "max_output_tokens",    # optional output cap (not mapped directly)
        "capabilities",         # provider-defined capabilities if present
    ):
        v = getattr(obj, attr, None)
        if v is not None:
            d[attr] = v
    # OpenAI often nests under .data; ignore here
    return d


def _norm_name_id(d: Dict[str, Any]) -> (str, str):
    """
    Determine (id, name) from a flexible dict.
    """
    sid = str(d.get("id") or d.get("model") or d.get("name") or d.get("slug") or "unknown")
    name = str(d.get("name") or d.get("display_name") or d.get("id") or d.get("model") or sid)
    return sid, name


def _infer_updated_at_from_id(model_id: str) -> Optional[str]:
    """
    Infer an ISO-like YYYY-MM-DD date from the model id when present, e.g.:
    'gpt-4o-mini-search-preview-2025-03-11' -> '2025-03-11'
    """
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", model_id)
    return m.group(1) if m else None


def _infer_family_from_id(model_id: str, provider: str) -> Optional[str]:
    """
    Infer a reasonable 'family' string from the model id for known providers.

    For OpenAI, prefer stable prefixes:
    - 'gpt-4o-mini-...' -> 'gpt-4o-mini'
    - 'gpt-4o-...'      -> 'gpt-4o'
    - 'o3-...'          -> 'o3'
    - 'o1-...'          -> 'o1'
    Fallback: first token before '-' or ':'.
    """
    lower = (model_id or "").lower()

    if provider == "openai":
        if "gpt-4o-mini" in lower:
            return "gpt-4o-mini"
        if "gpt-4o" in lower:
            return "gpt-4o"
        if lower.startswith("o3"):
            return "o3"
        if lower.startswith("o1"):
            return "o1"
        if lower.startswith("gpt-4"):
            return "gpt-4"

    # generic fallback: first token by dash/colon
    first = re.split(r"[-:]", lower)[0]
    return first or None


def normalize_items(provider: str, items: Iterable[Any]) -> List[ModelInfo]:
    """
    Convert arbitrary list of dictionaries/SDK objects into a list of ModelInfo.
    """
    out: List[ModelInfo] = []
    for it in items or []:
        if isinstance(it, ModelInfo):
            out.append(it)
            continue

        d = _as_dict(it)
        sid, name = _norm_name_id(d)

        # Family: prefer explicit field; otherwise infer from id
        fam = d.get("family") or d.get("series")
        if not isinstance(fam, str) or not fam:
            fam = _infer_family_from_id(sid, provider)

        # Context length: prefer explicit numeric fields from SDK; avoid fabricating unknowns
        ctx = (
            d.get("context_length")
            or d.get("max_context")
            or d.get("ctx")
            or d.get("input_token_limit")
            or d.get("context_window")
        )
        try:
            ctx_int = int(ctx) if ctx is not None else None
        except Exception:
            ctx_int = None

        # Capabilities: normalize to dict and enrich from modalities/id patterns when missing
        caps = d.get("capabilities")
        if caps is None:
            caps = {}
        elif not isinstance(caps, dict):
            caps = {"raw_capabilities": caps}

        # Map 'modalities' (if provided) to boolean flags inside capabilities
        mods = d.get("modalities")
        if isinstance(mods, (list, tuple)):
            for m in mods:
                mstr = str(m).lower()
                caps[mstr] = True
                # Normalize common synonyms
                if mstr in ("image", "vision"):
                    caps["vision"] = True

        # Baseline capability inference from model id patterns (provider-specific)
        def _infer_caps(provider: str, model_id: str) -> Dict[str, Any]:
            lower = (model_id or "").lower()
            inferred: Dict[str, Any] = {}
            if provider == "openai":
                # Reasoning + Responses API families
                if lower.startswith("o1") or lower.startswith("o3"):
                    inferred["reasoning"] = True
                    inferred["responses_api"] = True
                # Vision-capable families
                if "gpt-4o" in lower or "omni" in lower or "vision" in lower:
                    inferred["vision"] = True
                # Embeddings
                if "embedding" in lower or lower.startswith("text-embedding"):
                    inferred["embedding"] = True
                # Search-related previews
                if "search" in lower:
                    inferred["search"] = True
                # JSON/structured outputs are broadly supported; mark as likely
                inferred.setdefault("json_output", True)
            return inferred

        # Merge inferred capabilities if none present so far
        if not caps:
            caps = _infer_caps(provider, sid)
        else:
            # Non-destructive merge to add obvious booleans derived from id
            caps = {**_infer_caps(provider, sid), **caps}

        # Updated at: prefer explicit; else infer date fragment from id; else convert 'created' epoch if available
        updated = d.get("updated_at") if isinstance(d.get("updated_at"), str) else None
        if not updated:
            inferred_date = _infer_updated_at_from_id(sid)
            if inferred_date:
                updated = inferred_date
        if not updated and isinstance(d.get("created"), (int, float)):
            try:
                updated = datetime.fromtimestamp(d["created"], tz=timezone.utc).date().isoformat()
            except Exception:
                pass

        out.append(
            ModelInfo(
                id=sid,
                name=name,
                provider=provider,
                family=fam if isinstance(fam, str) and fam else None,
                context_length=ctx_int,
                capabilities=caps,
                updated_at=updated,
            )
        )
    return out


def save_provider_models(provider: str, items: Iterable[Any], fetched_via: str = "api", metadata: Optional[Dict[str, Any]] = None) -> Path:
    """
    Normalize and persist a provider model registry snapshot to its JSON file.

    Returns path to the written JSON.
    """
    models = normalize_items(provider, items)
    snapshot = ModelRegistrySnapshot(
        provider=provider,
        models=models,
        fetched_via=fetched_via,
        fetched_at=_now_iso(),
        metadata=metadata or {},
    )
    repo = ModelRegistryRepository()
    return repo.save_snapshot(snapshot)


def load_cached_models(provider: str) -> ModelRegistrySnapshot:
    """
    Load the last-saved snapshot from disk without refreshing.
    """
    repo = ModelRegistryRepository()
    return repo.list_models(provider, refresh=False)