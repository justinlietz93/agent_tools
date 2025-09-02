"""
Ollama: get models

Behavior
- Attempts to fetch locally installed models via the 'ollama list' CLI.
- Persists to JSON at: src/providers/ollama/ollama-models.json
- If CLI is unavailable or fails, falls back to cached JSON (no network).

Entry points recognized by the ModelRegistryRepository:
- run()  (preferred)
- get_models()/fetch_models()/update_models()/refresh_models() also provided for convenience
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List

from src.infrastructure.providers.base.get_models_base import save_provider_models, load_cached_models

PROVIDER = "ollama"


def _fetch_via_cli() -> List[Dict[str, Any]]:
    """
    Fetch model listings using 'ollama list' command.
    Tries JSON output first; falls back to parsing table output.
    Returns a list of dicts with at least {'id', 'name'} keys.
    """
    # Try JSON output first (supported on modern ollama)
    try:
        out = subprocess.check_output(
            ["ollama", "list", "--json"], stderr=subprocess.STDOUT, text=True, timeout=10
        )
        data = json.loads(out)
        # Accept either a plain list or {"models": [...]}
        raw = data.get("models", data) if isinstance(data, dict) else data
        items: List[Dict[str, Any]] = []
        for it in raw or []:
            if isinstance(it, dict):
                name = it.get("name") or it.get("model") or str(it)
                items.append({"id": name, "name": name, **it})
            else:
                items.append({"id": str(it), "name": str(it)})
        return items
    except Exception:
        pass

    # Fallback to parsing table output
    out = subprocess.check_output(["ollama", "list"], stderr=subprocess.STDOUT, text=True, timeout=10)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return []
    # Skip header if present
    data_lines = lines[1:] if "NAME" in lines[0].upper() and "SIZE" in lines[0].upper() else lines
    items: List[Dict[str, Any]] = []
    for ln in data_lines:
        # naive split: first token is model name
        parts = ln.split()
        if not parts:
            continue
        name = parts[0]
        items.append({"id": name, "name": name})
    return items


def run() -> List[Dict[str, Any]]:
    """
    Preferred entrypoint. Attempts local refresh; falls back to cached snapshot.

    Returns a list of dicts (models) for convenience; ModelRegistryRepository can
    also parse and persist this return value.
    """
    try:
        items = _fetch_via_cli()
        if items:
            save_provider_models(PROVIDER, items, fetched_via="ollama_list", metadata={"source": "ollama_cli"})
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
    print(f"[ollama] loaded {len(models)} models")