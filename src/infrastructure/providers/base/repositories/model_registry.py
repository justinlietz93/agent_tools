"""
Model Registry Repository

Purpose
- Provide a normalized way to load and refresh per-provider model listings.
- Persist model listings into versioned JSON files located under:
  src/providers/{provider}/{provider}-models.json

Design
- Read snapshot from JSON and return typed models (ModelInfo).
- Optional refresh uses a provider-specific "get models" module if present.
- Ollama: optional fallback via local 'ollama list' command when permitted.
- Does not import SDKs; provider modules own API interactions.

Notes
- JSON shape tolerated:
  - {"models": [...], "fetched_at": "...", "source": "..."} OR
  - simple list: [...]
- When unknown fields are present, we preserve them in snapshot.metadata.

Dependencies
- DTOs: src/providers/base/models.py
- Optional provider modules:
  - src/providers/openai/get_openai_models.py
  - src/providers/anthropic/get_anthropic_models.py
  - src/providers/ollama/get_ollama_models.py
  - ... similar for others (gemini, xai, openrouter)

Usage
- repo = ModelRegistryRepository()
- snap = repo.list_models("openai", refresh=True)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..models import ModelInfo, ModelRegistrySnapshot


PROVIDER_JSON_FILENAMES: Dict[str, str] = {
    "openai": "openai-models.json",
    "anthropic": "anthropic-models.json",
    "deepseek": "deepseek-models.json",
    "gemini": "gemini-models.json",
    "xai": "xai-models.json",
    "openrouter": "openrouter-models.json",
    "ollama": "ollama-models.json",
}


class ModelRegistryError(Exception):
    pass


class ModelRegistryRepository:
    """
    Manage model registry snapshots for providers.

    This repository only reads/writes JSON on disk and optionally invokes
    provider-specific refresh scripts. It intentionally avoids importing
    any cloud SDKs directly.
    """

    def __init__(self, providers_root: Optional[Path] = None) -> None:
        # Default to this file's ../../ directory: src/providers
        self.providers_root = providers_root or Path(__file__).resolve().parents[2]

    # -------------------- Public API --------------------

    def list_models(self, provider: str, refresh: bool = False) -> ModelRegistrySnapshot:
        """
        Load provider model list. If refresh=True, attempt to update from source first.
        """
        provider = provider.lower().strip()
        if refresh:
            try:
                self._refresh_provider_models(provider)
            except Exception as e:
                # Non-fatal: attempt to read cached JSON even if refresh failed
                print(f"[ModelRegistry] Refresh failed for {provider}: {e}. Falling back to cache.", file=sys.stderr)

        # Load JSON snapshot (may be empty)
        data, src_path = self._read_provider_json(provider)
        models, meta = self._parse_models(provider, data)
        snapshot = ModelRegistrySnapshot(
            provider=provider,
            models=models,
            fetched_via=meta.get("source") or meta.get("fetched_via"),
            fetched_at=meta.get("fetched_at"),
            metadata={k: v for k, v in meta.items() if k not in {"source", "fetched_via", "fetched_at"}},
        )
        return snapshot

    def save_snapshot(self, snapshot: ModelRegistrySnapshot) -> Path:
        """
        Persist a snapshot to the provider's JSON file. Overwrites existing content.
        """
        path = self._json_path(snapshot.provider)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prefer consistent top-level shape
        payload: Dict[str, Any] = {
            "provider": snapshot.provider,
            "models": [m.to_dict() for m in snapshot.models],
            "fetched_at": snapshot.fetched_at or self._now_iso(),
            "fetched_via": snapshot.fetched_via or "local",
            "metadata": snapshot.metadata,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return path

    # -------------------- Refresh strategies --------------------

    def _refresh_provider_models(self, provider: str) -> None:
        """
        Try to refresh using a provider-specific module. Non-fatal on error.

        Expected module path pattern:
          src.infrastructure.providers.{provider}.get_{provider}_models

        Expected callable names (first found is used):
          - refresh_models()
          - update_models()
          - fetch_models()
          - get_models()
          - run()
          - main()
        """
        # Special-case minimal local listing for ollama if no module present
        if provider == "ollama":
            if self._try_provider_refresh_module(provider):
                return
            try:
                self._refresh_via_ollama_cli()
                return
            except Exception as e:
                raise ModelRegistryError(f"Ollama refresh failed: {e}") from e

        # Default path: attempt provider module
        ok = self._try_provider_refresh_module(provider)
        if not ok:
            raise ModelRegistryError(f"No refresh entry point found for provider '{provider}'")

    def _try_provider_refresh_module(self, provider: str) -> bool:
        module_name = f"src.infrastructure.providers.{provider}.get_{provider}_models"
        try:
            mod = import_module(module_name)
        except Exception:
            # Try package-local (when executed from within providers)
            try:
                module_name_local = f".{provider}.get_{provider}_models"
                mod = import_module(module_name_local, package="src.infrastructure.providers")
            except Exception:
                return False

        # Candidate function names in order
        candidates = ["refresh_models", "update_models", "fetch_models", "get_models", "run", "main"]
        for name in candidates:
            fn = getattr(mod, name, None)
            if callable(fn):
                result = fn()  # Adapters should write JSON file and may return snapshot or list
                # If adapter returned a list/dict, persist in our canonical JSON.
                self._persist_result_if_returned(provider, result)
                return True
        return False

    def _persist_result_if_returned(self, provider: str, result: Any) -> None:
        if result is None:
            return
        # Normalize common shapes
        models: List[ModelInfo] = []
        meta: Dict[str, Any] = {"fetched_via": "api", "fetched_at": self._now_iso()}

        try:
            if isinstance(result, list):
                # Assume list of dict-like models
                for item in result:
                    if isinstance(item, ModelInfo):
                        models.append(item)
                    elif isinstance(item, dict):
                        models.append(self._model_from_dict(provider, item))
                    else:
                        models.append(ModelInfo(id=str(item), name=str(item), provider=provider))
            elif isinstance(result, dict):
                # Try 'models' key
                if "models" in result and isinstance(result["models"], list):
                    for item in result["models"]:
                        if isinstance(item, ModelInfo):
                            models.append(item)
                        elif isinstance(item, dict):
                            models.append(self._model_from_dict(provider, item))
                # Copy metadata if present
                for k in ("fetched_via", "source", "fetched_at"):
                    if k in result and isinstance(result[k], str):
                        meta[k] = result[k]
            else:
                # Fallback: stringify
                models.append(ModelInfo(id=str(result), name=str(result), provider=provider))
        except Exception:
            # On parsing failure, do not overwrite existing registry
            return

        if models:
            snapshot = ModelRegistrySnapshot(provider=provider, models=models, fetched_via=meta.get("fetched_via"), fetched_at=meta.get("fetched_at"), metadata={})
            self.save_snapshot(snapshot)

    def _refresh_via_ollama_cli(self) -> None:
        """
        Use 'ollama list' to produce a simple model registry when available.

        This does not validate contracts; it only captures model names and sizes.
        """
        try:
            out = subprocess.check_output(["ollama", "list"], stderr=subprocess.STDOUT, text=True, timeout=10)
        except Exception as e:
            raise RuntimeError(f"ollama list failed: {e}")

        models: List[ModelInfo] = []
        # Expected sample output header:
        # NAME                                 ID              SIZE    MODIFIED
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if lines:
            # Skip header if it looks like the column header
            data_lines = lines[1:] if "NAME" in lines[0].upper() and "SIZE" in lines[0].upper() else lines
            for ln in data_lines:
                # naive split on whitespace; first token is name
                name = ln.split()[0]
                models.append(ModelInfo(id=name, name=name, provider="ollama"))

        snapshot = ModelRegistrySnapshot(
            provider="ollama",
            models=models,
            fetched_via="ollama_list",
            fetched_at=self._now_iso(),
            metadata={"source": "ollama_cli"},
        )
        self.save_snapshot(snapshot)

    # -------------------- JSON IO --------------------

    def _read_provider_json(self, provider: str) -> Tuple[Dict[str, Any], Path]:
        path = self._json_path(provider)
        if not path.exists():
            return ({}, path)
        try:
            # Fast-path: tolerate empty or whitespace-only files
            try:
                if path.stat().st_size == 0:
                    print(f"[ModelRegistry] Empty JSON file at {path}; treating as empty registry.", file=sys.stderr)
                    return ({}, path)
                content = path.read_text(encoding="utf-8")
                if not content.strip():
                    print(f"[ModelRegistry] Whitespace-only JSON file at {path}; treating as empty registry.", file=sys.stderr)
                    return ({}, path)
            except Exception:
                # If a stat/read preview error occurs, fall through to json.load
                pass

            with path.open("r", encoding="utf-8") as f:
                data = json.load(f) or {}
            return (data, path)
        except Exception as e:
            # Be resilient during offline/unit tests: warn and return empty registry
            print(f"[ModelRegistry] Warning: failed to parse provider JSON at {path}: {e}. Treating as empty registry.", file=sys.stderr)
            return ({}, path)

    def _json_path(self, provider: str) -> Path:
        filename = PROVIDER_JSON_FILENAMES.get(provider)
        if not filename:
            # default filename if not mapped
            filename = f"{provider}-models.json"
        return self.providers_root / provider / filename

    # -------------------- Parsing helpers --------------------

    def _parse_models(self, provider: str, data: Dict[str, Any]) -> Tuple[List[ModelInfo], Dict[str, Any]]:
        """
        Accept either a dict with 'models' or a plain list.
        """
        meta: Dict[str, Any] = {}
        raw_models: List[Any] = []

        if isinstance(data, dict):
            # Copy meta-ish fields if present
            for k in ("provider", "source", "fetched_via", "fetched_at", "metadata"):
                if k in data:
                    meta[k] = data[k]
            if isinstance(data.get("models"), list):
                raw_models = data["models"]
            elif isinstance(data.get("data"), list):  # tolerate alt schema
                raw_models = data["data"]
            else:
                raw_models = []
        elif isinstance(data, list):
            raw_models = data
        else:
            raw_models = []

        models: List[ModelInfo] = []
        for item in raw_models:
            if isinstance(item, ModelInfo):
                models.append(item)
            elif isinstance(item, dict):
                models.append(self._model_from_dict(provider, item))
            else:
                sid = str(item)
                models.append(ModelInfo(id=sid, name=sid, provider=provider))

        return models, meta

    def _model_from_dict(self, provider: str, d: Dict[str, Any]) -> ModelInfo:
        # Try common fields; tolerate different schemas
        mid = str(d.get("id") or d.get("name") or d.get("slug") or d.get("model") or "unknown")
        name = str(d.get("name") or d.get("id") or d.get("slug") or d.get("model") or mid)
        fam = d.get("family") or d.get("series")
        ctx = d.get("context_length") or d.get("max_context") or d.get("ctx") or None
        caps = d.get("capabilities") or {}
        updated_at = d.get("updated_at") or d.get("fetched_at")

        # Idempotent cast
        try:
            ctx = int(ctx) if ctx is not None else None
        except Exception:
            ctx = None

        if not isinstance(caps, dict):
            caps = {"raw_capabilities": caps}

        return ModelInfo(
            id=mid,
            name=name,
            provider=provider,
            family=fam if isinstance(fam, str) else None,
            context_length=ctx,
            capabilities=caps,
            updated_at=updated_at if isinstance(updated_at, str) else None,
        )

    # -------------------- Utils --------------------

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()