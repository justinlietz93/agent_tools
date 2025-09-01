"""
Keys Repository

Purpose
- Centralize API key and related credential resolution for providers.
- Prefer environment variables; optionally read from unified config when available.
- Keep logic contained in providers layer (no external writes).

Design
- Non-throwing accessors that return None if a key is not resolved.
- Simple, explicit env var map per provider.
- Optional config fallbacks via src.config_loader if present.

Usage
- repo = KeysRepository()
- key = repo.get_api_key("openai")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


# Optional unified config loader (do not fail if absent)
try:
    from src.config_loader import config_loader  # type: ignore
except Exception:
    config_loader = None  # type: ignore


@dataclass
class KeyResolution:
    provider: str
    api_key: Optional[str]
    source: str  # "env", "config", "none"
    extra: Dict[str, Any]


class KeysRepository:
    """
    Resolve provider credentials with a strict priority order:

    1) Environment variables (authoritative)
    2) Unified config.yaml (when available)
    3) None

    This repository only reads values; it does not mutate any external state.
    """

    ENV_MAP: Dict[str, str] = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "xai": "XAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        # ollama uses local runtime; no API key by default
    }

    def get_api_key(self, provider: str) -> Optional[str]:
        return self.get_resolution(provider).api_key

    def get_resolution(self, provider: str) -> KeyResolution:
        p = (provider or "").lower().strip()
        env_var = self.ENV_MAP.get(p)

        # 1) Environment
        if env_var:
            val = os.getenv(env_var)
            if val:
                return KeyResolution(provider=p, api_key=val, source="env", extra={"env_var": env_var})

        # 2) Unified config (best-effort)
        cfg_key, extra = self._from_config(p)

        if cfg_key:
            return KeyResolution(provider=p, api_key=cfg_key, source="config", extra=extra)

        # 3) Settings repository (SQLite) fallback
        try:
            # Lazy import to avoid hard dependency at import time
            from src.settings import get_settings_repo  # type: ignore
            repo = get_settings_repo()
            db_key = repo.get_api_key(p)
            if db_key:
                return KeyResolution(provider=p, api_key=db_key, source="settings_db", extra={"repo": "sqlite"})
        except Exception:
            # Silent fallback to none if settings infra not available
            pass

        # 4) None
        return KeyResolution(provider=p, api_key=None, source="none", extra=extra)

    # -------------------- internal helpers --------------------

    def _from_config(self, provider: str) -> (Optional[str], Dict[str, Any]):
        """
        Attempt to read API key equivalents from config.yaml via config_loader.
        Returns (key_or_none, meta).
        """
        meta: Dict[str, Any] = {}
        if not config_loader:
            meta["config_loader"] = "absent"
            return None, meta

        try:
            api_cfg = config_loader.get_section("api") or {}
            prov_cfg = api_cfg.get(provider) or {}
            meta["has_api_section"] = bool(api_cfg)
            meta["has_provider_section"] = bool(prov_cfg)

            # Common key fields across providers
            for field in ("api_key", "resolved_key", "key"):
                if field in prov_cfg and isinstance(prov_cfg[field], str) and prov_cfg[field]:
                    meta["field"] = field
                    return prov_cfg[field], meta

            # Some configs may store under 'token'
            token = prov_cfg.get("token")
            if isinstance(token, str) and token:
                meta["field"] = "token"
                return token, meta

            return None, meta
        except Exception as e:
            meta["error"] = str(e)
            return None, meta