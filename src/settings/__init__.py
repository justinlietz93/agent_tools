"""
Settings repository factory (composition root helper)

Presentation code should depend on SettingsRepository (interfaces.py).
This module provides a small factory that lazily instantiates the chosen
infrastructure backend (SQLite by default) and returns a singleton instance.

This keeps presentation depending on a stable package boundary while allowing
the infrastructure implementation to live behind this factory.
"""

from __future__ import annotations

from typing import Optional
from .interfaces import SettingsRepository  # re-exported contract

_repo_singleton: Optional[SettingsRepository] = None


def get_settings_repo(db_path: Optional[str] = None) -> SettingsRepository:
    global _repo_singleton
    if _repo_singleton is None:
        # Lazy import to avoid hard-coupling at import time
        from .sqlite_repository import SqliteSettingsRepository
        _repo_singleton = SqliteSettingsRepository(db_path=db_path)
    return _repo_singleton