"""
SQLite Settings Repository (Infrastructure)

- Implements SettingsRepository interface to persist:
  * API keys per provider
  * Arbitrary user preferences (theme, colors, last provider, model per provider, etc.)

- File size limit: <500 LOC (kept small)
- No dependencies on presentation; pure infrastructure
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional, Dict

from .interfaces import SettingsRepository


class SqliteSettingsRepository(SettingsRepository):
    """
    SQLite-backed implementation for settings persistence.

    Schema:
      - api_keys(provider TEXT PRIMARY KEY, key TEXT)
      - prefs(key TEXT PRIMARY KEY, value TEXT)
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        # Default DB location: project-local ./data/settings.db (creates directory)
        default_dir = Path(os.getenv("AGENT_TOOLS_DATA_DIR", ".")) / "data"
        default_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or str(default_dir / "settings.db")
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
              provider TEXT PRIMARY KEY,
              key      TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS prefs (
              key   TEXT PRIMARY KEY,
              value TEXT
            )
            """
        )
        self._conn.commit()

    # ------------- API Keys -------------

    def get_api_key(self, provider: str) -> Optional[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT key FROM api_keys WHERE provider = ?", (provider.lower().strip(),))
        row = cur.fetchone()
        return row[0] if row else None

    def set_api_key(self, provider: str, key: Optional[str]) -> None:
        p = provider.lower().strip()
        cur = self._conn.cursor()
        if key is None or key == "":
            cur.execute("DELETE FROM api_keys WHERE provider = ?", (p,))
        else:
            cur.execute(
                "INSERT INTO api_keys(provider, key) VALUES(?, ?) "
                "ON CONFLICT(provider) DO UPDATE SET key = excluded.key",
                (p, key),
            )
        self._conn.commit()

    def all_api_keys(self) -> Dict[str, str]:
        cur = self._conn.cursor()
        cur.execute("SELECT provider, key FROM api_keys")
        return {row[0]: row[1] for row in cur.fetchall()}

    # ------------- Preferences -------------

    def get_pref(self, key: str) -> Optional[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT value FROM prefs WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set_pref(self, key: str, value: Optional[str]) -> None:
        cur = self._conn.cursor()
        if value is None:
            cur.execute("DELETE FROM prefs WHERE key = ?", (key,))
        else:
            cur.execute(
                "INSERT INTO prefs(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
        self._conn.commit()

    def all_prefs(self) -> Dict[str, str]:
        cur = self._conn.cursor()
        cur.execute("SELECT key, value FROM prefs")
        return {row[0]: row[1] for row in cur.fetchall()}