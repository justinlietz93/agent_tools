"""
Settings repository abstractions (Clean Architecture)

- Presentation (CLI) must depend only on these interfaces, not on concrete DBs.
- Infrastructure (sqlite, etc.) implements these contracts.
"""

from __future__ import annotations

from typing import Protocol, Optional, Dict


class SettingsRepository(Protocol):
    """
    Repository contract for persisting user settings, API keys, and preferences.
    """

    # API keys (by provider id, e.g., "openai", "deepseek", "ollama")
    def get_api_key(self, provider: str) -> Optional[str]:
        ...

    def set_api_key(self, provider: str, key: Optional[str]) -> None:
        """
        Persist/update API key for provider. Passing None deletes the key.
        """
        ...

    # Preferences (arbitrary simple string key/value pairs)
    def get_pref(self, key: str) -> Optional[str]:
        ...

    def set_pref(self, key: str, value: Optional[str]) -> None:
        """
        Persist/update preference. Passing None deletes the preference.
        """
        ...

    # Bulk helpers
    def all_api_keys(self) -> Dict[str, str]:
        ...

    def all_prefs(self) -> Dict[str, str]:
        ...