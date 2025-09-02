"""
Provider Factory

Purpose
- Centralized, provider-agnostic creation of LLMProvider adapters.
- Lazy-imports provider adapters to avoid heavy dependencies at import time.
- No side effects: strictly returns instances or raises a clear error.

Contracts
- Returns instances implementing [LLMProvider](Cogito/src/providers/base/interfaces.py:1).

Scope
- Providers supported in this scaffolding: openai (others to be added later).
"""

from __future__ import annotations

from typing import Optional, Dict, Type, Any

from typing import Any, Dict, Type


class UnknownProviderError(Exception):
    pass


def create_provider(provider: str, model: str | None = None, api_key: str | None = None, base_url: str | None = None, **kwargs: Any):
    """
    Create a provider wrapper instance based on provider name.
    """
    name = (provider or "").lower().strip()
    if name == "ollama":
        from src.infrastructure.llm.ollama_wrapper import OllamaWrapper
        wrapper = OllamaWrapper(api_key=api_key, base_url=base_url, model=model)
    elif name in {"openai", "deepseek", "gemini", "xai", "custom"}:
        from src.infrastructure.llm.openai_compatible import OpenAICompatibleWrapper
        wrapper = OpenAICompatibleWrapper(api_key=api_key, base_url=base_url, model=model)
    else:
        raise UnknownProviderError(f"Unknown provider '{provider}'")
    return wrapper


class ProviderFactory:
    """
    Create provider adapters based on a canonical name (e.g., 'openai').
    """

    # Map canonical provider names to import paths and class names
    _PROVIDERS: Dict[str, Dict[str, str]] = {
        "openai": {
            "module": "src.infrastructure.providers.openai.client",
            "class": "OpenAIProvider",
        },
        # Future entries:
        # "anthropic": {"module": "src.infrastructure.providers.anthropic.client", "class": "AnthropicProvider"},
        # "deepseek": {"module": "src.infrastructure.providers.deepseek.client", "class": "DeepseekProvider"},
        # "gemini": {"module": "src.infrastructure.providers.gemini.client", "class": "GeminiProvider"},
        # "xai": {"module": "src.infrastructure.providers.xai.client", "class": "XAIProvider"},
        # "openrouter": {"module": "src.infrastructure.providers.openrouter.client", "class": "OpenRouterProvider"},
        # "ollama": {"module": "src.infrastructure.providers.ollama.client", "class": "OllamaProvider"},
    }

    @classmethod
    def create(cls, provider: str, **kwargs: Any):
        """
        Create a provider adapter instance.

        Args:
            provider: Canonical provider name (e.g., 'openai')
            **kwargs: Adapter-specific constructor kwargs (optional)

        Returns:
            Instance implementing LLMProvider

        Raises:
            UnknownProviderError: if provider is not registered or cannot be imported.
        """
        name = (provider or "").lower().strip()
        spec = cls._PROVIDERS.get(name)
        if not spec:
            raise UnknownProviderError(f"Unknown provider '{provider}'")

        module_path, class_name = spec["module"], spec["class"]

        try:
            mod = __import__(module_path, fromlist=[class_name])
            klass: Type = getattr(mod, class_name)
            return klass(**kwargs)  # type: ignore[call-arg]
        except Exception as e:
            raise UnknownProviderError(f"Failed to initialize provider '{provider}': {e}") from e