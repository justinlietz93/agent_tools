"""
Composition module for CLI DI (edge wiring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.infrastructure.llm.adapter import LLMWrapperAdapter
from src.infrastructure.providers.base.repositories.keys import KeysRepository

if TYPE_CHECKING:
    from src.interfaces.repositories.settings_repository import SettingsRepository
    from src.interfaces.services.llm import ILLM
    from src.interfaces.services.tools import IToolCatalog, IToolInvocationAdapter


def build_illm(provider: str, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
    """
    Construct and return an ILLM instance based on provider choice.
    """
    if provider == "ollama":
        from src.infrastructure.llm.ollama_wrapper import OllamaWrapper
        wrapper = OllamaWrapper(api_key=api_key, base_url=base_url, model=model)
    elif provider == "openai":
        from src.infrastructure.llm.openai_compatible import OpenAICompatibleWrapper
        wrapper = OpenAICompatibleWrapper(api_key=api_key, base_url=base_url, model=model)
    elif provider == "deepseek":
        from src.wrappers.deepseek_wrapper import DeepseekToolWrapper
        wrapper = DeepseekToolWrapper(api_key=api_key, base_url=base_url, model=model)
    elif provider == "custom":
        from src.infrastructure.llm.openai_compatible import OpenAICompatibleWrapper
        wrapper = OpenAICompatibleWrapper(api_key=api_key, base_url=base_url, model=model)
    else:
        # Default to Ollama
        from src.infrastructure.llm.ollama_wrapper import OllamaWrapper
        wrapper = OllamaWrapper(api_key=api_key, base_url=base_url, model=model)

    return LLMWrapperAdapter(wrapper)


def build_settings_repository(db_path: str | None = None) -> "SettingsRepository":
    """
    Construct and return a SettingsRepository instance.
    """
    from src.repositories.sql.settings_sqlite import SqliteSettingsRepository
    return SqliteSettingsRepository(db_path=db_path)


def list_models(provider: str) -> list[str]:
    """
    List available models for the given provider using provider-specific getters.
    """
    provider = (provider or "").lower().strip()
    if provider == "ollama":
        from src.infrastructure.providers.ollama.get_ollama_models import get_models
        models = get_models()
    elif provider == "openai":
        from src.infrastructure.providers.openai.get_openai_models import get_models
        models = get_models()
    elif provider == "deepseek":
        from src.infrastructure.providers.deepseek.get_deepseek_models import get_models
        models = get_models()
    elif provider == "gemini":
        from src.infrastructure.providers.gemini.get_gemini_models import get_models
        models = get_models()
    elif provider == "xai":
        from src.infrastructure.providers.xai.get_xai_models import get_models
        models = get_models()
    else:
        models = []
    # Extract model ids
    model_ids = [m.get("id") or m.get("name") for m in models if isinstance(m, dict)]
    return sorted(set(model_ids))


def build_illm_via_providers(provider: str, model: str | None, settings: "SettingsRepository", api_key: str | None = None, base_url: str | None = None) -> "ILLM":
    """
    Build ILLM instance using provider factory and keys from providers.
    """
    # Resolve model if None
    if model is None:
        model = settings.get_pref(f"model.{provider}")

    # Get API key from providers keys repo if not provided
    if api_key is None:
        keys_repo = KeysRepository()
        api_key = keys_repo.get_resolution(provider).api_key

    # Create provider client
    from src.infrastructure.providers.base.factory import create_provider
    client = create_provider(provider, model=model, api_key=api_key, base_url=base_url)
    return LLMWrapperAdapter(client)

def get_registered_tools():
    """
    Get list of registered tool instances.
    """
    from src.infrastructure.tools.tool_manager import ToolManager
    manager = ToolManager(register_defaults=True)
    return list(manager.tools.values())
def build_tool_catalog() -> "IToolCatalog":
    """
    Construct and return an IToolCatalog instance.
    """
    from src.infrastructure.tools.catalog_adapter import ToolManagerCatalogAdapter
    return ToolManagerCatalogAdapter()


def build_tool_invoker() -> "IToolInvocationAdapter":
    """
    Construct and return an IToolInvocationAdapter instance.
    """
    from src.infrastructure.tools.invocation_adapter import ToolInvocationAdapter
    return ToolInvocationAdapter()