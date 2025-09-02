"""
Provider selection and instantiation utilities for CLI.
"""

import os
from getpass import getpass
from typing import Optional, Tuple

from prompt_toolkit.completion import WordCompleter

from rich.panel import Panel
from rich.box import ROUNDED

from src.interfaces.services.llm import ILLM
from src.api.di.cli_composition import build_illm_via_providers, build_settings_repository
from src.interfaces.repositories.settings_repository import SettingsRepository


def select_provider(console, session) -> str:
    """Prompt to select a provider."""
    # Page refresh UX: clear screen before rendering provider menu
    try:
        console.clear()
    except Exception:
        pass
    console.print(
        Panel(
            "Select Provider\n\n"
            "1) DeepSeek (hosted)\n"
            "2) Ollama (local)\n"
            "3) OpenAI (hosted)\n"
            "4) Custom (OpenAI-compatible)\n",
            title="Provider",
            box=ROUNDED
        )
    )
    completer = WordCompleter(["1", "2", "3", "4"], ignore_case=True)
    while True:
        choice = session.prompt("Choose [1-4]: ", completer=completer).strip()
        if choice in {"1", "2", "3", "4"}:
            return {"1": "deepseek", "2": "ollama", "3": "openai", "4": "custom"}[choice]
        console.print("[warning]Invalid option. Enter 1, 2, 3, or 4.[/warning]")


def input_custom_config(console, session) -> Tuple[str, str, str]:
    """Gather custom OpenAI-compatible config: base_url, model, api_key."""
    console.print(
        Panel(
            "[box_title]Custom Provider[/box_title]\nProvide OpenAI-compatible connection details.",
            title="Custom Config",
            box=ROUNDED,
            border_style="accent",
        )
    )
    base_url = session.prompt("Base URL (e.g., http://host:port/v1): ").strip() or "http://localhost:11434/v1"
    model = session.prompt("Model (e.g., llama3.1, deepseek-reasoner): ").strip() or "llama3.1"
    api_key = session.prompt("API Key (leave blank if not required): ").strip()
    return base_url, model, api_key


def instantiate_wrapper(provider: str, console, session, repo: Optional[SettingsRepository] = None) -> ILLM:
    """Create the appropriate ILLM instance based on selection."""
    settings_repo = repo or build_settings_repository()

    if provider == "deepseek":
        illm = build_illm_via_providers(provider, None, settings_repo)
        print("Using DeepSeek provider")
        # Restore saved model for this provider if present
        saved = settings_repo.get_pref(f"model.{provider}")
        if saved:
            illm.model = saved
            print(f"Restored model to '{saved}'")
        return illm

    elif provider == "ollama":
        illm = build_illm_via_providers(provider, None, settings_repo)
        print("Using Ollama local provider (http://localhost:11434/v1)")
        saved = settings_repo.get_pref(f"model.{provider}")
        if saved:
            illm.model = saved
            print(f"Restored model to '{saved}'")
        return illm

    elif provider == "openai":
        # Prefer env; then DB; else prompt (no placeholder fallback)
        key = os.getenv("OPENAI_API_KEY") or (settings_repo.get_api_key("openai") or "")
        if not key:
            try:
                key = getpass("Enter OPENAI_API_KEY (input hidden, press Enter to cancel): ").strip()
            except Exception:
                key = ""
        if not key:
            print("OpenAI API key not provided. Returning to provider selection.")
            # Re-enter provider selection flow
            new_provider = select_provider(console, session)
            return instantiate_wrapper(new_provider, console, session, repo=settings_repo)

        # Persist and set env for process-wide usage
        try:
            settings_repo.set_api_key("openai", key)
        except Exception:
            pass
        os.environ["OPENAI_API_KEY"] = key

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        illm = build_illm_via_providers(provider, model, settings_repo, api_key=key, base_url=base_url)
        print("Using OpenAI provider https://api.openai.com/v1")
        saved = settings_repo.get_pref(f"model.{provider}")
        if saved:
            illm.model = saved
            print(f"Restored model to '{saved}'")
        return illm

    # custom (OpenAI-compatible)
    base_url, model, api_key = input_custom_config(console, session)
    illm = build_illm_via_providers(provider, model, settings_repo, api_key=api_key, base_url=base_url)
    print(f"Using Custom provider {base_url} model={model}")
    saved = settings_repo.get_pref(f"model.{provider}")
    if saved:
        illm.model = saved
        print(f"Restored model to '{saved}'")
    return illm