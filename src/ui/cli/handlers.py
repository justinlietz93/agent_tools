"""
Command handlers for CLI.
"""
from __future__ import annotations

import json
import os
import sys
from getpass import getpass
from typing import Optional, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.box import ROUNDED

from .provider import select_provider, instantiate_wrapper

from src.interfaces.services.llm import ILLM
from src.interfaces.services.tools import IToolCatalog, IToolInvocationAdapter
from src.interfaces.repositories.settings_repository import SettingsRepository
from src.api.di.cli_composition import build_settings_repository, list_models


def list_tools(console: Console, catalog: IToolCatalog) -> None:
    """Render a table of registered tools."""
    table = Table(title="Registered Tools", box=ROUNDED)
    table.add_column("Name", no_wrap=True)
    table.add_column("Description")
    table.add_column("Required Params")

    for descriptor in catalog.list_tools():
        req = ", ".join(descriptor.raw_schema.get("required", []))
        table.add_row(descriptor.name, descriptor.description, req or "-")

    console.print(table)


def show_config(console: Console, wrapper: ILLM, provider_name: str) -> None:
    """Display current provider configuration."""
    content = (
        f"Provider: {provider_name}\n"
        f"Base URL: {getattr(wrapper, 'base_url', '')}\n"
        f"Model: {getattr(wrapper, 'model', '')}\n"
    )
    console.print(Panel(content, title="Configuration", box=ROUNDED))


def show_help(console: Console) -> None:
    """Print help panel."""
    console.print(
        Panel(
            "Commands\n"
            "/help      Show help\n"
            "/tools     List registered tools\n"
            "/config    Show provider configuration\n"
            "/models    List available models (use '/models refresh' to refresh)\n"
            "/model     Select or set the current model (e.g., '/model llama3.1')\n"
            "/call      Execute a tool directly (e.g., /call file {\"operation\":\"read\",\"path\":\"/tmp/a.txt\"})\n"
            "/auth      Set API key for a provider (e.g., /auth openai sk-...) or prompt if omitted\n"
            "/prefs     Show persisted preferences and API key providers\n"
            "/provider  Switch provider\n"
            "/theme     Toggle theme (dark/light)\n"
            "/colors    Toggle color output (on/off)\n"
            "/clear     Clear the screen\n"
            "/exit      Exit\n\n"
            "Type natural language instructions to let the LLM choose and run tools.",
            title="Help",
            box=ROUNDED
        )
    )


def _registry_provider_key(provider: str) -> Optional[str]:
    """
    Map CLI provider identifier to registry provider key.
    Returns None when no registry is supported (e.g., 'custom').
    """
    p = (provider or "").strip().lower()
    if p in {"deepseek", "ollama", "openai", "xai", "gemini", "openrouter", "anthropic"}:
        return p
    return None


def _render_models_table(console: Console, models: List[object]) -> None:
    table = Table(title="Available Models", box=ROUNDED)
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Family")
    table.add_column("Updated")

    for m in models or []:
        if isinstance(m, dict):
            mid = m.get("id", "-")
            name = m.get("name", "-")
            fam = m.get("family", "-")
            upd = m.get("updated_at", "-")
        else:
            mid = getattr(m, "id", None) or "-"
            name = getattr(m, "name", None) or "-"
            fam = getattr(m, "family", None) or "-"
            upd = getattr(m, "updated_at", None) or "-"
        table.add_row(str(mid), str(name), str(fam), str(upd))

    console.print(table)


def show_models(console: Console, provider: str, refresh: bool = False) -> None:
    """List models via provider getters for the current provider."""
    try:
        models = list_models(provider)
        if not models:
            console.print("[warning]No models found for this provider.[/warning]")
            return
        # Create dummy model objects for table rendering
        dummy_models = [{"id": m, "name": m, "family": "-", "updated_at": "-"} for m in models]
        _render_models_table(console, dummy_models)
    except Exception as e:
        console.print(Panel(str(e), title="Error", box=ROUNDED))


def select_model_interactive(console: Console, session, wrapper: ILLM, provider: str) -> None:
    """Prompt user to choose a model id and set it on the wrapper."""
    model_choices: List[str] = []

    try:
        model_choices = list_models(provider)
    except Exception:
        model_choices = []

    if model_choices:
        from prompt_toolkit.completion import WordCompleter
        completer = WordCompleter(model_choices, ignore_case=True, match_middle=True)
        sel = session.prompt("Model ID (tab to complete): ", completer=completer).strip()
    else:
        # Fallback: free text entry
        sel = session.prompt("Model ID: ").strip()

    if not sel:
        console.print("[warning]Model unchanged.[/warning]")
        return

    # Assign directly (OpenAICompatibleWrapper reads self.model)
    wrapper.model = sel
    print(f"Model set to '{sel}'")


def handle_call(console: Console, invoker: IToolInvocationAdapter, parts: List[str]) -> None:
    """Handle /call command."""
    if len(parts) < 3:
        print("Usage: /call <tool_name> <json_params>")
        return
    tool_name = parts[1].strip()
    params_raw = parts[2].strip()
    try:
        params = json.loads(params_raw)
    except Exception as e:
        print(f"Invalid JSON: {e}")
        return
    result = invoker.execute(tool_name, params)
    if result.ok:
        console.print(Panel(result.value, title=f"Tool Result: {tool_name}", box=ROUNDED))
    else:
        console.print(Panel(result.error, title=f"Tool Error: {tool_name}", box=ROUNDED))


def handle_auth(console: Console, session, repo: SettingsRepository, provider: str, parts: List[str]) -> None:
    """Handle /auth command."""
    if len(parts) < 2:
        print("Usage: /auth <provider> [key]")
        return
    auth_provider = parts[1].strip().lower()
    input_key = parts[2].strip() if len(parts) == 3 else None
    if not input_key:
        try:
            input_key = getpass(f"Enter API key for {auth_provider} (hidden): ").strip()
        except Exception:
            input_key = ""
    try:
        repo.set_api_key(auth_provider, input_key or None)
        # update process env for common providers
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "xai": "XAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        env_name = env_map.get(auth_provider)
        if env_name:
            if input_key:
                os.environ[env_name] = input_key
            elif env_name in os.environ:
                del os.environ[env_name]
        print(f"API key {'updated' if input_key else 'cleared'} for {auth_provider}.")
    except Exception as e:
        print(f"Auth update failed: {e}")


def handle_prefs(console: Console, repo: SettingsRepository) -> None:
    """Handle /prefs command."""
    try:
        data = {"prefs": repo.all_prefs(), "api_keys": list(repo.all_api_keys().keys())}
        console.print(Panel(json.dumps(data, ensure_ascii=False, indent=2), title="Preferences", box=ROUNDED))
    except Exception as e:
        console.print(Panel(f"Failed to load prefs: {e}", title="Preferences", box=ROUNDED))


def handle_theme(console, repo: SettingsRepository, theme: str) -> str:
    """Handle /theme command."""
    new_theme = "light" if theme == "dark" else "dark"
    try:
        repo.set_pref("cli_theme", new_theme)
    except Exception:
        pass
    return new_theme


def handle_colors(console, repo: SettingsRepository, use_color: bool) -> bool:
    """Handle /colors command."""
    new_color = not use_color
    try:
        repo.set_pref("cli_color", "1" if new_color else "0")
    except Exception:
        pass
    return new_color


def handle_clear(console) -> None:
    """Handle /clear command."""
    console.clear()


def handle_exit(console) -> None:
    """Handle /exit command."""
    console.print("\n[warning]Exiting...[/warning]")
    sys.exit(0)


def handle_provider(console, session, repo: SettingsRepository, current_provider: str, tools, wrapper: ILLM) -> tuple[str, ILLM]:
    """Handle /provider command."""
    new_provider = select_provider(console, session)
    new_wrapper = instantiate_wrapper(new_provider, console, session, repo=repo)
    try:
        repo.set_pref("last_provider", new_provider)
    except Exception:
        pass
    # Restore model if set
    saved_model = repo.get_pref(f"model.{new_provider}")
    if saved_model:
        new_wrapper.model = saved_model
    # Re-register tools
    for tool in tools:
        new_wrapper.register_tool(tool)
    return new_provider, new_wrapper