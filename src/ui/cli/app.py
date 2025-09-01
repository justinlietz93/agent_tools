"""
Interactive CLI for Agent Tools with OpenAI-compatible providers (DeepSeek, Ollama, Custom).

Features:
- Provider selection (DeepSeek, Ollama, Custom OpenAI-compatible)
- Local Ollama support out of the box (http://localhost:11434/v1)
- Color themes (dark/light) and rich panels for output
- Tool listing and live execution via wrappers
- Smooth prompt experience using prompt_toolkit
- Model selection and listing via providers registry for supported providers

Commands:
  /help      Show help
  /tools     List registered tools
  /config    Show current provider configuration
  /models    List available models for current provider (use "/models refresh" to refresh)
  /model     Select or set the current model ("/model <id>" or interactive)
  /provider  Switch provider (restarts wrapper)
  /theme     Toggle theme (dark/light)
  /clear     Clear the screen
  /exit      Exit

Run:
  python -m src.ui.cli.app
  or
  python src/ui/cli/app.py
"""

from __future__ import annotations

import os
import sys
import json
from getpass import getpass
from typing import Optional, Tuple, List

# Ensure project root is importable when run directly
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.completion import WordCompleter

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from rich.box import ROUNDED

from .console import make_console
from .provider import select_provider, instantiate_wrapper
from .handlers import (
    list_tools, show_config, show_help, show_models, select_model_interactive,
    handle_call, handle_auth, handle_prefs, handle_theme, handle_colors,
    handle_clear, handle_exit, handle_provider
)
from src.interfaces.services.llm import ILLM
from src.api.di.cli_composition import build_settings_repository, build_tool_catalog, build_tool_invoker, get_registered_tools
from src.interfaces.repositories.settings_repository import SettingsRepository


def run() -> None:
    """Main interactive loop."""
    settings_repo: SettingsRepository = build_settings_repository()

    # Theme and color policy (ENV overrides DB, then defaults)
    theme = (os.getenv("CLI_THEME") or (settings_repo.get_pref("cli_theme") or "dark")).lower()
    # Conservative default: disable colors unless explicitly enabled
    use_color_env = os.getenv("CLI_COLOR")
    if use_color_env is not None:
        use_color = use_color_env.lower() in ("1", "true", "yes", "on")
    else:
        use_color = (settings_repo.get_pref("cli_color") or "").lower() in ("1", "true", "yes", "on")
    console = make_console("light" if theme in ("light", "white") else "dark", use_color=use_color)
    session = PromptSession(history=InMemoryHistory())

    console.print(
        Panel(
            "Agent Tools CLI\n"
            "OpenAI-compatible with tool-use and local Ollama support.",
            title="Welcome",
            box=ROUNDED
        )
    )

    provider = select_provider(console, session)
    wrapper: ILLM = instantiate_wrapper(provider, console, session, repo=settings_repo)
    try:
        settings_repo.set_pref("last_provider", provider)
    except Exception:
        pass
    # Restore model if set (also done in instantiate_wrapper but safe)
    saved_model = settings_repo.get_pref(f"model.{provider}")
    if saved_model:
        wrapper.model = saved_model

    # Register default tools
    tools = get_registered_tools()
    catalog = build_tool_catalog()
    invoker = build_tool_invoker()
    for tool in tools:
        wrapper.register_tool(tool)

    show_help(console)

    completer = WordCompleter(
        ["/help", "/tools", "/config", "/models", "/model", "/call", "/auth", "/prefs", "/provider", "/theme", "/colors", "/clear", "/exit"],
        ignore_case=True,
        match_middle=True,
    )

    with patch_stdout():
        while True:
            try:
                user_input = session.prompt("> ", completer=completer)
            except (KeyboardInterrupt, EOFError):
                console.print("\n[warning]Exiting...[/warning]")
                break

            cmd = user_input.strip()
            if not cmd:
                continue

            # Built-in commands
            if cmd == "/help":
                show_help(console)
                continue
            if cmd == "/tools":
                list_tools(console, catalog)
                continue
            if cmd == "/config":
                show_config(console, wrapper, provider_name=provider)
                continue
            if cmd.startswith("/models"):
                refresh = any(tok.lower().startswith("ref") for tok in cmd.split()[1:])
                show_models(console, provider, refresh=refresh)
                continue
            if cmd.startswith("/model"):
                parts = cmd.split(maxsplit=1)
                if len(parts) == 2 and parts[1].strip():
                    wrapper.model = parts[1].strip()
                    print(f"Model set to '{wrapper.model}'")
                    try:
                        settings_repo.set_pref(f"model.{provider}", wrapper.model)
                    except Exception:
                        pass
                else:
                    select_model_interactive(console, session, wrapper, provider)
                    try:
                        if getattr(wrapper, "model", None):
                            settings_repo.set_pref(f"model.{provider}", wrapper.model)
                    except Exception:
                        pass
                continue
            if cmd.startswith("/call"):
                parts = cmd.split(maxsplit=2)
                handle_call(console, invoker, parts)
                continue
            if cmd.startswith("/auth"):
                parts = cmd.split(maxsplit=2)
                handle_auth(console, session, settings_repo, provider, parts)
                # If updating the current provider, re-instantiate wrapper to pick changes
                auth_provider = parts[1].strip().lower() if len(parts) > 1 else ""
                if auth_provider == provider:
                    wrapper: ILLM = instantiate_wrapper(provider, console, session, repo=settings_repo)
                    for tool in tools:
                        wrapper.register_tool(tool)
                continue
            if cmd == "/prefs":
                handle_prefs(console, settings_repo)
                continue
            if cmd == "/provider":
                provider, wrapper = handle_provider(console, session, settings_repo, provider, tools, wrapper)
                continue
            if cmd == "/theme":
                theme = handle_theme(console, settings_repo, theme)
                console = make_console(theme, use_color=use_color)
                print(f"Theme switched to {theme}")
                continue
            if cmd == "/colors":
                use_color = handle_colors(console, settings_repo, use_color)
                console = make_console(theme, use_color=use_color)
                print(f"Colors {'enabled' if use_color else 'disabled'}")
                continue
            if cmd == "/clear":
                handle_clear(console)
                continue
            if cmd == "/exit":
                handle_exit(console)

            # Default: execute as natural language instruction
            try:
                result = wrapper.execute(user_input)
                console.print(Panel(result, title="Response", box=ROUNDED))
            except Exception as e:
                console.print(Panel(f"Error: {str(e)}", title="Error", box=ROUNDED))


if __name__ == "__main__":
    run()