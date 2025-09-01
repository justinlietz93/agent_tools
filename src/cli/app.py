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
  python -m src.cli.app
  or
  python src/cli/app.py
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

from src.wrappers.deepseek_wrapper import DeepseekToolWrapper
from src.wrappers.ollama_wrapper import OllamaWrapper
from src.wrappers.openai_compatible import OpenAICompatibleWrapper
from src.tools.tool_manager import ToolManager
from src.providers.base.repositories.model_registry import ModelRegistryRepository
from src.settings import get_settings_repo


def make_console(theme_name: str, use_color: bool = True) -> Console:
    """Create a Rich console with the selected theme and color policy."""
    if theme_name == "light":
        theme = Theme(
            {
                "primary": "black",
                "accent": "dark_green",
                "warning": "dark_orange",
                "error": "red",
                "success": "green",
                "muted": "grey42",
                "box_title": "bold black",
            }
        )
    else:
        # dark
        theme = Theme(
            {
                "primary": "white",
                "accent": "cyan",
                "warning": "yellow",
                "error": "bold red",
                "success": "green",
                "muted": "grey70",
                "box_title": "bold cyan",
            }
        )
    # no_color=True disables ANSI codes for environments that don't support them
    return Console(
        theme=theme,
        no_color=not use_color,
        color_system=("standard" if use_color else None),
        force_terminal=use_color,
        markup=False,
    )


def select_provider(console: Console, session: PromptSession) -> str:
    """Prompt to select a provider."""
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


def input_custom_config(console: Console, session: PromptSession) -> Tuple[str, str, str]:
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


def instantiate_wrapper(provider: str, console: Console, session: PromptSession, repo=None) -> OpenAICompatibleWrapper:
    """Create the appropriate wrapper instance based on selection."""
    repo = repo or get_settings_repo()

    if provider == "deepseek":
        wrapper = DeepseekToolWrapper()
        print("Using DeepSeek provider")
        # Restore saved model for this provider if present
        saved = repo.get_pref(f"model.{provider}")
        if saved:
            wrapper.model = saved
            print(f"Restored model to '{saved}'")
        return wrapper

    elif provider == "ollama":
        wrapper = OllamaWrapper()
        print("Using Ollama local provider (http://localhost:11434/v1)")
        saved = repo.get_pref(f"model.{provider}")
        if saved:
            wrapper.model = saved
            print(f"Restored model to '{saved}'")
        return wrapper

    elif provider == "openai":
        # Prefer env; then DB; else prompt (no placeholder fallback)
        key = os.getenv("OPENAI_API_KEY") or (repo.get_api_key("openai") or "")
        if not key:
            try:
                key = getpass("Enter OPENAI_API_KEY (input hidden, press Enter to cancel): ").strip()
            except Exception:
                key = ""
        if not key:
            print("OpenAI API key not provided. Returning to provider selection.")
            # Re-enter provider selection flow
            new_provider = select_provider(console, session)
            return instantiate_wrapper(new_provider, console, session, repo=repo)

        # Persist and set env for process-wide usage
        try:
            repo.set_api_key("openai", key)
        except Exception:
            pass
        os.environ["OPENAI_API_KEY"] = key

        wrapper = OpenAICompatibleWrapper(
            api_key=key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
        print("Using OpenAI provider https://api.openai.com/v1")
        saved = repo.get_pref(f"model.{provider}")
        if saved:
            wrapper.model = saved
            print(f"Restored model to '{saved}'")
        return wrapper

    # custom (OpenAI-compatible)
    base_url, model, api_key = input_custom_config(console, session)
    wrapper = OpenAICompatibleWrapper(api_key=api_key or None, base_url=base_url, model=model)
    print(f"Using Custom provider {base_url} model={model}")
    saved = repo.get_pref(f"model.{provider}")
    if saved:
        wrapper.model = saved
        print(f"Restored model to '{saved}'")
    return wrapper


def list_tools(console: Console, manager: ToolManager) -> None:
    """Render a table of registered tools."""
    table = Table(title="Registered Tools", box=ROUNDED)
    table.add_column("Name", no_wrap=True)
    table.add_column("Description")
    table.add_column("Required Params")

    for info in manager.list_tools():
        req = ", ".join(info["input_schema"].get("required", []))
        table.add_row(info["name"], info["description"], req or "-")

    console.print(table)


def show_config(console: Console, wrapper: OpenAICompatibleWrapper, provider_name: str) -> None:
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
        mid = getattr(m, "id", None) or "-"
        name = getattr(m, "name", None) or "-"
        fam = getattr(m, "family", None) or "-"
        upd = getattr(m, "updated_at", None) or "-"
        table.add_row(str(mid), str(name), str(fam), str(upd))

    console.print(table)


def show_models(console: Console, provider: str, refresh: bool = False) -> None:
    """List models via ModelRegistryRepository for the current provider."""
    key = _registry_provider_key(provider)
    if not key:
        console.print("[warning]Model registry not supported for this provider (custom connection).[/warning]")
        return
    try:
        repo = ModelRegistryRepository()
        snap = repo.list_models(key, refresh=refresh)
        if not snap.models:
            console.print("[warning]No models found in registry. Try '/models refresh' if applicable.[/warning]")
            return
        _render_models_table(console, snap.models)
    except Exception as e:
        console.print(Panel(str(e), title="Error", box=ROUNDED))


def select_model_interactive(console: Console, session: PromptSession, wrapper: OpenAICompatibleWrapper, provider: str) -> None:
    """Prompt user to choose a model id and set it on the wrapper."""
    key = _registry_provider_key(provider)
    model_choices: List[str] = []

    if key:
        try:
            repo = ModelRegistryRepository()
            snap = repo.list_models(key, refresh=False)
            model_choices = [str(getattr(m, "id", "") or getattr(m, "name", "")) for m in snap.models if getattr(m, "id", None) or getattr(m, "name", None)]
        except Exception:
            model_choices = []

    if model_choices:
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


def run() -> None:
    """Main interactive loop."""
    repo = get_settings_repo()

    # Theme and color policy (ENV overrides DB, then defaults)
    theme = (os.getenv("CLI_THEME") or (repo.get_pref("cli_theme") or "dark")).lower()
    # Conservative default: disable colors unless explicitly enabled
    use_color_env = os.getenv("CLI_COLOR")
    if use_color_env is not None:
        use_color = use_color_env.lower() in ("1", "true", "yes", "on")
    else:
        use_color = (repo.get_pref("cli_color") or "").lower() in ("1", "true", "yes", "on")
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
    wrapper = instantiate_wrapper(provider, console, session, repo=repo)
    try:
        repo.set_pref("last_provider", provider)
    except Exception:
        pass
    # Restore model if set (also done in instantiate_wrapper but safe)
    saved_model = repo.get_pref(f"model.{provider}")
    if saved_model:
        wrapper.model = saved_model

    # Register default tools
    manager = ToolManager(register_defaults=True)
    for tool in manager.tools.values():
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
                list_tools(console, manager)
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
                        repo.set_pref(f"model.{provider}", wrapper.model)
                    except Exception:
                        pass
                else:
                    select_model_interactive(console, session, wrapper, provider)
                    try:
                        if getattr(wrapper, "model", None):
                            repo.set_pref(f"model.{provider}", wrapper.model)
                    except Exception:
                        pass
                continue
            if cmd.startswith("/call"):
                # Direct tool execution: /call <tool_name> <json_params>
                parts = cmd.split(maxsplit=2)
                if len(parts) < 3:
                    print("Usage: /call <tool_name> {\"param\":\"value\", ...}")
                    continue
                tool_name = parts[1].strip()
                params_raw = parts[2].strip()
                try:
                    params = json.loads(params_raw)
                except Exception as e:
                    print(f"Invalid JSON: {e}")
                    continue
                try:
                    tool = manager.get_tool(tool_name)
                except KeyError as e:
                    print(str(e))
                    continue
                try:
                    # Prefer run(input_dict) signature
                    result = tool.run(params)  # type: ignore[arg-type]
                except TypeError:
                    # Fallback for tools using run(tool_call_id, **kwargs)
                    try:
                        result = tool.run("cli", **(params if isinstance(params, dict) else {}))  # type: ignore[call-arg]
                    except Exception as e:
                        print(f"Error executing tool: {e}")
                        continue
                try:
                    rendered = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)
                except Exception:
                    rendered = str(result)
                console.print(Panel(rendered, title=f"Tool Result: {tool_name}", box=ROUNDED))
                continue
            if cmd.startswith("/auth"):
                # /auth <provider> [key]
                parts = cmd.split(maxsplit=2)
                if len(parts) < 2:
                    print("Usage: /auth <provider> [key]")
                    continue
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
                    # If updating the current provider, re-instantiate wrapper to pick changes
                    if auth_provider == provider:
                        wrapper = instantiate_wrapper(provider, console, session, repo=repo)
                        for tool in manager.tools.values():
                            wrapper.register_tool(tool)
                except Exception as e:
                    print(f"Auth update failed: {e}")
                continue
            if cmd == "/prefs":
                try:
                    data = {"prefs": repo.all_prefs(), "api_keys": list(repo.all_api_keys().keys())}
                    console.print(Panel(json.dumps(data, ensure_ascii=False, indent=2), title="Preferences", box=ROUNDED))
                except Exception as e:
                    console.print(Panel(f"Failed to load prefs: {e}", title="Preferences", box=ROUNDED))
                continue
            if cmd == "/provider":
                provider = select_provider(console, session)
                wrapper = instantiate_wrapper(provider, console, session, repo=repo)
                # Re-register tools on the new wrapper
                for tool in manager.tools.values():
                    wrapper.register_tool(tool)
                try:
                    repo.set_pref("last_provider", provider)
                except Exception:
                    pass
                print("Provider switched.")
                continue
            if cmd == "/theme":
                theme = "light" if theme == "dark" else "dark"
                console = make_console(theme, use_color=use_color)
                try:
                    repo.set_pref("cli_theme", theme)
                except Exception:
                    pass
                print(f"Theme switched to {theme}")
                continue
            if cmd.startswith("/colors"):
                parts = cmd.split(maxsplit=1)
                if len(parts) == 2 and parts[1].lower() in ("on", "off"):
                    use_color = parts[1].lower() == "on"
                else:
                    use_color = not use_color
                console = make_console(theme, use_color=use_color)
                status = "enabled" if use_color else "disabled"
                try:
                    repo.set_pref("cli_color", "on" if use_color else "off")
                except Exception:
                    pass
                print(f"Colors {status}")
                continue
            if cmd == "/clear":
                console.clear()
                continue
            if cmd == "/exit":
                console.print("[warning]Goodbye.[/warning]")
                break

            # Execute via wrapper
            console.print(Panel("Thinking...", box=ROUNDED))
            try:
                result = wrapper.execute(cmd)
                console.print(
                    Panel(
                        result,
                        title="LLM Response",
                        box=ROUNDED,
                        border_style="accent",
                    )
                )
            except Exception as e:
                console.print(Panel(f"[error]{str(e)}[/error]", title="Error", box=ROUNDED, border_style="error"))


if __name__ == "__main__":
    run()