"""
Interactive CLI for Agent Tools with OpenAI-compatible providers (DeepSeek, Ollama, Custom).

Features:
- Provider selection (DeepSeek, Ollama, Custom OpenAI-compatible)
- Local Ollama support out of the box (http://localhost:11434/v1)
- Color themes (dark/light) and rich panels for output
- Tool listing and live execution via wrappers
- Smooth prompt experience using prompt_toolkit

Commands:
  /help    Show help
  /tools   List registered tools
  /config  Show current provider configuration
  /theme   Toggle theme (dark/light)
  /clear   Clear the screen
  /exit    Exit

Run:
  python -m agent_tools.src.cli.app
  or
  python agent_tools/src/cli/app.py
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

# Ensure 'src' sibling packages are importable when run directly
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.completion import WordCompleter

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from rich.box import ROUNDED

from wrappers.deepseek_wrapper import DeepseekToolWrapper
from wrappers.ollama_wrapper import OllamaWrapper
from wrappers.openai_compatible import OpenAICompatibleWrapper
from src.tools.tool_manager import ToolManager


def make_console(theme_name: str) -> Console:
    """Create a Rich console with the selected theme."""
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
    return Console(theme=theme)


def select_provider(console: Console, session: PromptSession) -> str:
    """Prompt to select a provider."""
    console.print(
        Panel(
            "[box_title]Select Provider[/box_title]\n\n"
            "1) DeepSeek (hosted)\n"
            "2) Ollama (local)\n"
            "3) Custom (OpenAI-compatible)\n",
            title="Provider",
            box=ROUNDED,
            border_style="accent",
        )
    )
    completer = WordCompleter(["1", "2", "3"], ignore_case=True)
    while True:
        choice = session.prompt("Choose [1-3]: ", completer=completer).strip()
        if choice in {"1", "2", "3"}:
            return {"1": "deepseek", "2": "ollama", "3": "custom"}[choice]
        console.print("[warning]Invalid option. Enter 1, 2, or 3.[/warning]")


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


def instantiate_wrapper(provider: str, console: Console, session: PromptSession) -> OpenAICompatibleWrapper:
    """Create the appropriate wrapper instance based on selection."""
    if provider == "deepseek":
        wrapper = DeepseekToolWrapper()
        console.print("[success]Using DeepSeek provider[/success]")
        return wrapper
    if provider == "ollama":
        wrapper = OllamaWrapper()
        console.print("[success]Using Ollama local provider[/success] (http://localhost:11434/v1)")
        return wrapper
    # custom
    base_url, model, api_key = input_custom_config(console, session)
    wrapper = OpenAICompatibleWrapper(api_key=api_key or None, base_url=base_url, model=model)
    console.print(f"[success]Using Custom provider[/success] [muted]{base_url} model={model}[/muted]")
    return wrapper


def list_tools(console: Console, manager: ToolManager) -> None:
    """Render a table of registered tools."""
    table = Table(title="Registered Tools", box=ROUNDED, border_style="accent", title_style="box_title")
    table.add_column("Name", style="primary", no_wrap=True)
    table.add_column("Description", style="muted")
    table.add_column("Required Params", style="accent")

    for info in manager.list_tools():
        req = ", ".join(info["input_schema"].get("required", []))
        table.add_row(info["name"], info["description"], req or "-")

    console.print(table)


def show_config(console: Console, wrapper: OpenAICompatibleWrapper, provider_name: str) -> None:
    """Display current provider configuration."""
    content = (
        f"[primary]Provider:[/primary] {provider_name}\n"
        f"[primary]Base URL:[/primary] {getattr(wrapper, 'base_url', '')}\n"
        f"[primary]Model:[/primary] {getattr(wrapper, 'model', '')}\n"
    )
    console.print(Panel(content, title="Configuration", box=ROUNDED, border_style="accent", title_style="box_title"))


def show_help(console: Console) -> None:
    """Print help panel."""
    console.print(
        Panel(
            "[box_title]Commands[/box_title]\n"
            "/help   Show help\n"
            "/tools  List registered tools\n"
            "/config Show provider configuration\n"
            "/theme  Toggle theme (dark/light)\n"
            "/clear  Clear the screen\n"
            "/exit   Exit\n\n"
            "Type natural language instructions to let the LLM choose and run tools.",
            title="Help",
            box=ROUNDED,
            border_style="accent",
            title_style="box_title",
        )
    )


def run() -> None:
    """Main interactive loop."""
    theme = os.getenv("CLI_THEME", "dark").lower()
    console = make_console("light" if theme in ("light", "white") else "dark")
    session = PromptSession(history=InMemoryHistory())

    console.print(
        Panel(
            "[box_title]Agent Tools CLI[/box_title]\n"
            "OpenAI-compatible with tool-use and local Ollama support.",
            title="Welcome",
            box=ROUNDED,
            border_style="accent",
            title_style="box_title",
        )
    )

    provider = select_provider(console, session)
    wrapper = instantiate_wrapper(provider, console, session)

    # Register default tools
    manager = ToolManager(register_defaults=True)
    for tool in manager.tools.values():
        wrapper.register_tool(tool)

    show_help(console)

    completer = WordCompleter(
        ["/help", "/tools", "/config", "/theme", "/clear", "/exit"],
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

            if cmd == "/help":
                show_help(console)
                continue
            if cmd == "/tools":
                list_tools(console, manager)
                continue
            if cmd == "/config":
                show_config(console, wrapper, provider_name=provider)
                continue
            if cmd == "/theme":
                theme = "light" if theme == "dark" else "dark"
                console = make_console(theme)
                console.print(f"[success]Theme switched to {theme}[/success]")
                continue
            if cmd == "/clear":
                console.clear()
                continue
            if cmd == "/exit":
                console.print("[warning]Goodbye.[/warning]")
                break

            # Execute via wrapper
            console.print(Panel("[accent]Thinking...[/accent]", box=ROUNDED, border_style="accent"))
            try:
                result = wrapper.execute(cmd)
                console.print(
                    Panel(
                        result,
                        title="LLM Response",
                        box=ROUNDED,
                        border_style="accent",
                        title_style="box_title",
                    )
                )
            except Exception as e:
                console.print(Panel(f"[error]{str(e)}[/error]", title="Error", box=ROUNDED, border_style="error"))


if __name__ == "__main__":
    run()