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
import subprocess
import shutil
import shlex
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
from rich.live import Live

from .console import make_console, rebuild_console
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
    # Optional: launch in a dedicated terminal window to ensure full ANSI support.
    # Enabled by default unless CLI_SPAWN_TERMINAL is set to 0/false.
    try:
        spawn_flag = (os.getenv("CLI_SPAWN_TERMINAL") or "1").lower() in ("1", "true", "yes", "on")
        if not os.getenv("CLI_CHILD") and spawn_flag:
            PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
            # Ensure we execute from the project root so `python -m src.ui.cli.app` resolves correctly.
            # On failure, keep the child window open for inspection.
            cmd = f'cd "{PROJECT_ROOT}" && {shlex.quote(sys.executable)} -m src.ui.cli.app || {{ echo "CLI exited with status $?" ; read -n 1 -s -r -p "Press any key to close"; }}'
            candidates = [
                ("gnome-terminal", ["gnome-terminal", "--", "bash", "-lc", cmd]),
                ("wezterm", ["wezterm", "start", "--", "bash", "-lc", cmd]),
                ("kitty", ["kitty", "bash", "-lc", cmd]),
                ("alacritty", ["alacritty", "-e", "bash", "-lc", cmd]),
                ("konsole", ["konsole", "-e", "bash", "-lc", cmd]),
                ("tilix", ["tilix", "-e", "bash", "-lc", cmd]),
                ("xfce4-terminal", ["xfce4-terminal", "-e", "bash", "-lc", cmd]),
                ("xterm", ["xterm", "-e", "bash", "-lc", cmd]),
                ("foot", ["foot", "bash", "-lc", cmd]),
            ]
            for name, argv in candidates:
                if shutil.which(name):
                    env = os.environ.copy()
                    env["CLI_CHILD"] = "1"
                    # Hint truecolor for better rendering if not already set
                    if os.getenv("COLORTERM") is None:
                        env["COLORTERM"] = "truecolor"
                    subprocess.Popen(argv, env=env, cwd=PROJECT_ROOT)
                    return  # Parent exits; child runs CLI in dedicated terminal
    except Exception:
        # If terminal spawn fails, continue in current terminal
        pass

    settings_repo: SettingsRepository = build_settings_repository()

    # Theme and color policy (ENV overrides DB; default ON in TTY/truecolor terminals)
    theme = (os.getenv("CLI_THEME") or (settings_repo.get_pref("cli_theme") or "dark")).lower()
    use_color_env = os.getenv("CLI_COLOR")
    if use_color_env is not None:
        use_color = use_color_env.lower() in ("1", "true", "yes", "on")
    else:
        pref = str(settings_repo.get_pref("cli_color") or "").lower()
        if pref in ("1", "true", "yes", "on"):
            use_color = True
        elif pref in ("0", "false", "no", "off"):
            use_color = False
        else:
            # Auto-detect: prefer color when attached to a TTY or COLORTERM is set
            use_color = sys.stdout.isatty() or sys.stderr.isatty() or bool(os.getenv("COLORTERM"))
    console = make_console("light" if theme in ("light", "white") else "dark", use_color=use_color)
    session = PromptSession(history=InMemoryHistory())

    try:
        console.clear()
    except Exception:
        pass
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

    # Clear after provider selection before rendering main help
    try:
        console.clear()
    except Exception:
        pass
    show_help(console)

    completer = WordCompleter(
        ["/help", "/tools", "/config", "/models", "/model", "/call", "/auth", "/prefs", "/provider", "/theme", "/colors", "/clear", "/streamtest", "/exit"],
        ignore_case=True,
        match_middle=True,
    )

    while True:
        try:
            with patch_stdout():
                user_input = session.prompt("> ", completer=completer)
        except (KeyboardInterrupt, EOFError):
            console.print("\nExiting...", style="yellow")
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
        if cmd.startswith("/provider"):
            parts = cmd.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                requested = parts[1].strip().lower()
                if requested in {"deepseek", "ollama", "openai", "custom"}:
                    # Direct provider switch without interactive prompt
                    try:
                        console.clear()
                    except Exception:
                        pass
                    provider = requested
                    wrapper = instantiate_wrapper(provider, console, session, repo=settings_repo)
                    try:
                        settings_repo.set_pref("last_provider", provider)
                    except Exception:
                        pass
                    # Restore model if available
                    saved_model = settings_repo.get_pref(f"model.{provider}")
                    if saved_model:
                        wrapper.model = saved_model
                    # Re-register tools on new wrapper
                    for tool in tools:
                        wrapper.register_tool(tool)
                    console.print(Panel(f"Switched provider to [accent]{provider}[/accent]", title="Provider", box=ROUNDED))
                    continue
            # Fallback to interactive selector
            try:
                console.clear()
            except Exception:
                pass
            provider, wrapper = handle_provider(console, session, settings_repo, provider, tools, wrapper)
            continue
        if cmd == "/theme":
            theme = handle_theme(console, settings_repo, theme)
            console = make_console(theme, use_color=use_color)
            try:
                console.clear()
            except Exception:
                pass
            console.print(Panel(f"Theme switched to [accent]{theme}[/accent]", title="Theme", box=ROUNDED))
            continue
        if cmd == "/colors":
            use_color = handle_colors(console, settings_repo, use_color)
            console = rebuild_console(enable=use_color)
            try:
                console.clear()
            except Exception:
                pass
            # Re-render the main help/home panel to reflect color changes immediately
            show_help(console)
            continue
        if cmd == "/clear":
            handle_clear(console)
            continue
        if cmd.startswith("/streamtest"):
            parts = cmd.split(maxsplit=1)
            demo_text = parts[1] if len(parts) == 2 else "hello"
            try:
                # Clear page before rendering streaming test
                try:
                    console.clear()
                except Exception:
                    pass
                # Stream directly to stdout to avoid redraw artifacts
                out = getattr(sys, "__stdout__", sys.stdout)
                console.print("[accent]Response (streaming):[/accent]")
                def _on_delta(s: str) -> None:
                    out.write(s)
                    out.flush()
                final = wrapper.stream_and_collect(demo_text, _on_delta)
                out.write("\n")
                out.flush()
                console.print(Panel(final, title="Response", box=ROUNDED))
            except Exception as e:
                console.print(Panel(f"Stream test error: {e}", title="Error", box=ROUNDED))
            continue
        if cmd == "/exit":
            handle_exit(console)

        # Default: execute as natural language instruction
        try:
            # Attempt streaming first; adapter will fall back to non-streaming execute() if unsupported
            streamed_parts: list[str] = []

            # Stream directly to the real stdout to avoid redraw artifacts from prompt_toolkit/Rich Live
            out = getattr(sys, "__stdout__", sys.stdout)
            # Page refresh UX: clear the screen before streaming header
            try:
                console.clear()
            except Exception:
                pass
            try:
                out.write("\x1b[2J\x1b[H")  # ANSI clear screen + cursor home
            except Exception:
                pass
            out.write("Response (streaming):\n")
            out.flush()

            def on_delta(s: str) -> None:
                streamed_parts.append(s)
                out.write(s)
                out.flush()

            # Stream tokens directly; then print a newline so the prompt resumes cleanly
            final = wrapper.stream_and_collect(user_input, on_delta)
            out.write("\n")
            out.flush()

            # Render the final, fully formatted result (Reasoning/Response/Tool Call)
            try:
                console.clear()
            except Exception:
                pass
            console.print(Panel(final, title="Response", box=ROUNDED))
        except Exception:
            # Fallback to non-streaming execute on any error
            try:
                result = wrapper.execute(user_input)
                console.print(Panel(result, title="Response", box=ROUNDED))
            except Exception as e2:
                console.print(Panel(f"Error: {str(e2)}", title="Error", box=ROUNDED))


if __name__ == "__main__":
    run()