"""
Console utilities for CLI.
"""

from typing import Optional
from rich.console import Console
from rich.theme import Theme
import os
import sys

_CURRENT_THEME_NAME = "dark"


def _build_theme(theme_name: str) -> Theme:
    if theme_name == "light":
        return Theme(
            {
                "primary": "white",
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
        return Theme(
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


def _should_enable_color(enable: Optional[bool]):
    """
    Compute effective color enablement, force_terminal, and color_system.

    Rules:
    - Respect NO_COLOR unless AGENT_CLI_FORCE_COLOR is set
    - When enable is None: auto-detect via isatty
    - If enable is True: prefer colors; If False: disable and force_terminal=False
    - force_terminal=True when desired and attached to a TTY, or when AGENT_CLI_FORCE_COLOR is set
    """
    force_color = (os.getenv("AGENT_CLI_FORCE_COLOR") or "").lower() in ("1", "true", "yes", "on")
    no_color_env = os.getenv("NO_COLOR") is not None
    tty = bool(getattr(sys.stdout, "isatty", lambda: False)() or getattr(sys.stderr, "isatty", lambda: False)())

    if enable is None:
        desired_raw = tty
    else:
        desired_raw = bool(enable)

    # Apply NO_COLOR unless forced
    desired = desired_raw and (not no_color_env or force_color)

    # Compute force_terminal with explicit overrides
    if enable is False:
        force_terminal = False
    elif force_color:
        force_terminal = True
    else:
        force_terminal = bool(desired and tty and not no_color_env)

    color_system = "auto" if desired else None
    return desired, force_terminal, color_system


def make_console(theme_name: str, use_color: Optional[bool] = True) -> Console:
    """Create a Rich console with the selected theme and color policy."""
    global _CURRENT_THEME_NAME
    _CURRENT_THEME_NAME = "light" if theme_name == "light" else "dark"

    theme = _build_theme(_CURRENT_THEME_NAME)
    desired, force_terminal, color_system = _should_enable_color(use_color)

    return Console(
        theme=theme,
        no_color=not desired,
        color_system=color_system,
        force_terminal=force_terminal,
        markup=True,       # render style tags like [warning]...[/warning]
        emoji=True,
        highlight=False,
    )


def rebuild_console(enable: Optional[bool] = None) -> Console:
    """
    Rebuild console with the last theme and desired color state.

    Args:
        enable: True to force color, False to force no color, None to auto-detect.
    """
    return make_console(_CURRENT_THEME_NAME, use_color=enable)


def clear(console: Console) -> None:
    """Clear the console screen."""
    console.clear()


__all__ = ["make_console", "rebuild_console", "clear"]