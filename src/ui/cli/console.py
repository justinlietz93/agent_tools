"""
Console utilities for CLI.
"""

from rich.console import Console
from rich.theme import Theme


def make_console(theme_name: str, use_color: bool = True) -> Console:
    """Create a Rich console with the selected theme and color policy."""
    if theme_name == "light":
        theme = Theme(
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