"""
Compatibility shim to preserve legacy import path.

Do NOT implement wrapper logic here. This file simply re-exports the
DeepseekToolWrapper from the canonical wrappers package.

Original imports in tests:
    from src.infrastructure.tools.deepseek_wrapper import DeepseekToolWrapper
"""

from ..wrappers.deepseek_wrapper import DeepseekToolWrapper  # noqa: F401