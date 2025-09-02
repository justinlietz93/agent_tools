"""
DeepseekToolWrapper

Provider-specific specialization for the DeepSeek OpenAI-compatible API that
preserves the legacy behavior and output format used across tests:

- Reasoning:\\n{reasoning}\\n\\nTool Call:\\n{json}\\n\\nResult:\\n{tool_result}

This class simply configures OpenAICompatibleWrapper with DeepSeek defaults and
can be instantiated directly or via DI composition.

Usage:
    from src.infrastructure.llm.deepseek_wrapper import DeepseekToolWrapper
"""

from __future__ import annotations

import os

from .openai_compatible import OpenAICompatibleWrapper


class DeepseekToolWrapper(OpenAICompatibleWrapper):
    """
    Thin wrapper over OpenAICompatibleWrapper configured for DeepSeek.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "deepseek",
            base_url=base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner"),
        )


__all__ = ["DeepseekToolWrapper"]