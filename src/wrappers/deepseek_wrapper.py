from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

from .openai_compatible import OpenAICompatibleWrapper


class DeepseekToolWrapper(OpenAICompatibleWrapper):
    """
    DeepSeek-specialized wrapper that delegates to the OpenAI-compatible implementation.

    Defaults:
      - base_url: https://api.deepseek.com
      - model: deepseek-reasoner
      - api_key: DEEPSEEK_API_KEY (from environment)

    Interface preserved:
      - register_tool(tool)
      - execute(user_input: str) -> str
      - Tool listing and prompt construction via the shared base
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        load_dotenv()
        super().__init__(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner"),
        )