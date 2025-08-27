"""
Ollama LLM wrapper.

Defaults to local OpenAI-compatible endpoint:
  base_url: http://localhost:11434/v1
  model:    OLLAMA_MODEL (default: llama3.1)

Examples:
- Use defaults (local server, default model):
    wrapper = OllamaWrapper()

- Custom model:
    wrapper = OllamaWrapper(model="qwen2.5-coder:7b")

- Custom base URL (remote ollama):
    wrapper = OllamaWrapper(base_url="http://remote-host:11434/v1", model="llama3.1")

Delegates all functionality to [OpenAICompatibleWrapper](agent_tools/src/wrappers/openai_compatible.py:21).
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

from .openai_compatible import OpenAICompatibleWrapper


class OllamaWrapper(OpenAICompatibleWrapper):
    """
    Preconfigured OpenAI-compatible wrapper for Ollama.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        load_dotenv()

        # Ollama typically ignores API keys; we pass a placeholder by default
        super().__init__(
            api_key=api_key or os.getenv("OLLAMA_API_KEY", "ollama"),
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            model=model or os.getenv("OLLAMA_MODEL", "llama3.1"),
        )