"""
OpenAI-compatible LLM wrapper.

Supports any provider exposing an OpenAI Chat Completions-compatible API:
- DeepSeek (https://api.deepseek.com)
- Ollama (http://localhost:11434/v1)
- Local/self-hosted OpenAI-compatible servers

Behavior:
- Uses the canonical tool-use system prompt from [LLMWrapper](agent_tools/src/wrappers/base.py:24) to elicit a strict TOOL_CALL JSON
- Parses TOOL_CALL and executes the mapped Tool.run(params)
- Returns a legacy-compatible string: 'Reasoning:\\n...\\n\\nTool Call:\\n{json}\\n\\nResult:\\n{tool_result}'

File length constraint: <500 LOC
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .base import LLMWrapper
from ..tools.tool_base import Tool  # for type hints


class OpenAICompatibleWrapper(LLMWrapper):
    """
    Provider-agnostic wrapper for OpenAI-compatible chat API.

    Example usage for DeepSeek:
        wrapper = OpenAICompatibleWrapper(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            model="deepseek-reasoner",
        )

    Example usage for Ollama:
        wrapper = OpenAICompatibleWrapper(
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),  # placeholder if unused
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__()
        load_dotenv()

        # Defaults allow easy ollama usage without extra config
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OLLAMA_API_KEY") or "ollama"
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
        self.model = model or os.getenv("OPENAI_MODEL") or os.getenv("DEEPSEEK_MODEL") or os.getenv("OLLAMA_MODEL") or "llama3.1"

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _extract_tool_call(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Extract tool call JSON following the sentinel 'TOOL_CALL:'.

        The model is instructed to output EXACT JSON immediately after the sentinel.
        """
        try:
            marker = "TOOL_CALL:"
            if marker not in content:
                return None
            tool_json = content.split(marker, 1)[1].strip()
            return json.loads(tool_json)
        except Exception:
            return None

    def _infer_reasoning(self, content: str, reasoning_content: Optional[str]) -> str:
        """
        Prefer provider-supplied 'reasoning_content' (DeepSeek), else infer
        from content before TOOL_CALL or fall back to entire content.
        """
        if reasoning_content:
            return reasoning_content

        marker = "TOOL_CALL:"
        if marker in content:
            return content.split(marker, 1)[0].strip()

        return content.strip()

    def execute(self, user_input: str) -> str:
        """
        Execute the LLM call and return legacy-compatible formatted output.

        Returns:
            str: "Reasoning:\\n{reasoning}\\n\\nTool Call:\\n{json}\\n\\nResult:\\n{result or error}"
        """
        try:
            system_prompt = self._create_system_prompt()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
            )

            # OpenAI-compatible response structure
            message = response.choices[0].message
            content = message.content or ""
            # DeepSeek-specific extension (present only on deepseek-reasoner)
            reasoning_content = getattr(message, "reasoning_content", None)
            reasoning = self._infer_reasoning(content, reasoning_content)

            tool_call = self._extract_tool_call(content)
            if not tool_call:
                return f"Reasoning:\n{reasoning}\n\nNo valid tool call was made."

            tool_name = tool_call.get("tool")
            params = tool_call.get("input_schema", {}) or {}

            if not tool_name or tool_name not in self.tools:
                return f"Reasoning:\n{reasoning}\n\nError: Tool '{tool_name}' not found."

            tool: Tool = self.tools[tool_name]["tool"]  # type: ignore[assignment]
            result = tool.run(params)

            return (
                f"Reasoning:\n{reasoning}\n\n"
                f"Tool Call:\n{json.dumps(tool_call, indent=2)}\n\n"
                f"Result:\n{result}"
            )

        except Exception as e:
            return f"Error executing tool: {str(e)}"