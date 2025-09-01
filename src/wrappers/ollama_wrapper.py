"""
Ollama LLM wrapper (first-class Harmony tool calling).

Defaults:
  base_url: http://localhost:11434/v1
  model:    OLLAMA_MODEL (default: llama3.1)

This wrapper uses Ollama's native /api/chat endpoint with 'tools' to enable
server-managed (Harmony) function calling. We still reuse the system prompt
from OpenAICompatibleWrapper to maintain consistent instructions, and we
fall back to sentinel JSON parsing if the server does not emit tool calls.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from .openai_compatible import OpenAICompatibleWrapper


class OllamaWrapper(OpenAICompatibleWrapper):
    """
    Preconfigured wrapper for Ollama with first-class Harmony tool calling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        strict: Optional[bool] = None,
    ) -> None:
        load_dotenv()

        # Strict tool-call prompting defaults ON for Ollama (more reliable JSON tool use)
        env_strict = os.getenv("OLLAMA_STRICT_PROMPT", "1").lower() in ("1", "true", "yes", "on")
        self._strict: bool = strict if strict is not None else env_strict

        # Ollama typically ignores API keys; we pass a placeholder by default
        super().__init__(
            api_key=api_key or os.getenv("OLLAMA_API_KEY", "ollama"),
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            model=model or os.getenv("OLLAMA_MODEL", "llama3.1"),
        )

    # Use stricter system prompt by default to coerce pure JSON tool calls from Ollama models.
    # Falls back to the standard prompt if strict mode is disabled.
    def _create_system_prompt(self) -> str:  # noqa: D401  (override base)
        if getattr(self, "_strict", False) and hasattr(self, "_build_strict_tool_prompt"):
            return self._build_strict_tool_prompt()  # provided by OpenAICompatibleWrapper
        # Fallback to the canonical prompt from the base class
        return super()._create_system_prompt()

    def _root_host(self) -> str:
        """
        Convert base_url like http://localhost:11434/v1 -> http://localhost:11434 for /api/* calls.
        """
        host = (self.base_url or "").rstrip("/")
        return host[:-3] if host.endswith("/v1") else host

    def _make_ollama_tool_def(self, name: str, description: str, raw_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert our internal tool schema into Ollama Tool JSON.
        """
        properties = (raw_schema.get("properties") or {})
        required = (raw_schema.get("required") or [])

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def _build_ollama_tools(self) -> List[Dict[str, Any]]:
        """
        Build the 'tools' array for Ollama Harmony.

        Important: Include both '<name>' and 'tool.<name>' aliases to satisfy models
        that emit Harmony-styled names like 'tool.file'. This prevents reverse mapping
        failures like: "harmony parser: no reverse mapping found for function name".
        """
        defs: List[Dict[str, Any]] = []
        for name, info in self.tools.items():
            raw_schema = info.get("raw_schema") or {}
            desc = info.get("description") or ""
            # canonical name
            defs.append(self._make_ollama_tool_def(name, desc, raw_schema))
            # harmony alias
            defs.append(self._make_ollama_tool_def(f"tool.{name}", desc, raw_schema))
        return defs

    @staticmethod
    def _normalize_tool_name(fn_name: str) -> str:
        """
        Normalize Harmony function names to our local registry:
        - 'tool.file' -> 'file'
        - 'file'      -> 'file'
        """
        if not fn_name:
            return fn_name
        return fn_name.split(".")[-1] if "." in fn_name else fn_name

    def _strip_code_fences(self, s: str) -> str:
        """
        Remove a single leading or first code-fence block ```...``` if present.
        Returns the inner content trimmed, else the original string trimmed.
        """
        if not isinstance(s, str):
            return s  # type: ignore[return-value]
        text = s.strip()
        if text.startswith("```"):
            end = text.find("```", 3)
            if end != -1:
                return text[3:end].strip()
        idx = text.find("```")
        if idx != -1:
            end = text.find("```", idx + 3)
            if end != -1:
                return text[idx + 3:end].strip()
        return text

    def _parse_args_safely(self, raw: Any) -> Dict[str, Any]:
        """
        Best-effort conversion of tool 'arguments' into a dict.
        Handles:
        - dict pass-through
        - JSON string (with or without ``` fences)
        - Strings containing extra trailing text by extracting the first balanced {...} object
        """
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            text = self._strip_code_fences(raw)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                # attempt to extract first balanced JSON object
                start = text.find("{")
                while start != -1:
                    depth = 0
                    for i in range(start, len(text)):
                        ch = text[i]
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                candidate = text[start : i + 1]
                                try:
                                    obj = json.loads(candidate)
                                    if isinstance(obj, dict):
                                        return obj
                                except Exception:
                                    pass
                                break
                    start = text.find("{", start + 1)
        return {}

    def _parse_json_relaxed(self, text: str) -> Dict[str, Any]:
        """
        Parse Ollama /api/chat response bodies that may contain NDJSON or extra text.
        Strategy:
        - Try strict json.loads
        - Fallback to NDJSON: decode each non-empty line and take the last object
        - Fallback to balanced-brace scan to extract the first JSON object
        """
        # Strict parse first
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # NDJSON fallback
        last_obj: Dict[str, Any] | None = None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                if isinstance(o, dict):
                    last_obj = o
            except Exception:
                continue
        if last_obj is not None:
            return last_obj

        # Balanced-brace scan
        start = text.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            o = json.loads(candidate)
                            if isinstance(o, dict):
                                return o
                        except Exception:
                            pass
                        break
            start = text.find("{", start + 1)
        return {}
    def _strip_code_fences(self, s: str) -> str:
        """
        Remove a single leading or first code-fence block ```...``` if present.
        """
        if not isinstance(s, str):
            return s
        text = s.strip()
        if text.startswith("```"):
            end = text.find("```", 3)
            if end != -1:
                return text[3:end].strip()
        idx = text.find("```")
        if idx != -1:
            end = text.find("```", idx + 3)
            if end != -1:
                return text[idx + 3:end].strip()
        return text

    def _parse_args_safely(self, raw: Any) -> Dict[str, Any]:
        """
        Best-effort conversion of tool 'arguments' into a dict.
        Handles:
        - dict pass-through
        - JSON string (with or without ``` fences)
        - Strings containing extra trailing text by extracting the first balanced {...} object
        """
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            text = self._strip_code_fences(raw)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                # attempt to extract first balanced JSON object
                start = text.find("{")
                while start != -1:
                    depth = 0
                    for i in range(start, len(text)):
                        ch = text[i]
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                candidate = text[start : i + 1]
                                try:
                                    obj = json.loads(candidate)
                                    if isinstance(obj, dict):
                                        return obj
                                except Exception:
                                    pass
                                break
                    start = text.find("{", start + 1)
        return {}

    def execute(self, user_input: str) -> str:
        """
        Execute via Ollama /api/chat with server-managed tool calling.
        Falls back to sentinel JSON parsing when no tool_calls are returned.
        """
        try:
            system_prompt = self._create_system_prompt()

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                "tools": self._build_ollama_tools(),
            }

            url = f"{self._root_host()}/api/chat"
            headers = {"Content-Type": "application/json", "Accept": "application/json"}

            resp = requests.post(url, headers=headers, data=json.dumps(payload))
            resp.raise_for_status()
            data: Dict[str, Any] = resp.json()

            message: Dict[str, Any] = (data.get("message") or {})
            content: str = (message.get("content") or "").strip()
            reasoning: str = content  # use assistant content as the reasoning section

            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                chosen_name: str = ""
                args: Dict[str, Any] = {}
                tool_obj = None

                # Prefer the first tool call that maps to a registered tool
                for call in tool_calls:
                    fn = (call or {}).get("function") or {}
                    raw_name = str(fn.get("name") or "")
                    fn_name = self._normalize_tool_name(raw_name)

                    raw_args = fn.get("arguments")
                    args = self._parse_args_safely(raw_args)

                    if fn_name and fn_name in self.tools:
                        chosen_name = fn_name
                        tool_obj = self.tools[fn_name]["tool"]
                        break

                if tool_obj is None:
                    # No matching tool provided by server; try sentinel parsing fallback
                    tool_call = self._extract_tool_call(content) if hasattr(self, "_extract_tool_call") else None
                    if not tool_call:
                        return f"Reasoning:\n{reasoning}\n\nError: Tool '{(chosen_name or '(none)')}' not found."
                    chosen_name = tool_call.get("tool") or ""
                    args = tool_call.get("input_schema", {}) or {}
                    tool_obj = self.tools.get(chosen_name, {}).get("tool")
                    if tool_obj is None:
                        return f"Reasoning:\n{reasoning}\n\nError: Tool '{chosen_name}' not found."

                result = tool_obj.run(args)  # type: ignore[attr-defined]
                tool_call_json = {"tool": chosen_name, "input_schema": args}
                return (
                    f"Reasoning:\n{reasoning}\n\n"
                    f"Tool Call:\n{json.dumps(tool_call_json, indent=2)}\n\n"
                    f"Result:\n{result}"
                )

            # No server-side tool call: attempt sentinel parsing from content
            tool_call = self._extract_tool_call(content) if hasattr(self, "_extract_tool_call") else None
            if not tool_call:
                return f"Reasoning:\n{reasoning}\n\nNo valid tool call was made."

            tool_name = tool_call.get("tool")
            params = tool_call.get("input_schema", {}) or {}
            if not tool_name or tool_name not in self.tools:
                return f"Reasoning:\n{reasoning}\n\nError: Tool '{tool_name}' not found."

            tool = self.tools[tool_name]["tool"]
            result = tool.run(params)
            return (
                f"Reasoning:\n{reasoning}\n\n"
                f"Tool Call:\n{json.dumps(tool_call, indent=2)}\n\n"
                f"Result:\n{result}"
            )

        except Exception as e:
            return f"Error executing tool: {str(e)}"