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
from src.infrastructure.tools.tool_base import Tool  # for type hints

# Safe import for OpenAI Agents SDK integration
try:
    from ...integrations.openai_agents.bridge import run_openai_agents, SDK_AVAILABLE
except ImportError:
    run_openai_agents = None  # type: ignore
    SDK_AVAILABLE = False


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
        Robustly extract a tool call JSON object from model output.

        Accepted forms (case-insensitive):
        - TOOL_CALL: { ... }                 # preferred sentinel
        - Tool Call: ```json { ... } ```     # fenced JSON
        - Any first JSON object in content that has keys: "tool" and "input_schema"
        """
        txt = content or ""
        try:
            # 1) Preferred sentinel, case-insensitive
            lower = txt.lower()
            for sentinel in ("tool_call:", "tool call:", "toolcall:"):
                if sentinel in lower:
                    idx = lower.index(sentinel) + len(sentinel)
                    tail = txt[idx:].strip()
                    # if fenced, strip first fence
                    if tail.startswith("```"):
                        fence_end = tail.find("```", 3)
                        if fence_end != -1:
                            tail = tail[3:fence_end]
                    # attempt direct JSON parse
                    try:
                        obj = json.loads(tail)
                        if isinstance(obj, dict) and "tool" in obj and "input_schema" in obj:
                            return obj
                    except Exception:
                        pass
            # 2) Code-fenced JSON blocks ```json ... ```
            import re
            fence_re = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
            for m in fence_re.finditer(txt):
                block = m.group(1).strip()
                try:
                    obj = json.loads(block)
                    if isinstance(obj, dict) and "tool" in obj and "input_schema" in obj:
                        return obj
                except Exception:
                    continue
            # 3) First JSON object heuristic (balanced braces) that looks like a tool call
            #    Scan for a '{' then attempt to parse incrementally until balanced.
            start = txt.find("{")
            while start != -1:
                depth = 0
                for end in range(start, len(txt)):
                    ch = txt[end]
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = txt[start : end + 1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict) and "tool" in obj and "input_schema" in obj:
                                    return obj
                            except Exception:
                                pass
                            break
                start = txt.find("{", start + 1)
        except Exception:
            pass
        return None

    def _build_strict_tool_prompt(self) -> str:
        """
        Build a strict system instruction forcing a pure JSON tool call.
        This improves reliability on weaker OpenAI-compatible models (e.g., some Ollama models).
        """
        # Expose available tools with their JSONSchema exactly
        chunks = [
            "You MUST output ONLY a JSON object for a tool call. No prose, no backticks, no extra text.",
            "The JSON MUST have exactly these keys: tool (string), input_schema (object).",
            "Available tools and their JSON schemas follow. Choose exactly one and fill input_schema accordingly."
        ]
        for name, info in self.tools.items():
            schema = info.get("raw_schema") or {}
            chunks.append(f"TOOL {name} SCHEMA:")
            try:
                chunks.append(json.dumps(schema, ensure_ascii=False))
            except Exception:
                chunks.append(str(schema))
        chunks.append('Example output shape (choose valid tool): {"tool":"file","input_schema":{"operation":"read","path":"..."} }')
        return "\n".join(chunks)

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
        # Check for OpenAI Agents SDK integration flag
        flag_enabled = os.getenv("AGENT_TOOLS_OPENAI_AGENTS") in ("1", "true", "yes", "on")

        if flag_enabled and SDK_AVAILABLE and run_openai_agents:
            try:
                # Use SDK integration
                system_prompt = self._create_system_prompt()
                result = run_openai_agents(
                    model=self.model,
                    system_prompt=system_prompt,
                    user_input=user_input,
                    tools_registry=self.tools,
                    api_key=self.api_key,
                    base_url=self.base_url
                )

                # Format output to match legacy format
                if result.get("tool_call"):
                    # Tool call path
                    tool_call_json = json.dumps({
                        "tool": result["tool_call"]["tool"],
                        "input_schema": result["tool_call"]["input_schema"]
                    }, indent=2)
                    return (
                        f"Reasoning:\n{result.get('reasoning', '')}\n\n"
                        f"Tool Call:\n{tool_call_json}\n\n"
                        f"Result:\n{result.get('tool_result', '')}"
                    )
                else:
                    # No tool call path
                    response = result.get("response", "")
                    reasoning_str = str(result.get("reasoning", "") or "")
                    if response:
                        resp_str = str(response)
                        # De-duplicate: if reasoning equals response, suppress reasoning block
                        if reasoning_str.strip() == resp_str.strip():
                            reasoning_str = ""
                        return f"Reasoning:\n{reasoning_str}\n\nResponse:\n{resp_str}"
                    else:
                        return f"Reasoning:\n{reasoning_str}\n\nNo valid tool call was made."

            except Exception as e:
                # Fallback to legacy behavior on SDK errors
                pass

        # Legacy behavior (fallback)
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
                # Conversational fallback: return model text in Response, avoid duplicating in Reasoning
                content_str = (content or "").strip()
                if content_str:
                    reasoning_str = (reasoning or "").strip()
                    if reasoning_str == content_str:
                        reasoning_str = ""
                    return f"Reasoning:\n{reasoning_str}\n\nResponse:\n{content_str}"
                else:
                    return f"Reasoning:\n{(reasoning or '').strip()}\n\nNo valid tool call was made."

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