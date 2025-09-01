"""
LLM Wrapper Base Interface.

Defines a provider-agnostic interface for tool-enabled LLM wrappers that:
- Register Tool instances and expose their NL schema
- Build a consistent system prompt instructing EXACT tool call format
- Execute a user prompt and return a formatted string with Reasoning, Tool Call, and Result

Design constraints:
- One class per file, <500 LOC
- Separation of concerns: Provider-specific logic in subclasses
- Maintain compatibility with existing DeepseekToolWrapper result formatting
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

# Sibling package import using absolute path
from src.infrastructure.tools.tool_base import Tool  # noqa: F401


class LLMWrapper(ABC):
    """
    Abstract base for LLM wrappers that orchestrate tool usage via prompt engineering.

    Responsibilities:
    - Tool registration (with NL schema exposition)
    - System prompt construction (consistent format across providers)
    - Provider-specific execution in concrete subclasses
    """

    def __init__(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {}

    def _convert_schema_to_nl(self, schema: Dict[str, Any]) -> str:
        """
        Convert JSONSchema to a natural-language bullet list.

        Example line:
        - param (string*): Description here
        """
        nl_desc = []
        props = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])

        for name, details in props.items():
            desc = details.get("description", "No description available")
            type_info = details.get("type", "any")
            req_mark = "*" if name in required else ""
            nl_desc.append(f"- {name} ({type_info}{req_mark}): {desc}")

        return "\n".join(nl_desc)

    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool with its description and NL schema.

        This maintains backward-compatible fields used by the legacy Deepseek wrapper.
        """
        self.tools[tool.name] = {
            "tool": tool,
            "description": tool.description,
            "schema": self._convert_schema_to_nl(tool.input_schema),
            "raw_schema": tool.input_schema,
        }

    def _create_system_prompt(self) -> str:
        """
        Build the canonical system prompt that:
        - Lists available tools with descriptions and input schemas (NL)
        - Instructs the model to first explain reasoning, then emit EXACT tool call JSON
        - Enforces strict JSON format under a TOOL_CALL: sentinel
        """
        tools_desc = []
        for name, info in self.tools.items():
            tools_desc.append(
                f"Tool: {name}\n"
                f"Description: {info['description']}\n"
                f"Input_schema:\n{info['schema']}\n"
            )
        tools_block = "\n".join(tools_desc)

        return (
            "You are an AI assistant with access to the following tools:\n"
            f"{tools_block}\n"
            "To use a tool, first explain your reasoning using Chain of Thought, then respond with a tool call in this EXACT format:\n"
            "TOOL_CALL:\n"
            "{\n"
            '    "tool": "tool_name",\n'
            '    "input_schema": {\n'
            '        "param1": "value1",\n'
            '        "param2": "value2"\n'
            "    }\n"
            "}\n\n"
            "Make sure to:\n"
            "1. Use valid JSON format\n"
            "2. Include all required input_schema\n"
            "3. Use correct parameter types\n"
            "4. Only use tools that are listed above"
        )

    @abstractmethod
    def execute(self, user_input: str) -> str:
        """
        Execute a tool based on user input via the provider-specific LLM.

        Return format MUST match existing DeepseekToolWrapper behavior exactly:
        - Reasoning:\n{reasoning}\n\nTool Call:\n{json}\n\nResult:\n{tool_result}
        """
        raise NotImplementedError("Subclasses must implement execute()")