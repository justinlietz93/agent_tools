# selfprompter/tools/tool_base.py
"""
Based on: https://docs.anthropic.com/claude/docs/tool-use
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypedDict, List, Literal

class ToolInput_schema(TypedDict):
    type: str
    function: Dict[str, Any]

class ToolCall(TypedDict):
    id: str
    type: str
    function: Dict[str, Any]

class ToolResult(TypedDict):
    type: str
    tool_use_id: str
    content: str

class Tool(ABC):
    """
    See: https://docs.anthropic.com/claude/docs/tool-use
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in function calling format."""
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        pass
        
    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """
        JSONSchema object defining accepted input_schema.
        Must include:
        - type: "object"
        - properties: Parameter definitions
        - required: List of required input_schema
        """
        pass

    @abstractmethod
    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with the given input input_schema."""
        pass

    def get_tool_definition(self) -> Dict[str, Any]:
        """Get the tool definition in Anthropic's format.

        Robust against mis-implemented input_schema properties in subclasses that
        accidentally recurse (e.g., property returns self.input_schema).
        """
        schema: Dict[str, Any] | None = None

        # Primary attempt - may raise RecursionError if subclass is faulty
        try:
            schema = self.input_schema
        except RecursionError:
            schema = None
        except Exception:
            # Ignore and try fallbacks
            schema = None

        # Fallback: walk MRO to find a sane input_schema on a base class
        if not isinstance(schema, dict) or not schema:
            for cls in type(self).mro()[1:]:
                if "input_schema" in cls.__dict__:
                    attr = cls.__dict__["input_schema"]
                    try:
                        if isinstance(attr, property):
                            candidate = attr.__get__(self, cls)
                        else:
                            candidate = attr
                    except RecursionError:
                        continue
                    except Exception:
                        continue
                    if isinstance(candidate, dict) and candidate:
                        schema = candidate
                        break

        # Final guard: ensure dictionary shape
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}, "required": []}

        properties = schema.get("properties", {}) or {}
        required = schema.get("required", []) or []

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def format_result(self, tool_call_id: str, content: str) -> ToolResult:
        """Format a successful result in Anthropic's format."""
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": content
        }

    def format_error(self, tool_call_id: str, error: str) -> ToolResult:
        """Format an error result in Anthropic's format."""
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": f"Error: {error}"
        }
