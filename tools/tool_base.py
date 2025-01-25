# selfprompter/tools/tool_base.py
"""
Base classes for Anthropic Claude tool implementation.
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
    Abstract base class for tools following Anthropic Claude standards.
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
        """Get the tool definition in Anthropic's format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.input_schema["properties"],
                "required": self.input_schema["required"]
            }
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
