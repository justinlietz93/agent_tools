"""
Tool catalog and invocation ports.
"""
from __future__ import annotations
from typing import Protocol, List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.abstractions.dto.tools import ToolDescriptor, ToolInvocationResult

class IToolCatalog(Protocol):
    def list_tools(self) -> List["ToolDescriptor"]:
        ...
    def get_tool(self, name: str) -> Optional["ToolDescriptor"]:
        ...

class IToolInvocationAdapter(Protocol):
    def execute(self, name: str, params: Dict[str, Any], call_id: Optional[str] = None) -> "ToolInvocationResult":
        ...

__all__ = ["IToolCatalog", "IToolInvocationAdapter"]