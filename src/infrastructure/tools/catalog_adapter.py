"""
Tool catalog adapter implementing IToolCatalog interface.
"""

from typing import List, Optional, TYPE_CHECKING
from src.abstractions.dto.tools import ToolDescriptor

if TYPE_CHECKING:
    from src.interfaces.services.tools import IToolCatalog

from .tool_manager import ToolManager


class ToolManagerCatalogAdapter:
    """
    Adapter for ToolManager to implement IToolCatalog interface.
    """

    def __init__(self):
        self.manager = ToolManager(register_defaults=True)

    def list_tools(self) -> List["ToolDescriptor"]:
        """
        List all registered tools as ToolDescriptor objects.
        """
        tools_info = self.manager.list_tools()
        return [
            ToolDescriptor(
                name=info["name"],
                description=info["description"],
                raw_schema=info["input_schema"]
            )
            for info in tools_info
        ]

    def get_tool(self, name: str) -> Optional["ToolDescriptor"]:
        """
        Get a tool descriptor by name.
        """
        try:
            tool = self.manager.get_tool(name)
            return ToolDescriptor(
                name=tool.name,
                description=tool.description,
                raw_schema=tool.input_schema
            )
        except KeyError:
            return None