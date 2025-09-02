# selfprompter/tools/tool_manager.py

import os
from typing import Dict, Any, List, Type
from .tool_base import Tool

# Import all available tools
from .shell_tool import ShellTool
from .requests_tool import RequestsTool
from .file_tool import FileTool
from .web_search_tool import WebSearchTool
from .web_browser_tool import WebBrowserTool
from .package_manager_tool import PackageManagerTool
from .advanced_file_tool import AdvancedFileTool
from .code_runner_tool import CodeRunnerTool

# Optional: DocCheckTool is registered conditionally based on Context7 availability or env gate
from .mcp_tools.doc_check_tool import DocCheckTool  # type: ignore


class ToolManager:
    """
    Manages a collection of tools and handles tool registration and execution.
    
    """

    def __init__(self, register_defaults: bool = True):
        """
        Initialize tool registry.
        
        Args:
            register_defaults: Whether to register default tools
        """
        self.tools: Dict[str, Tool] = {}
        if register_defaults:
            self.register_default_tools()

    def register_default_tools(self) -> None:
        """Register all default tools with standard configurations."""
        default_tools = [
            ShellTool(),
            RequestsTool(),
            FileTool(),
            WebSearchTool(),       # No API key args; uses direct HTTP fetch
            WebBrowserTool(),      # No timeout arg; accepts optional user_agent only
            PackageManagerTool(),
            AdvancedFileTool(),
            CodeRunnerTool(),
        ]
        for tool in default_tools:
            self.register_tool(tool)

        # Feature-gated registration for DocCheckTool (Context7 client)
        self._maybe_register_doc_tool()

    def _maybe_register_doc_tool(self) -> None:
        """
        Register DocCheckTool if:
        - CONTEXT7_ENABLE_DOC_TOOL in {"1","true","yes","on"} OR
        - Context7 client health_check() is True at startup.
        """
        gate = (os.getenv("CONTEXT7_ENABLE_DOC_TOOL", "") or "").strip().lower()
        if gate in {"1", "true", "yes", "on"}:
            # Register with a lazy client (constructor injection handled by tool if None)
            self.register_tool(DocCheckTool())
            return

        # Auto-enable if reachable
        try:
            from src.infrastructure.mcp.context7_client import Context7MCPClient  # lazy import
            client = Context7MCPClient()
            if client.health_check(timeout=1.5):
                self.register_tool(DocCheckTool(context7_client=client))
        except Exception:
            # Unreachable or client import failure -> remain hidden
            pass

    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool instance.
        
        Args:
            tool: Tool instance to register
        """
        self.tools[tool.name] = tool

    def register_tool_class(self, tool_class: Type[Tool], **kwargs) -> None:
        """
        Register a tool class by instantiating and registering it.
        
        Args:
            tool_class: Tool class to instantiate and register
            **kwargs: Arguments to pass to tool constructor
        """
        tool = tool_class(**kwargs)
        self.register_tool(tool)

    def get_tool(self, name: str) -> Tool:
        """
        Get a registered tool by name.
        
        Args:
            name: Name of the tool to retrieve
            
        Returns:
            The requested tool instance
            
        Raises:
            KeyError: If tool is not found
        """
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found")
        return self.tools[name]

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered tools.
        
        Returns:
            List of tool information dictionaries containing name,
            description and input_schema for each tool
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema
            }
            for tool in self.tools.values()
        ]

    def execute_tool(self, tool_call_id: str, name: str, **kwargs) -> Any:
        """
        Execute a tool by name with given arguments.
        
        Args:
            tool_call_id: Unique ID for this tool call
            name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            KeyError: If tool is not found
        """
        tool = self.get_tool(name)
        return tool.run(tool_call_id, **kwargs)