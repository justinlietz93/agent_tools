"""
Tool invocation adapter implementing IToolInvocationAdapter interface.
"""

import json
from typing import Dict, Any, Optional, TYPE_CHECKING
from src.abstractions.dto.tools import ToolInvocationResult

if TYPE_CHECKING:
    from src.interfaces.services.tools import IToolInvocationAdapter

from .tool_manager import ToolManager


class ToolInvocationAdapter:
    """
    Adapter for ToolManager to implement IToolInvocationAdapter interface.
    """

    def __init__(self):
        self.manager = ToolManager(register_defaults=True)

    def execute(self, name: str, params: Dict[str, Any], call_id: Optional[str] = None) -> "ToolInvocationResult":
        """
        Execute a tool by name with given parameters.
        """
        try:
            tool = self.manager.get_tool(name)
        except KeyError:
            return ToolInvocationResult(
                ok=False,
                value=None,
                error=f"Tool '{name}' not found",
                tool_name=name,
                signature_used=""
            )

        try:
            # Try tool.run(params) first
            result = tool.run(params)
            signature_used = "run(input)"
        except TypeError:
            # Fallback to tool.run(call_id, **params)
            try:
                result = tool.run(call_id, **params)
                signature_used = "run(call_id, **kwargs)"
            except Exception as e:
                return ToolInvocationResult(
                    ok=False,
                    value=None,
                    error=f"Tool execution failed: {str(e)}",
                    tool_name=name,
                    signature_used=""
                )

        # Serialize result
        try:
            if isinstance(result, dict):
                value = json.dumps(result, ensure_ascii=False)
            else:
                value = str(result)
        except Exception as e:
            value = f"Serialization error: {str(e)}"

        return ToolInvocationResult(
            ok=True,
            value=value,
            error=None,
            tool_name=name,
            signature_used=signature_used
        )