"""
OpenAI Agents SDK Bridge Module.

Provides integration with OpenAI Agents Python SDK for tool orchestration.
Handles safe imports, router tool creation, and agent execution.

File length constraint: <500 LOC
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

# Safe imports with fallbacks
try:
    from agents import Agent, Runner, function_tool
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    Agent = None  # type: ignore
    Runner = None  # type: ignore
    function_tool = None  # type: ignore


def create_router_tool(tools_registry: Dict[str, Any]) -> Any:
    """
    Create a single router function tool that delegates to ToolManager.

    Args:
        tools_registry: Dict mapping tool names to tool info (from wrapper.tools)

    Returns:
        Function tool if SDK available, None otherwise
    """
    if not SDK_AVAILABLE or not function_tool:
        return None

    @function_tool(name="tool_router")
    def router_tool(tool_name: str, input: Dict[str, Any]) -> str:
        """
        Route tool execution to registered tools.

        Args:
            tool_name: Name of the tool to execute
            input: Input parameters as dict

        Returns:
            Tool execution result as string
        """
        if tool_name not in tools_registry:
            return f"Error: Tool '{tool_name}' not found."

        tool_info = tools_registry[tool_name]
        tool = tool_info["tool"]

        try:
            # Try calling with input_dict first (preferred)
            if hasattr(tool, 'run') and callable(tool.run):
                # Check signature - if it takes exactly 2 args (self, input_dict), use that
                import inspect
                sig = inspect.signature(tool.run)
                params = list(sig.parameters.values())
                if len(params) == 2 and params[1].name != 'tool_call_id':
                    # Assume run(input_dict)
                    result = tool.run(input)
                else:
                    # Use fallback: run("sdk", **input)
                    result = tool.run("sdk", **input)
            else:
                return f"Error: Tool '{tool_name}' has no callable run method."

            # Serialize result if dict
            if isinstance(result, dict):
                return json.dumps(result)
            return str(result)

        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"

    return router_tool


def run_openai_agents(
    model: str,
    system_prompt: str,
    user_input: str,
    tools_registry: Dict[str, Any],
    api_key: Optional[str],
    base_url: Optional[str]
) -> Dict[str, Any]:
    """
    Execute user input using OpenAI Agents SDK.

    Args:
        model: Model name
        system_prompt: System instructions
        user_input: User query
        tools_registry: Tool registry dict
        api_key: API key
        base_url: Base URL

    Returns:
        Dict with reasoning, tool_call, tool_result, response, raw_items
    """
    if not SDK_AVAILABLE or not Agent or not Runner:
        raise ImportError("OpenAI Agents SDK not available")

    # Create router tool
    router_tool = create_router_tool(tools_registry)
    if not router_tool:
        raise RuntimeError("Failed to create router tool")

    # Create agent with router tool
    agent = Agent(
        name="ToolOrchestrator",
        instructions=system_prompt,
        model=model,
        tools=[router_tool],
        # Use OpenAI client config if provided
        model_settings={
            "api_key": api_key,
            "base_url": base_url,
        } if api_key or base_url else None
    )

    # Run synchronously (since wrapper is sync)
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If event loop is already running, use run_until_complete
            result = loop.run_until_complete(Runner.run(agent, user_input))
        else:
            # Create new event loop
            result = asyncio.run(Runner.run(agent, user_input))
    except RuntimeError:
        # Fallback for environments without event loop
        import nest_asyncio
        nest_asyncio.apply()
        result = asyncio.run(Runner.run(agent, user_input))

    # Extract components from RunResult
    reasoning = ""
    tool_call = None
    tool_result = None
    response = None

    # Parse final_output and items
    final_output = result.final_output or ""

    # Look for tool calls in new_items
    for item in result.new_items:
        if hasattr(item, 'type'):
            if item.type == 'tool_call':
                # Extract tool call info
                tool_call = {
                    "tool": getattr(item, 'tool_name', ''),
                    "input_schema": getattr(item, 'arguments', {})
                }
                # Tool result might be in subsequent items
                if hasattr(item, 'output'):
                    tool_result = item.output
            elif item.type == 'text' and not response:
                response = getattr(item, 'content', '')

    # If no explicit tool call found, check if final_output contains tool info
    if not tool_call and "tool_router" in final_output:
        # Parse from text (fallback)
        if "tool_name" in final_output and "input" in final_output:
            try:
                # Simple parsing - in practice, SDK should provide structured data
                tool_call = {"tool": "unknown", "input_schema": {}}
                tool_result = final_output
            except:
                pass

    # If no tool call, use final_output as response
    if not tool_call:
        response = final_output
        reasoning = ""  # SDK handles reasoning internally

    return {
        "reasoning": reasoning,
        "tool_call": tool_call,
        "tool_result": tool_result,
        "response": response,
        "raw_items": [str(item) for item in result.new_items]
    }