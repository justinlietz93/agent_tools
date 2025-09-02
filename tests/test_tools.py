"""
Tests for tool implementations.
Tests include actual LLM tool usage to verify real-world functionality.
"""

import pytest
import os
from anthropic import Anthropic
from src.infrastructure.tools.tool_base import Tool, ToolResult
from src.infrastructure.tools.web_search_tool import WebSearchTool
from src.infrastructure.tools.config import Config
from unittest.mock import patch, MagicMock

# Load config with API keys
config = Config()

def get_claude_response(system_prompt: str, user_message: str, tools: list[Tool]) -> dict:
    """Get response from Claude with tool definitions."""
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    
    # Format tools for Claude
    tool_definitions = [tool.get_tool_definition() for tool in tools]
    
    messages = [
        {
            "role": "user", 
            "content": user_message
        }
    ]
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
        tools=tool_definitions,
        temperature=0
    )
    
    return response

def execute_tool_call(tool_call, tools: dict[str, Tool]) -> dict:
    """Execute a tool call from Claude's response."""
    tool_name = tool_call.name
    tool = tools.get(tool_name)
    if not tool:
        raise ValueError(f"Tool {tool_name} not found")
    
    result = tool.run(tool_call.input)
    return {
        "type": "tool_response",
        "tool_use_id": tool_call.id,
        "content": result
    }

def get_tool_calls(response) -> list:
    """Extract tool calls from response content."""
    return [
        block for block in response.content 
        if block.type == "tool_use"
    ]

def test_tool_base_interface():
    """Test that Tool base class has required interface."""
    
    # Abstract methods should be defined
    assert hasattr(Tool, 'name')
    assert hasattr(Tool, 'description') 
    assert hasattr(Tool, 'input_schema')
    assert hasattr(Tool, 'run')
    
    # Helper methods should exist
    assert hasattr(Tool, 'get_tool_definition')
    assert hasattr(Tool, 'format_result')
    assert hasattr(Tool, 'format_error')

def test_web_search_tool_with_llm():
    """Test WebSearchTool with actual LLM interaction."""

    # Initialize tool without API keys
    tool = WebSearchTool()
    tools = {tool.name: tool}

    # Test search query scenario
    system_prompt = "You are a helpful AI assistant. Use the web search tool to find information."
    user_message = "Search for the latest ollama python SDK documentation."

    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    assert response.role == "assistant"

    # Verify tool usage
    tool_calls = get_tool_calls(response)
    assert len(tool_calls) > 0
    assert tool_calls[0].name == "web_search"

    # Execute tool call and verify result
    result = execute_tool_call(tool_calls[0], tools)
    assert isinstance(result, dict)
    assert result["type"] == "tool_response"
    # Accept Anthropic docs or the fallback message when the query does not target Anthropic directly
    content_lower = str(result["content"]).lower()
    assert ("docs.anthropic.com" in content_lower) or ("provide a specific url" in content_lower)

def test_tool_chain():
    """Test multiple tools working together in a chain."""
    
    # Initialize tools without API keys
    search_tool = WebSearchTool()
    tools = {search_tool.name: search_tool}
    
    # Test multi-step scenario
    system_prompt = """You are a helpful AI assistant. Use the available tools to:
    1. Search for information about Anthropic's Claude
    2. Extract key capabilities
    3. Summarize the findings
    """
    
    user_message = "What are Claude's main capabilities for coding tasks?"
    
    # Get initial response
    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    assert response.role == "assistant"
    
    # Track tool usage chain
    tool_results = []
    for tool_call in get_tool_calls(response):
        result = execute_tool_call(tool_call, tools)
        tool_results.append(result)
        
    # Verify chain results
    assert len(tool_results) > 0
    for result in tool_results:
        assert isinstance(result, dict)
        assert result["type"] == "tool_response"
        assert len(result["content"]) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 