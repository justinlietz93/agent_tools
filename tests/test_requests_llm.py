"""
Tests for RequestsTool with actual LLM interactions.
Verifies that Claude can properly understand and use the HTTP request tool.
"""

import pytest
import requests
from tools.requests_tool import RequestsTool
from typing import Dict, Any
from anthropic import Anthropic
import os
from unittest.mock import patch, MagicMock

# Create concrete class for testing
class ConcreteRequestsTool(RequestsTool):
    """Concrete implementation of RequestsTool for testing"""
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return self.input_schema

# Test fixtures
@pytest.fixture
def requests_tool():
    """Create RequestsTool instance"""
    return ConcreteRequestsTool()

@pytest.fixture
def mock_anthropic():
    """Create mock Anthropic client"""
    with patch('anthropic.Anthropic') as mock:
        yield mock

@pytest.fixture
def mock_requests():
    """Create mock requests module"""
    with patch('requests.request') as mock:
        yield mock

# Unit Tests (Mock LLM and Mock Tool Calls)
@pytest.mark.unit
def test_tool_definition(requests_tool):
    """Test that tool definition is properly formatted"""
    definition = requests_tool.get_tool_definition()
    assert isinstance(definition, dict)
    assert "name" in definition
    assert "description" in definition
    assert "input_schema" in definition
    assert definition["name"] == "http_request"
    assert "properties" in definition["input_schema"]
    assert "url" in definition["input_schema"]["properties"]
    assert "method" in definition["input_schema"]["properties"]

@pytest.mark.unit
def test_basic_get_request_unit(requests_tool, mock_requests):
    """Test basic GET request with mocked requests"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Success"
    mock_response.headers = {"Content-Type": "text/plain"}
    mock_requests.return_value = mock_response

    result = requests_tool.run(
        tool_call_id="test1",
        url="https://test.com/api"
    )

    mock_requests.assert_called_once_with(
        method="GET",
        url="https://test.com/api",
        headers={},
        data=None,
        timeout=30
    )

    assert isinstance(result, dict)
    assert "content" in result
    assert "200" in result["content"]
    assert "Success" in result["content"]

@pytest.mark.unit
def test_post_request_unit(requests_tool, mock_requests):
    """Test POST request with mocked requests"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"status": "success"}'
    mock_response.headers = {"Content-Type": "application/json"}
    mock_requests.return_value = mock_response

    test_data = '{"key": "value"}'
    result = requests_tool.run(
        tool_call_id="test2",
        url="https://test.com/api",
        method="POST",
        data=test_data,
        headers={"Content-Type": "application/json"}
    )

    mock_requests.assert_called_once_with(
        method="POST",
        url="https://test.com/api",
        headers={"Content-Type": "application/json"},
        data=test_data,
        timeout=30
    )

    assert isinstance(result, dict)
    assert "200" in result["content"]
    assert "success" in result["content"]

@pytest.mark.unit
def test_error_handling_unit(requests_tool, mock_requests):
    """Test error handling with mocked requests"""
    mock_requests.side_effect = requests.RequestException("Connection failed")

    result = requests_tool.run(
        tool_call_id="test3",
        url="https://test.com/api"
    )

    assert isinstance(result, dict)
    assert "error" in result["content"].lower()
    assert "failed" in result["content"].lower()

@pytest.mark.unit
def test_timeout_handling_unit(requests_tool, mock_requests):
    """Test timeout handling with mocked requests"""
    mock_requests.side_effect = requests.Timeout("Request timed out")

    result = requests_tool.run(
        tool_call_id="test4",
        url="https://test.com/api",
        timeout=1
    )

    assert isinstance(result, dict)
    assert "error" in result["content"].lower()
    assert "request failed" in result["content"].lower()
    assert "timed out" in result["content"].lower()

@pytest.mark.unit
def test_invalid_method_unit(requests_tool, mock_requests):
    """Test invalid HTTP method handling"""
    result = requests_tool.run(
        tool_call_id="test5",
        url="https://test.com/api",
        method="INVALID"
    )

    assert isinstance(result, dict)
    assert "error" in result["content"].lower()
    assert "unsupported" in result["content"].lower()
    mock_requests.assert_not_called()

@pytest.mark.unit
def test_missing_url_unit(requests_tool, mock_requests):
    """Test missing URL handling"""
    result = requests_tool.run(
        tool_call_id="test6"
    )

    assert isinstance(result, dict)
    assert "error" in result["content"].lower()
    assert "url" in result["content"].lower()
    assert "required" in result["content"].lower()
    mock_requests.assert_not_called()

# Helper functions for integration tests
def get_claude_response(system_prompt: str, user_message: str, tools: list) -> dict:
    """Get response from Claude with tool definitions."""
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "dummy_key"))
    
    # Format tools for Claude
    tool_definitions = [tool.get_tool_definition() for tool in tools]
    
    messages = [
        {
            "role": "user", 
            "content": user_message
        }
    ]
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
        tools=tool_definitions,
        temperature=0
    )
    
    return response

def get_tool_calls(response) -> list:
    """Extract tool calls from response content."""
    return [
        block for block in response.content 
        if block.type == "tool_use"
    ]

def execute_tool_call(tool_call, tools: dict) -> dict:
    """Execute a tool call from Claude's response."""
    tool_name = tool_call.name
    tool = tools.get(tool_name)
    if not tool:
        raise ValueError(f"Tool {tool_name} not found")
    
    result = tool.run(tool_call_id=tool_call.id, **tool_call.input)
    return {
        "type": "tool_response",
        "tool_use_id": tool_call.id,
        "content": result["content"]
    }

# Integration Tests (Real LLM with Fake Target)
@pytest.mark.integration
@pytest.mark.llm
def test_basic_get_request_integration(requests_tool):
    """Test that Claude can make a basic GET request to httpbin."""
    tools = {requests_tool.name: requests_tool}

    system_prompt = """You are a helpful AI assistant. Use the HTTP request tool to make web requests.
    The tool can make GET, POST, PUT, and DELETE requests with custom headers and data."""

    user_message = "Make a GET request to https://httpbin.org/get"

    response = get_claude_response(system_prompt, user_message, list(tools.values()))

    # Debug: Print response content
    print("\nLLM Response Content:")
    for block in response.content:
        print(f"Block type: {block.type}")
        if hasattr(block, "text"):
            print(f"Text: {block.text}")

    # Verify LLM used the tool correctly
    tool_calls = get_tool_calls(response)
    assert len(tool_calls) > 0, "LLM should use the HTTP request tool"

    tool_call = tool_calls[0]
    assert tool_call.name == "http_request"
    assert tool_call.input.get("url") == "https://httpbin.org/get"
    assert tool_call.input.get("method", "GET") == "GET"

    # Execute the tool call with mocked response
    with patch('requests.request') as mock_request:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"args": {}, "headers": {}, "url": "https://httpbin.org/get"}'
        mock_request.return_value = mock_response
        
        result = execute_tool_call(tool_call, tools)

    # Verify the result format
    assert isinstance(result, dict)
    assert "content" in result
    assert "tool_use_id" in result
    assert "type" in result
    assert "200" in result["content"]

@pytest.mark.integration
@pytest.mark.llm
def test_post_request_with_data_integration(requests_tool):
    """Test that Claude can make a POST request with data to httpbin."""
    tools = {requests_tool.name: requests_tool}

    system_prompt = """You are a helpful AI assistant. Use the HTTP request tool to make web requests."""

    user_message = """Make a POST request to https://httpbin.org/post with this JSON data:
    {
        "name": "test",
        "value": "data"
    }"""

    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    tool_calls = get_tool_calls(response)
    assert len(tool_calls) > 0

    tool_call = tool_calls[0]
    assert tool_call.name == "http_request"
    assert tool_call.input.get("url") == "https://httpbin.org/post"
    assert tool_call.input.get("method") == "POST"
    assert "test" in tool_call.input.get("data", "")
    assert "data" in tool_call.input.get("data", "")

    # Execute the tool call with mocked response
    with patch('requests.request') as mock_request:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"json": {"name": "test", "value": "data"}}'
        mock_request.return_value = mock_response
        
        result = execute_tool_call(tool_call, tools)

    # Verify the result
    assert isinstance(result, dict)
    assert "200" in result["content"]
    assert "test" in result["content"]
    assert "data" in result["content"]

@pytest.mark.integration
@pytest.mark.llm
def test_custom_headers_integration(requests_tool):
    """Test that Claude can make requests with custom headers to httpbin."""
    tools = {requests_tool.name: requests_tool}

    system_prompt = """You are a helpful AI assistant. Use the HTTP request tool to make web requests."""

    user_message = """Make a GET request to https://httpbin.org/headers with these custom headers:
    {
        "X-Custom-Header": "test-value",
        "Accept": "application/json"
    }"""

    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    tool_calls = get_tool_calls(response)
    assert len(tool_calls) > 0

    tool_call = tool_calls[0]
    assert tool_call.name == "http_request"
    headers = tool_call.input.get("headers", {})
    assert headers.get("X-Custom-Header") == "test-value"
    assert headers.get("Accept") == "application/json"

    # Execute the tool call with mocked response
    with patch('requests.request') as mock_request:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"headers": {"X-Custom-Header": "test-value", "Accept": "application/json"}}'
        mock_request.return_value = mock_response
        
        result = execute_tool_call(tool_call, tools)

    # Verify the result
    assert isinstance(result, dict)
    assert "200" in result["content"]
    assert "X-Custom-Header" in result["content"]
    assert "test-value" in result["content"]

@pytest.mark.integration
@pytest.mark.llm
def test_error_handling_integration(requests_tool):
    """Test that Claude handles request errors appropriately with httpbin."""
    tools = {requests_tool.name: requests_tool}

    system_prompt = """You are a helpful AI assistant. Use the HTTP request tool to make web requests."""

    user_message = "Make a GET request to https://httpbin.org/status/404"

    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    tool_calls = get_tool_calls(response)
    assert len(tool_calls) > 0

    tool_call = tool_calls[0]
    assert tool_call.name == "http_request"

    # Execute the tool call with mocked 404 response
    with patch('requests.request') as mock_request:
        mock_request.side_effect = requests.HTTPError("404 Client Error")
        result = execute_tool_call(tool_call, tools)

    # Verify error handling
    assert isinstance(result, dict)
    assert "error" in result["content"].lower()
    assert "404" in result["content"].lower()

    # Send error back to Claude
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "dummy_key"))
    final_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response.content[0].text},
            {"role": "assistant", "content": f"The request failed: {result['content']}"}
        ],
        temperature=0
    )

    # Verify Claude acknowledges the error - check for various error-related terms
    error_terms = ["error", "failed", "unsuccessful", "404", "not found"]
    assert any(
        any(term in block.text.lower() for term in error_terms)
        for block in final_response.content
        if hasattr(block, "text")
    ), "Claude should acknowledge the error in some way"

@pytest.mark.integration
@pytest.mark.llm
def test_timeout_handling_integration(requests_tool):
    """Test that Claude handles request timeouts appropriately with httpbin."""
    tools = {requests_tool.name: requests_tool}

    system_prompt = """You are a helpful AI assistant. Use the HTTP request tool to make web requests."""

    user_message = """Make a GET request to https://httpbin.org/delay/10 with a 2 second timeout"""

    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    tool_calls = get_tool_calls(response)
    assert len(tool_calls) > 0

    tool_call = tool_calls[0]
    assert tool_call.name == "http_request"
    assert tool_call.input.get("timeout", 30) <= 2

    # Execute the tool call with mocked timeout
    with patch('requests.request') as mock_request:
        mock_request.side_effect = requests.Timeout("Request timed out")
        result = execute_tool_call(tool_call, tools)

    # Verify timeout handling
    assert isinstance(result, dict)
    assert "error" in result["content"].lower()
    assert "request failed" in result["content"].lower()
    assert "timed out" in result["content"].lower()  # Check for actual timeout message

# System Tests (Real LLM with Real Target)
@pytest.mark.system
@pytest.mark.llm
def test_basic_get_request_system(requests_tool):
    """Test that Claude can make a basic GET request to a real endpoint and understand the response."""
    tools = {requests_tool.name: requests_tool}

    system_prompt = """You are a helpful AI assistant. Use the HTTP request tool to make web requests.
    After getting the response, explain what you found in the response."""

    user_message = "Make a GET request to https://api.github.com and tell me what endpoints are available."

    print("\n=== Initial Claude Response ===")
    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    for block in response.content:
        print(f"\nBlock Type: {block.type}")
        if hasattr(block, "text"):
            print(f"Text: {block.text}")
        elif block.type == "tool_use":
            print(f"Tool Call: {block.name}")
            print(f"Tool Input: {block.input}")

    tool_calls = get_tool_calls(response)
    assert len(tool_calls) > 0, "Claude should decide to use the HTTP request tool"

    tool_call = tool_calls[0]
    assert tool_call.name == "http_request"
    assert tool_call.input.get("url") == "https://api.github.com"
    assert tool_call.input.get("method", "GET") == "GET"

    # Execute the tool call with real request
    result = execute_tool_call(tool_call, tools)

    # Verify the result format
    assert isinstance(result, dict)
    assert "200" in result["content"]
    assert "github" in result["content"].lower()

    # Send the result back to Claude for analysis
    print("\n=== Sending Result to Claude ===")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "dummy_key"))
    final_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response.content[0].text},
            {"role": "assistant", "content": f"Here's what I found from the API: {result['content']}"}
        ],
        temperature=0
    )

    print("\n=== Claude's Analysis ===")
    for block in final_response.content:
        print(f"\nBlock Type: {block.type}")
        if hasattr(block, "text"):
            print(f"Text: {block.text}")

    # Verify Claude's understanding
    response_text = " ".join(
        block.text.lower()
        for block in final_response.content
        if hasattr(block, "text")
    )

    # Check for key GitHub API concepts in Claude's response
    api_concepts = [
        "api", "endpoint", "github", "repository", "user", 
        "documentation", "resource", "url", "response"
    ]
    matching_concepts = [concept for concept in api_concepts if concept in response_text]
    assert len(matching_concepts) >= 3, f"Claude should understand the API response (found concepts: {matching_concepts})"

@pytest.mark.system
@pytest.mark.llm
def test_error_handling_system(requests_tool):
    """Test that Claude can handle errors from a real endpoint appropriately."""
    tools = {requests_tool.name: requests_tool}

    system_prompt = """You are a helpful AI assistant. Use the HTTP request tool to make web requests.
    If you encounter any errors, explain what went wrong and what could be done to fix it."""

    user_message = "Make a GET request to https://api.github.com/nonexistent"

    print("\n=== Initial Claude Response ===")
    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    for block in response.content:
        print(f"\nBlock Type: {block.type}")
        if hasattr(block, "text"):
            print(f"Text: {block.text}")
        elif block.type == "tool_use":
            print(f"Tool Call: {block.name}")
            print(f"Tool Input: {block.input}")

    tool_calls = get_tool_calls(response)
    assert len(tool_calls) > 0

    tool_call = tool_calls[0]
    assert tool_call.name == "http_request"
    assert "nonexistent" in tool_call.input.get("url")

    # Execute the tool call (will get a real 404 error)
    result = execute_tool_call(tool_call, tools)

    # Verify error in result
    assert isinstance(result, dict)
    assert "404" in result["content"] or "not found" in result["content"].lower()

    # Send error back to Claude for analysis
    print("\n=== Sending Error to Claude ===")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "dummy_key"))
    final_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response.content[0].text},
            {"role": "assistant", "content": f"The request failed: {result['content']}"}
        ],
        temperature=0
    )

    print("\n=== Claude's Error Analysis ===")
    for block in final_response.content:
        print(f"\nBlock Type: {block.type}")
        if hasattr(block, "text"):
            print(f"Text: {block.text}")

    # Verify Claude understands and explains the error
    response_text = " ".join(
        block.text.lower()
        for block in final_response.content
        if hasattr(block, "text")
    )

    error_concepts = ["404", "not found", "error", "invalid", "endpoint", "exist"]
    matching_concepts = [concept for concept in error_concepts if concept in response_text]
    assert len(matching_concepts) >= 2, f"Claude should explain the error (found concepts: {matching_concepts})"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 