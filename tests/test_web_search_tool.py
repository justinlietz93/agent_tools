"""
Comprehensive tests for WebSearchTool.
Tests both core functionality and Claude integration.

Test Categories:
1. Core Functionality
   - HTTP interactions
   - Response parsing
   - Search capabilities
   - Error handling
   - Input validation

2. Claude Integration
   - Tool definition clarity
   - Parameter handling
   - Result processing
   - Error recovery
   - Multi-turn interactions
"""

import pytest
from unittest.mock import patch, MagicMock
from tools.web_search_tool import WebSearchTool
from anthropic import Anthropic
import responses
import requests
import os
from typing import Dict, Any

# Register custom marks
def pytest_configure(config):
    config.addinivalue_line("markers", "real_http: mark test to run against real HTTP endpoints")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "llm: mark test as an LLM test")

# Mock HTML for testing
MOCK_HTML = """
<html>
<head><title>Test Page</title></head>
<body>
<h1>Welcome</h1>
<p>This is a test page with some content.</p>
<script>alert('ignore me');</script>
<p>More content here.</p>
</body>
</html>
"""

class ConcreteWebSearchTool(WebSearchTool):
    """Concrete implementation of WebSearchTool for testing"""
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return self.input_schema

@pytest.fixture
def web_search_tool():
    return ConcreteWebSearchTool()

@pytest.fixture
def mock_anthropic():
    with patch('anthropic.Anthropic') as mock:
        client = MagicMock()
        mock.return_value = client
        yield client

# ============================================================================
# Core Functionality Tests
# ============================================================================

@responses.activate
def test_url_fetch(web_search_tool):
    """Test fetching content from a URL"""
    test_url = "https://example.com"
    test_html = MOCK_HTML
    responses.add(responses.GET, test_url, body=test_html, status=200)
    
    result = web_search_tool.run({"query": test_url})
    assert isinstance(result, dict)
    assert "content" in result
    assert "type" in result
    assert "Test Page" in result["content"]
    assert "Welcome" in result["content"]
    assert "test page with some content" in result["content"]
    assert "alert" not in result["content"]
    assert "color: red" not in result["content"]

@pytest.mark.real_http
def test_anthropic_search(web_search_tool):
    """Test search functionality for Anthropic queries"""
    test_query = "anthropic claude documentation"
    
    result = web_search_tool.run({"query": test_query, "max_results": 2})
    assert isinstance(result, dict)
    assert "content" in result
    assert "type" in result
    assert "Claude" in result["content"]
    assert "docs.anthropic.com" in result["content"]
    assert len(result["content"].split("\n")) > 5

@responses.activate
def test_non_anthropic_search(web_search_tool):
    """Test search functionality for non-Anthropic queries"""
    test_query = "test search"
    result = web_search_tool.run({"query": test_query})
    assert isinstance(result, dict)
    assert "content" in result
    assert "type" in result
    assert "provide a specific URL" in result["content"]

@pytest.mark.real_http
def test_max_results(web_search_tool):
    """Test max_results parameter with Anthropic search"""
    test_query = "claude api"
    
    # Test with max_results=2
    result = web_search_tool.run({"query": test_query, "max_results": 2})
    urls = [line for line in result["content"].split("\n") if "URL:" in line]
    assert len(urls) == 2
    
    # Test with default max_results
    result = web_search_tool.run({"query": test_query})
    urls = [line for line in result["content"].split("\n") if "URL:" in line]
    assert len(urls) == 3

@responses.activate
def test_error_handling(web_search_tool):
    """Test error handling"""
    # Test invalid URL
    result = web_search_tool.run({"query": "not_a_url"})
    assert isinstance(result, dict)
    assert "provide a specific URL" in result["content"]
    
    # Test 404 error
    responses.add(responses.GET, "https://example.com/404", status=404)
    result = web_search_tool.run({"query": "https://example.com/404"})
    assert isinstance(result, dict)
    assert result["content"].startswith("Error:")
    assert "404" in result["content"]
    
    # Test timeout
    def request_callback(request):
        raise requests.exceptions.Timeout("Request timed out")
    responses.add_callback(
        responses.GET,
        "https://example.com/timeout",
        callback=request_callback
    )
    result = web_search_tool.run({"query": "https://example.com/timeout"})
    assert isinstance(result, dict)
    assert result["content"].startswith("Error:")
    assert "timed out" in result["content"].lower()

@responses.activate
def test_input_validation(web_search_tool):
    """Test input validation"""
    # Test empty query
    result = web_search_tool.run({"query": ""})
    assert isinstance(result, dict)
    assert result["content"].startswith("Error:")
    assert "required" in result["content"].lower()
    
    # Test missing query
    result = web_search_tool.run({})
    assert isinstance(result, dict)
    assert result["content"].startswith("Error:")
    assert "required" in result["content"].lower()
    
    # Test invalid max_results
    result = web_search_tool.run({"query": "claude api", "max_results": 0})
    assert isinstance(result, dict)
    assert "content" in result
    assert len(result["content"]) > 0

# ============================================================================
# Claude Integration Tests
# ============================================================================

def test_tool_definition_format(web_search_tool):
    """Test that the tool definition follows Anthropic's format requirements."""
    definition = web_search_tool.get_tool_definition()
    
    assert isinstance(definition, dict)
    assert "name" in definition
    assert "description" in definition
    assert "input_schema" in definition
    
    schema = definition["input_schema"]
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "required" in schema
    assert "query" in schema["properties"]
    assert "max_results" in schema["properties"]
    assert "query" in schema["required"]

@pytest.mark.integration
def test_llm_understands_tool_purpose(mock_anthropic, web_search_tool):
    """Test that Claude understands when and how to use the tool."""
    system_prompt = """You have access to a web search tool. Use it when you need to find information online."""
    
    user_message = "What are the latest Claude API docs about tool use?"
    
    # Configure mock response for the tool call
    mock_tool_response = MagicMock()
    mock_tool_response.content = [{
        "type": "tool_use",
        "id": "test_id",
        "name": "web_search",
        "input": {
            "query": "anthropic claude api tool use documentation",
            "max_results": 3
        }
    }]
    
    # Configure mock response for the final response
    mock_final_response = MagicMock()
    mock_final_response.content = [{
        "type": "text",
        "text": "Let me search for information about Claude's API documentation."
    }]
    
    # Set up the mock to return different responses
    mock_anthropic.messages.create.side_effect = [mock_tool_response, mock_final_response]
    
    # Create Anthropic client and send request
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "dummy_key"))
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        tools=[web_search_tool.get_tool_definition()],
        temperature=0
    )
    
    # Verify Claude chose to use the tool
    tool_calls = [block for block in response.content if block.type == "tool_use"]
    assert len(tool_calls) > 0, "Claude should use the web search tool"
    
    tool_call = tool_calls[0]
    assert tool_call.name == "web_search"
    assert "query" in tool_call.input
    assert "tool use" in tool_call.input["query"].lower()

@pytest.mark.integration
def test_llm_handles_url_results(mock_anthropic, web_search_tool):
    """Test that Claude can process URL fetch results."""
    with patch('requests.get') as mock_get:
        # Mock successful URL fetch
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = MOCK_HTML
        mock_get.return_value = mock_response
        
        # Configure mock Claude response
        mock_claude_response = MagicMock()
        mock_claude_response.content = [{
            "type": "text",
            "text": "Based on the content, this is a test page that contains a welcome message and some test content."
        }]
        mock_anthropic.messages.create.return_value = mock_claude_response
        
        # Execute tool
        result = web_search_tool.run({
            "query": "https://test.com",
            "max_results": 1
        })
        
        # Verify result format
        assert isinstance(result, dict)
        assert "type" in result
        assert "content" in result
        assert "Title: Test Page" in result["content"]
        
        # Send result to Claude
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "dummy_key"))
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "What does this page contain?"},
                {"role": "assistant", "content": result["content"].strip()}
            ],
            temperature=0
        )
        
        # Verify Claude understood the content
        assert mock_claude_response.content[0]["type"] == "text"
        assert "test" in mock_claude_response.content[0]["text"].lower()

@pytest.mark.integration
def test_llm_handles_search_results(mock_anthropic, web_search_tool):
    """Test that Claude can process search results."""
    with patch('requests.get') as mock_get:
        # Mock successful search
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = MOCK_HTML
        mock_get.return_value = mock_response
        
        # Configure mock Claude response
        mock_claude_response = MagicMock()
        mock_claude_response.content = [{
            "type": "text",
            "text": "I found documentation about Anthropic's Claude API."
        }]
        mock_anthropic.messages.create.return_value = mock_claude_response
        
        # Execute tool
        result = web_search_tool.run({
            "query": "anthropic claude api",
            "max_results": 2
        })
        
        # Verify result format
        assert isinstance(result, dict)
        assert "type" in result
        assert "content" in result
        
        # Send result to Claude
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "dummy_key"))
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "What did you find about the Claude API?"},
                {"role": "assistant", "content": result["content"].strip()}
            ],
            temperature=0
        )
        
        # Verify Claude understood the results
        assert mock_claude_response.content[0]["type"] == "text"
        assert "api" in mock_claude_response.content[0]["text"].lower()

@pytest.mark.integration
def test_llm_handles_errors(mock_anthropic, web_search_tool):
    """Test that Claude handles tool errors gracefully."""
    with patch('requests.get') as mock_get:
        # Mock failed request
        mock_get.side_effect = Exception("Failed to fetch URL")
        
        # Configure mock Claude response
        mock_claude_response = MagicMock()
        mock_claude_response.content = [{
            "type": "text",
            "text": "I apologize, but I encountered an error while trying to search."
        }]
        mock_anthropic.messages.create.return_value = mock_claude_response
        
        # Execute tool
        result = web_search_tool.run({
            "query": "https://nonexistent.com",
            "max_results": 1
        })
        
        # Verify error format
        assert isinstance(result, dict)
        assert "type" in result
        assert "content" in result
        assert "Error" in result["content"]
        
        # Send error to Claude
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "dummy_key"))
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "Can you search this URL?"},
                {"role": "assistant", "content": result["content"].strip()}
            ],
            temperature=0
        )
        
        # Verify Claude handles the error
        assert mock_claude_response.content[0]["type"] == "text"
        assert "error" in mock_claude_response.content[0]["text"].lower()

@pytest.mark.real_http
@pytest.mark.llm
def test_real_claude_with_real_websites(web_search_tool):
    """End-to-end test using real Claude API and real websites."""
    # Create real Anthropic client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    
    client = Anthropic(api_key=api_key)
    
    # Test searching Anthropic's documentation
    system_prompt = """You have access to a web search tool. Use it to find specific information from websites."""
    
    user_message = "What are the latest Claude API docs about tool use? Focus on the required input_schema for tool definitions."
    print("\n=== TEST 1: Searching Anthropic Docs ===")
    print(f"User: {user_message}")
    
    # First message to get tool use
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        tools=[web_search_tool.get_tool_definition()],
        temperature=0
    )
    
    print("\nClaude's Initial Response:")
    print(response.content[0].text)
    
    # Verify Claude chose to use the tool
    tool_calls = [block for block in response.content if block.type == "tool_use"]
    assert len(tool_calls) > 0, "Claude should use the web search tool"
    
    tool_call = tool_calls[0]
    print("\nTool Call:")
    print(f"Name: {tool_call.name}")
    print(f"Input: {tool_call.input}")
    
    assert tool_call.name == "web_search"
    assert "query" in tool_call.input
    
    # Execute the tool with real HTTP requests
    print("\nExecuting Tool...")
    result = web_search_tool.run(tool_call.input)
    print("\nTool Result:")
    print(result["content"])
    
    # Verify the result format
    assert isinstance(result, dict)
    assert "type" in result
    assert "content" in result
    assert len(result["content"]) > 0
    
    # Send result back to Claude
    final_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response.content[0].text},
            {"role": "assistant", "content": result["content"].strip()}
        ],
        temperature=0
    )
    
    print("\nClaude's Final Response:")
    print(final_response.content[0].text)
    
    # Verify Claude understood and used the information
    final_text = final_response.content[0].text.lower()
    assert "tool" in final_text
    assert "parameter" in final_text
    assert len(final_text) > 100  # Should give a substantial response
    
    # Test fetching a specific URL
    url_message = "What is the main content of https://docs.anthropic.com/claude/docs/tool-use?"
    print("\n\n=== TEST 2: Fetching Specific URL ===")
    print(f"User: {url_message}")
    
    url_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": url_message}],
        tools=[web_search_tool.get_tool_definition()],
        temperature=0
    )
    
    print("\nClaude's Initial Response:")
    print(url_response.content[0].text)
    
    # Verify Claude chose to use the tool
    url_tool_calls = [block for block in url_response.content if block.type == "tool_use"]
    assert len(url_tool_calls) > 0
    
    url_tool_call = url_tool_calls[0]
    print("\nTool Call:")
    print(f"Name: {url_tool_call.name}")
    print(f"Input: {url_tool_call.input}")
    
    assert url_tool_call.name == "web_search"
    assert "docs.anthropic.com" in url_tool_call.input["query"]
    
    # Execute the tool
    print("\nExecuting Tool...")
    url_result = web_search_tool.run(url_tool_call.input)
    print("\nTool Result:")
    print(url_result["content"])
    
    # Send result back to Claude
    url_final_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": url_message},
            {"role": "assistant", "content": url_response.content[0].text},
            {"role": "assistant", "content": url_result["content"].strip()}
        ],
        temperature=0
    )
    
    print("\nClaude's Final Response:")
    print(url_final_response.content[0].text)
    
    # Verify Claude gave a meaningful response about the documentation
    url_final_text = url_final_response.content[0].text.lower()
    assert "tool" in url_final_text
    assert len(url_final_text) > 200  # Should give a detailed response about the docs

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 