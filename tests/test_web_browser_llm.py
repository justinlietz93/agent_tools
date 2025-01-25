"""
Tests for WebBrowserTool with actual LLM interactions.
Verifies that Claude can properly understand and use the web browser tool.
"""

import pytest
from tools.web_browser_tool import WebBrowserTool
from tools.config import Config
from anthropic import Anthropic
import responses  # For mocking HTTP requests
from tools.tool_base import ToolResult
from typing import Dict, Any
import requests

# Load config with API keys
config = Config()

class ConcreteWebBrowserTool(WebBrowserTool):
    """Concrete implementation of WebBrowserTool for testing"""
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return self.input_schema

@pytest.fixture
def web_browser():
    """Create WebBrowserTool instance"""
    return ConcreteWebBrowserTool()

def get_claude_response(system_prompt: str, user_message: str, tools: list) -> dict:
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
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
        tools=tool_definitions,
        temperature=0
    )
    
    return response

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
        "content": result
    }

def get_tool_calls(response) -> list:
    """Extract tool calls from response content."""
    return [
        block for block in response.content 
        if block.type == "tool_use"
    ]

@pytest.mark.real_http
def test_basic_text_extraction_llm(web_browser):
    """Test fetching and extracting text from a real webpage through LLM interaction"""
    tools = {web_browser.name: web_browser}
    
    # Use Anthropic's API docs as a real and meaningful test case
    test_url = "https://docs.anthropic.com/en/api/getting-started"
    
    # System prompt that explains the tool's purpose
    system_prompt = """You are a helpful AI assistant. Use the web browser tool to fetch and read webpage content.
    The tool can extract text content, links, and titles from web pages."""
    
    # Ask the LLM to fetch the content
    user_message = f"Please check what content is on {test_url} and tell me what you find."
    
    # Get LLM's response with tool use
    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    
    # Debug: Print response content
    print("\nLLM Response Content:")
    for block in response.content:
        print(f"Block type: {block.type}")
        if hasattr(block, "text"):
            print(f"Text: {block.text}")
    
    # Verify LLM used the tool correctly
    tool_calls = get_tool_calls(response)
    assert len(tool_calls) > 0, "LLM should have used the web browser tool"
    
    tool_call = tool_calls[0]
    assert tool_call.name == "web_browser", "LLM should use the web browser tool"
    assert tool_call.input.get("url") == test_url, "LLM should use the provided URL"
    
    # Execute the tool call
    result = execute_tool_call(tool_call, tools)
    
    # Debug: Print result structure
    print("\nTool Result:")
    print(f"Type: {result['type']}")
    print(f"Tool Use ID: {result['tool_use_id']}")
    print(f"Content: {result['content']}")
    
    # Verify the result format
    assert isinstance(result, dict)
    assert "content" in result
    assert "tool_use_id" in result
    assert "type" in result
    
    # Verify we got some content (exact content may change)
    content = result["content"]["content"] if isinstance(result["content"], dict) else result["content"]
    assert len(content) > 0, "Should get some content back"
    assert "Getting started" in content, "Should find the page title"
    
    # Feed the result back to the LLM as part of the conversation
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    final_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response.content[0].text},
            {"role": "assistant", "content": f"I've checked the webpage and found: {content}"}
        ],
        temperature=0
    )
    
    # Debug: Print final response
    print("\nFinal LLM Response:")
    for block in final_response.content:
        print(f"Block type: {block.type}")
        if hasattr(block, "text"):
            print(f"Text: {block.text}")
    
    # Verify LLM can understand and describe the response
    assert any(
        block.type == "text" and (
            "API" in block.text and 
            any(keyword in block.text.lower() for keyword in [
                "documentation", 
                "authentication", 
                "content types",
                "examples"
            ])
        )
        for block in final_response.content
    ), "LLM should acknowledge and describe the API documentation content"

@responses.activate
def test_link_extraction_llm(web_browser):
    """Test extracting links from a webpage"""
    test_url = "https://example.com"
    test_html = """
    <html>
        <body>
            <a href="https://link1.com">Link 1</a>
            <a href="https://link2.com">Link 2</a>
        </body>
    </html>
    """
    responses.add(responses.GET, test_url, body=test_html, status=200)
    
    result = web_browser.run("test_call", url=test_url, extract_type="links")
    assert isinstance(result, dict)
    assert "content" in result
    assert "tool_use_id" in result
    assert "type" in result
    assert "link1.com" in result["content"]
    assert "link2.com" in result["content"]

@responses.activate
def test_title_extraction_llm(web_browser):
    """Test extracting page title"""
    test_url = "https://example.com"
    test_html = """
    <html>
        <head>
            <title>Test Title</title>
        </head>
        <body>
            <h1>Content</h1>
        </body>
    </html>
    """
    responses.add(responses.GET, test_url, body=test_html, status=200)
    
    result = web_browser.run("test_call", url=test_url, extract_type="title")
    assert isinstance(result, dict)
    assert "content" in result
    assert "tool_use_id" in result
    assert "type" in result
    assert "Test Title" in result["content"]

@responses.activate
def test_error_handling_llm(web_browser):
    """Test how Claude handles web errors"""
    test_url = "https://example.com/404"
    responses.add(responses.GET, test_url, status=404)
    
    result = web_browser.run("test_call", url=test_url)
    assert isinstance(result, dict)
    assert "content" in result
    assert "tool_use_id" in result
    assert "type" in result
    assert result["content"].startswith("Error:")
    assert "404" in result["content"]

@responses.activate
def test_timeout_handling_llm(web_browser):
    """Test how Claude handles timeouts"""
    test_url = "https://example.com/slow"
    
    def request_callback(request):
        raise requests.exceptions.Timeout("Request timed out")
    
    responses.add_callback(
        responses.GET,
        test_url,
        callback=request_callback
    )
    
    result = web_browser.run("test_call", url=test_url, timeout=1)
    assert isinstance(result, dict)
    assert "content" in result
    assert "tool_use_id" in result
    assert "type" in result
    assert result["content"].startswith("Error:")
    assert "timed out" in result["content"].lower()

@pytest.mark.real_http
def test_link_following_llm(web_browser):
    """Test the LLM's ability to discover and follow links from a webpage"""
    tools = {web_browser.name: web_browser}
    
    # Start with Anthropic's API docs which has links to other sections
    start_url = "https://docs.anthropic.com/en/api/getting-started"
    
    # System prompt that explains the tool's purpose and encourages link exploration
    system_prompt = """You are a helpful AI assistant. Use the web browser tool to fetch and read webpage content.
    You can extract links using extract_type="links" and then visit any relevant links you find.
    When exploring documentation, first get the list of available links, then navigate to the most relevant one."""
    
    # First ask the LLM to get the list of links
    user_message = f"""First use extract_type="links" to get all links from {start_url}, 
    then find and follow the link related to rate limits."""
    
    # Get LLM's first response - it should extract links
    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    
    # Debug: Print response content
    print("\nInitial LLM Response Content:")
    for block in response.content:
        print(f"Block type: {block.type}")
        if hasattr(block, "text"):
            print(f"Text: {block.text}")
    
    # Verify first tool call is for link extraction
    tool_calls = get_tool_calls(response)
    assert len(tool_calls) > 0, "LLM should use the web browser tool"
    
    first_call = tool_calls[0]
    assert first_call.name == "web_browser", "LLM should use the web browser tool"
    assert first_call.input.get("extract_type") == "links", "First call should extract links"
    assert first_call.input.get("url") == start_url, "Should use the starting URL"
    
    # Execute the first tool call to get the links
    first_result = execute_tool_call(first_call, tools)
    links_content = first_result["content"]["content"] if isinstance(first_result["content"], dict) else first_result["content"]
    
    print("\nExtracted Links:")
    print(links_content)
    
    # Now have the LLM analyze the links and choose the rate limits one
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    second_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "I'll help you find the rate limits documentation."},
            {"role": "assistant", "content": f"Here are the links I found:\n{links_content}\n\nI'll look for the rate limits link and visit it."}
        ],
        tools=[tool.get_tool_definition() for tool in tools.values()],
        temperature=0
    )
    
    # Debug: Print second response
    print("\nLLM Response After Link Analysis:")
    for block in second_response.content:
        print(f"Block type: {block.type}")
        if hasattr(block, "text"):
            print(f"Text: {block.text}")
    
    # The LLM should make another tool call to visit the chosen link
    second_tool_calls = get_tool_calls(second_response)
    assert len(second_tool_calls) > 0, "LLM should make another tool call to visit the chosen link"
    
    second_call = second_tool_calls[0]
    assert second_call.name == "web_browser", "LLM should use the web browser tool again"
    assert second_call.input.get("extract_type") == "text", "Second call should get full text"
    chosen_url = second_call.input.get("url")
    assert chosen_url and "rate" in chosen_url.lower(), f"LLM should choose a rate limits related URL, got: {chosen_url}"
    
    # Execute the second tool call to get the page content
    second_result = execute_tool_call(second_call, tools)
    page_content = second_result["content"]["content"] if isinstance(second_result["content"], dict) else second_result["content"]
    
    print("\nRate Limits Page Content:")
    print(page_content)
    
    # Have the LLM summarize what it found
    final_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": "What did you find about rate limits?"},
            {"role": "assistant", "content": f"I found and visited this page about rate limits: {chosen_url}\n\nHere's what it says:\n{page_content}"}
        ],
        temperature=0
    )
    
    # Debug: Print final response
    print("\nFinal LLM Summary:")
    for block in final_response.content:
        print(f"Block type: {block.type}")
        if hasattr(block, "text"):
            print(f"Text: {block.text}")
    
    # Verify the LLM found and understood the rate limits information
    assert any(
        block.type == "text" and all(
            term in block.text.lower() 
            for term in ["rate", "limit"]
        )
        for block in final_response.content
    ), "LLM should provide information about rate limits"

@pytest.mark.real_http
def test_breadcrumb_navigation_llm(web_browser):
    """Test the LLM's ability to use breadcrumb navigation on a webpage"""
    tools = {web_browser.name: web_browser}
    
    # Start with a deep documentation page
    start_url = "https://docs.anthropic.com/en/api/rate-limits"
    
    # System prompt that explains breadcrumb navigation
    system_prompt = """You are a helpful AI assistant. Use the web browser tool to fetch and read webpage content.
    When navigating documentation, you can:
    1. Look for breadcrumb navigation (usually at the top of the page showing the hierarchy like "Home > Section > Subsection")
    2. Use these breadcrumbs to navigate to parent sections
    3. Extract both the current page content and understand its place in the documentation hierarchy"""
    
    # Ask the LLM to analyze the page hierarchy
    user_message = f"""Please check {start_url} and tell me:
    1. What section of the documentation we're in
    2. What the parent sections are
    3. A summary of the current page content"""
    
    # Get LLM's response
    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    
    # Debug: Print response content
    print("\nInitial LLM Response Content:")
    for block in response.content:
        print(f"Block type: {block.type}")
        if hasattr(block, "text"):
            print(f"Text: {block.text}")
    
    # Verify LLM used the tool
    tool_calls = get_tool_calls(response)
    assert len(tool_calls) > 0, "LLM should use the web browser tool"
    
    first_call = tool_calls[0]
    assert first_call.name == "web_browser", "LLM should use the web browser tool"
    
    # Execute the tool call
    result = execute_tool_call(first_call, tools)
    content = result["content"]["content"] if isinstance(result["content"], dict) else result["content"]
    
    # Have the LLM analyze the page structure
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    final_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"I've found this content:\n{content}"}
        ],
        temperature=0
    )
    
    # Debug: Print final response
    print("\nFinal LLM Analysis:")
    for block in final_response.content:
        print(f"Block type: {block.type}")
        if hasattr(block, "text"):
            print(f"Text: {block.text}")
    
    # Verify the LLM understood the page hierarchy
    assert any(
        block.type == "text" and (
            "using the api" in block.text.lower() and  # Parent section
            "rate limits" in block.text.lower() and    # Current section
            any(hierarchy_term in block.text.lower() for hierarchy_term in [
                "section",
                "page",
                "hierarchy",
                "parent"
            ])
        )
        for block in final_response.content
    ), "LLM should understand the page's location in the documentation hierarchy"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 