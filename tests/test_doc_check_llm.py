"""Tests for DocCheckTool LLM integration.

This module tests the DocCheckTool's ability to be used correctly by an LLM,
verifying that the tool interface and responses are properly understood and utilized.
"""

import os
import pytest
from typing import Dict, Any
from anthropic import Anthropic
from tools.doc_check_tool import DocCheckTool
from tools.config import Config

# Load config with API keys
config = Config()

# Test fixtures
@pytest.fixture
def doc_check_tool(tmp_path):
    """Create a DocCheckTool instance with a temporary docs directory."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    return DocCheckTool(docs_root=str(docs_dir))

@pytest.fixture
def sample_md_file(tmp_path):
    """Create a sample markdown file for testing."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(exist_ok=True)
    file_path = docs_dir / "test.md"
    content = """# Test Document
    
## Introduction
This is a test document.

## API Reference
Some API details.

[Valid Link](other.md)
[Broken Link](nonexistent.md)
"""
    file_path.write_text(content)
    return file_path

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

@pytest.mark.llm
def test_basic_doc_check_llm(doc_check_tool, sample_md_file):
    """Test basic documentation check functionality through LLM interface."""
    tools = {doc_check_tool.name: doc_check_tool}
    
    system_prompt = """You are a helpful AI assistant. Use the documentation check tool to validate documentation files.
    The tool requires:
    - path: Path to the file to check
    - check_type: Type of check to perform (completeness, links, formatting, sites, all)
    - required_sections: Optional list of required section names
    """
    
    user_message = "Please check the test.md file for all issues, making sure it has Introduction and API Reference sections."
    
    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    assert response.role == "assistant"
    
    tool_calls = get_tool_calls(response)
    assert len(tool_calls) > 0
    assert tool_calls[0].name == "documentation_check"
    
    result = execute_tool_call(tool_calls[0], tools)
    assert result["type"] == "tool_response"
    content = result["content"]
    content = content if isinstance(content, str) else str(content)
    assert "test.md" in content

@pytest.mark.llm
def test_completeness_check_llm(doc_check_tool, sample_md_file):
    """Test documentation completeness checking through LLM interface."""
    tools = {doc_check_tool.name: doc_check_tool}
    
    system_prompt = """You are a helpful AI assistant. Use the documentation check tool to validate required sections."""
    
    user_message = "Check if test.md has all required sections including 'Introduction' and 'Missing Section'."
    
    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    tool_calls = get_tool_calls(response)
    result = execute_tool_call(tool_calls[0], tools)
    
    content = result["content"]
    content = content if isinstance(content, str) else str(content)
    assert "Missing Section" in content

@pytest.mark.llm
def test_links_check_llm(doc_check_tool, sample_md_file):
    """Test link validation through LLM interface."""
    tools = {doc_check_tool.name: doc_check_tool}
    
    system_prompt = """You are a helpful AI assistant. Use the documentation check tool to validate links."""
    
    user_message = "Check test.md for broken links."
    
    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    tool_calls = get_tool_calls(response)
    result = execute_tool_call(tool_calls[0], tools)
    
    content = result["content"]
    content = content if isinstance(content, str) else str(content)
    assert "nonexistent.md" in content

@pytest.mark.llm
def test_sites_check_llm(doc_check_tool):
    """Test external sites checking through LLM interface."""
    tools = {doc_check_tool.name: doc_check_tool}
    
    system_prompt = """You are a helpful AI assistant. Use the documentation check tool to check external sites."""
    
    user_message = "Check the Anthropic documentation site for updates."
    
    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    tool_calls = get_tool_calls(response)
    result = execute_tool_call(tool_calls[0], tools)
    
    content = result["content"]
    content = content if isinstance(content, str) else str(content)
    assert any(url in content for url in ["docs.anthropic.com", "anthropic.com/docs"])

@pytest.mark.llm
def test_recursive_directory_check_llm(doc_check_tool, tmp_path):
    """Test recursive directory checking through LLM interface."""
    tools = {doc_check_tool.name: doc_check_tool}
    
    # Create nested directory structure
    docs_dir = tmp_path / "docs"
    subdir = docs_dir / "subdir"
    subdir.mkdir(parents=True)
    
    (docs_dir / "root.md").write_text("# Root Doc\n## Section")
    (subdir / "nested.md").write_text("# Nested Doc\n## Section")
    
    system_prompt = """You are a helpful AI assistant. Use the documentation check tool to check directories recursively."""
    
    user_message = "Check all documentation files recursively in the current directory."
    
    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    tool_calls = get_tool_calls(response)
    result = execute_tool_call(tool_calls[0], tools)
    
    content = result["content"]
    content = content if isinstance(content, str) else str(content)
    assert "root.md" in content
    assert "nested.md" in content

@pytest.mark.llm
def test_error_handling_llm(doc_check_tool):
    """Test error handling through LLM interface."""
    tools = {doc_check_tool.name: doc_check_tool}
    
    system_prompt = """You are a helpful AI assistant. Use the documentation check tool to handle errors."""
    
    user_message = "Check a non-existent file called 'nonexistent.md'."
    
    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    tool_calls = get_tool_calls(response)
    result = execute_tool_call(tool_calls[0], tools)
    
    content = result["content"]
    content = content if isinstance(content, str) else str(content)
    assert "error" in content.lower()
    assert "not found" in content.lower()

@pytest.mark.llm
def test_formatting_check_llm(doc_check_tool, sample_md_file):
    """Test formatting validation through LLM interface."""
    tools = {doc_check_tool.name: doc_check_tool}
    
    system_prompt = """You are a helpful AI assistant. Use the documentation check tool to validate formatting."""
    
    user_message = "Check test.md for formatting issues."
    
    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    tool_calls = get_tool_calls(response)
    result = execute_tool_call(tool_calls[0], tools)
    
    assert result["type"] == "tool_response"
    assert isinstance(result["content"], (str, dict))

@pytest.mark.llm
def test_invalid_check_type_llm(doc_check_tool, sample_md_file):
    """Test handling of invalid check type through LLM interface."""
    tools = {doc_check_tool.name: doc_check_tool}
    
    system_prompt = """You are a helpful AI assistant. Use the documentation check tool to validate documentation files.
    The tool requires a valid check_type from: completeness, links, formatting, sites, all."""
    
    user_message = "Check test.md using an invalid check type 'invalid_type'."
    
    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    tool_calls = get_tool_calls(response)
    
    # Claude might not make a tool call with an invalid check type
    if not tool_calls:
        assert "invalid" in response.content[0].text.lower()
        return
        
    result = execute_tool_call(tool_calls[0], tools)
    content = result["content"]
    if isinstance(content, dict):
        content = str(content.get("error", "")) if "error" in content else str(content)
    elif not isinstance(content, str):
        content = str(content)
    
    assert "error" in content.lower() or "invalid" in content.lower() 