"""
Tests for DocCheckTool with actual LLM interactions.
Verifies that Claude can properly understand and use the tool for documentation validation.
"""

import os
import pytest
from unittest.mock import Mock, patch
from tools.doc_check_tool import DocCheckTool
from anthropic import Anthropic
import tempfile
import shutil

# Test Data
SAMPLE_DOC = """# Test Documentation

## Overview
This is a test document.

## Usage
[Valid Link](https://www.google.com)
[Broken Link](https://nonexistent.example.com)

## API Reference
Some API details.
"""

REQUIRED_SECTIONS = ["Overview", "Usage", "API Reference"]

# Fixtures
@pytest.fixture
def doc_check_tool():
    """Create a DocCheckTool instance with a temporary docs directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test documentation
        os.makedirs(os.path.join(temp_dir, "docs"))
        with open(os.path.join(temp_dir, "docs", "test.md"), "w") as f:
            f.write(SAMPLE_DOC)
        
        tool = DocCheckTool(docs_root=os.path.join(temp_dir, "docs"))
        yield tool

@pytest.fixture
def claude_client():
    """Create an Anthropic client for testing."""
    return Anthropic()

# Helper Functions
def get_claude_response(client, system_prompt, user_message, tools):
    """Get response from Claude with tool definitions."""
    return client.messages.create(
        model="claude-3-5-sonnet-20241022",
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        tools=tools,
        temperature=0
    )

# Unit Tests
@pytest.mark.unit
def test_tool_definition(doc_check_tool):
    """Verify tool definition format."""
    definition = doc_check_tool.get_tool_definition()
    assert definition["name"] == "documentation_check"
    assert "description" in definition
    assert "input_schema" in definition
    assert "check_type" in definition["input_schema"]["properties"]
    assert "path" in definition["input_schema"]["properties"]

@pytest.mark.unit
def test_basic_functionality(doc_check_tool):
    """Test core functionality with mocks."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "Sample documentation"
        
        result = doc_check_tool.run({
            "tool_use_id": "test",
            "check_type": "all",
            "path": "test.md",
            "required_sections": REQUIRED_SECTIONS
        })
        
        assert "type" in result
        assert "content" in result
        assert "tool_use_id" in result

@pytest.mark.unit
def test_error_handling(doc_check_tool):
    """Test error cases with mocks."""
    result = doc_check_tool.run({
        "tool_use_id": "test",
        "check_type": "all",
        "path": "nonexistent.md"
    })
    
    assert result["type"] == "error"
    assert "not found" in result["content"]

@pytest.mark.unit
def test_parameter_validation(doc_check_tool):
    """Test input parameter validation."""
    result = doc_check_tool.run({
        "tool_use_id": "test",
        "check_type": "invalid_type",
        "path": "test.md"
    })
    
    assert result["type"] == "error"

# Integration Tests
@pytest.mark.integration
@pytest.mark.llm
def test_basic_interaction(doc_check_tool, claude_client):
    """Test LLM understands and uses the tool."""
    system_prompt = """You are a documentation reviewer. Use the documentation_check tool to validate documentation files."""
    user_message = """Please check the test.md file for completeness and formatting."""
    
    response = get_claude_response(
        claude_client,
        system_prompt,
        user_message,
        [doc_check_tool.get_tool_definition()]
    )
    
    tool_calls = [block for block in response.content if block.type == "tool_use"]
    assert len(tool_calls) > 0
    assert tool_calls[0].name == "documentation_check"

@pytest.mark.integration
@pytest.mark.llm
def test_error_handling_llm(doc_check_tool, claude_client):
    """Test LLM handles errors appropriately."""
    system_prompt = """You are a documentation reviewer. Use the documentation_check tool to validate documentation files."""
    user_message = """Please check a nonexistent.md file."""
    
    response = get_claude_response(
        claude_client,
        system_prompt,
        user_message,
        [doc_check_tool.get_tool_definition()]
    )
    
    tool_calls = [block for block in response.content if block.type == "tool_use"]
    assert len(tool_calls) > 0
    
    result = doc_check_tool.run(tool_calls[0].input)
    assert result["type"] == "error"
    
    final_response = get_claude_response(
        claude_client,
        system_prompt,
        user_message,
        [doc_check_tool.get_tool_definition()]
    )
    
    assert any("not found" in block.text.lower() for block in final_response.content if block.type == "text")

@pytest.mark.integration
@pytest.mark.llm
def test_complex_interaction(doc_check_tool, claude_client):
    """Test multi-turn interactions."""
    system_prompt = """You are a documentation reviewer. Use the documentation_check tool to validate documentation files."""
    user_message = """First check test.md for completeness, then check external documentation sites."""
    
    # First interaction
    response = get_claude_response(
        claude_client,
        system_prompt,
        user_message,
        [doc_check_tool.get_tool_definition()]
    )
    
    tool_calls = [block for block in response.content if block.type == "tool_use"]
    assert len(tool_calls) > 0
    
    # Second interaction
    result = doc_check_tool.run(tool_calls[0].input)
    final_response = get_claude_response(
        claude_client,
        system_prompt,
        f"Here's what I found: {result['content']}",
        [doc_check_tool.get_tool_definition()]
    )
    
    assert any("sites" in block.text.lower() for block in final_response.content if block.type == "text")

# System Tests
@pytest.mark.system
@pytest.mark.llm
def test_basic_functionality_system(doc_check_tool, claude_client):
    """Test basic end-to-end flow with real dependencies."""
    print("\n=== Initial Claude Response ===")
    system_prompt = """You are a documentation reviewer. Use the documentation_check tool to validate documentation files."""
    user_message = """Please perform a complete check of test.md and the default documentation sites."""
    
    response = get_claude_response(
        claude_client,
        system_prompt,
        user_message,
        [doc_check_tool.get_tool_definition()]
    )
    print(response.content)
    
    tool_calls = [block for block in response.content if block.type == "tool_use"]
    assert len(tool_calls) > 0
    
    print("\n=== Tool Result ===")
    result = doc_check_tool.run(tool_calls[0].input)
    print(result)
    
    print("\n=== Claude's Analysis ===")
    final_response = get_claude_response(
        claude_client,
        system_prompt,
        f"Here's what I found: {result['content']}",
        [doc_check_tool.get_tool_definition()]
    )
    print(final_response.content)
    
    # Verify end-to-end functionality
    assert result["type"] != "error"
    assert any("documentation" in block.text.lower() for block in final_response.content if block.type == "text") 