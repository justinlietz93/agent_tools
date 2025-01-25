"""
Shared test fixtures and utilities for LLM tool testing.

This module provides fixtures and utilities for testing the Anthropic Tool Use framework,
including both mocked and real LLM interactions.
"""

import pytest
import os
import tempfile
from typing import AsyncGenerator, Dict, Any, List
from anthropic import Anthropic
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

@pytest.fixture(scope="session")
def test_env():
    """Get the test environment type."""
    return os.getenv("TEST_ENV", "unit")

@pytest.fixture
def mock_claude_response():
    """Return a mock Claude response for testing."""
    def _mock_response(message: str) -> str:
        return f"Mock Claude Response: {message}"
    return _mock_response

@pytest.fixture
def real_claude_response(test_env):
    """Get a real response from Claude based on test environment."""
    def _get_response(message: str) -> str:
        if test_env == "unit":
            return f"Mock Claude Response: {message}"
            
        response = anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": message
            }]
        )
        return response.content[0].text
    return _get_response

@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

def create_test_file(path: str, content: str) -> None:
    """Create a test file with given content"""
    with open(path, "w") as f:
        f.write(content)

def get_tool_calls(response) -> List[Dict[str, Any]]:
    """Extract tool calls from Claude's response.
    
    Args:
        response: Claude's response object
        
    Returns:
        List of tool call dictionaries with name and input_schema
    """
    tool_calls = []
    
    for content in response.content:
        if content.type == "tool_calls":
            for tool_call in content.tool_calls:
                tool_calls.append({
                    "name": tool_call.name,
                    "input_schema": tool_call.input_schema
                })
    
    return tool_calls

def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from a text response.
    
    This is a backup method for when we can't access the structured response.
    It looks for tool calls in the format:
    <tool>name: input_schema</tool>
    
    Args:
        text: Response text containing tool calls
        
    Returns:
        List of tool call dictionaries with name and input_schema
    """
    tool_calls = []
    
    # Simple regex-free parsing
    lines = text.split("\n")
    current_tool = None
    current_params = []
    
    for line in lines:
        line = line.strip()
        if line.startswith("<tool>"):
            if current_tool:
                # Parse previous tool
                tool_calls.append(parse_tool_call(current_tool, "\n".join(current_params)))
            current_tool = line[6:].strip()  # Remove <tool> prefix
            current_params = []
        elif line.endswith("</tool>"):
            current_params.append(line[:-7].strip())  # Remove </tool> suffix
            if current_tool:
                tool_calls.append(parse_tool_call(current_tool, "\n".join(current_params)))
            current_tool = None
            current_params = []
        elif current_tool:
            current_params.append(line)
            
    return tool_calls

def parse_tool_call(name: str, params_text: str) -> Dict[str, Any]:
    """Parse a tool call from name and input_schema text."""
    try:
        params = json.loads(params_text)
    except json.JSONDecodeError:
        params = {"error": f"Failed to parse input_schema: {params_text}"}
        
    return {
        "name": name,
        "input_schema": params  # Changed from input_schema
    } 