"""
Tests for FileTool with actual LLM interactions.
Verifies that Claude can properly understand and use the tool for file operations.
"""

import pytest
from unittest.mock import Mock, patch
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.file_tool import FileTool
from tests.conftest import get_claude_response, create_test_file

def normalize_path(path):
    """Normalize path for display in messages"""
    return os.path.basename(path)

def format_message(*parts):
    """Format message parts safely, handling Windows paths"""
    return ''.join(str(part).replace('\\', '/') for part in parts)

# Remove the asyncio mark since we're not using async/await
# pytestmark = pytest.mark.asyncio

# Unit Tests (Mock LLM and Mock Tool Calls)
@pytest.mark.unit
def test_tool_definition():
    """Verify FileTool definition format and input_schema"""
    tool = FileTool()
    assert tool.name == "file"
    assert "read" in tool.description.lower()
    assert "write" in tool.description.lower()
    assert "edit" in tool.description.lower()

@pytest.mark.unit
def test_basic_read_operation():
    """Test basic file read operation with mocks"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        temp_path = f.name
    
    tool = FileTool()
    result = tool.run({
        "operation": "read",
        "path": temp_path
    })
    
    assert "test content" in result["content"]
    os.unlink(temp_path)

@pytest.mark.unit
def test_error_handling():
    """Test error cases with mocks"""
    tool = FileTool()
    result = tool.run({
        "operation": "read",
        "path": "nonexistent.txt"
    })
    assert "error" in result["content"].lower()

# Integration Tests (Real LLM with Fake Files)
@pytest.fixture
def test_dir():
    """Create a temporary test directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.mark.integration
@pytest.mark.llm
def test_basic_file_operations(test_dir):
    """Test LLM can perform basic file operations"""
    # Create a test file
    test_file = os.path.join(test_dir, "test.txt")
    content = "Hello World!"
    with open(test_file, "w") as f:
        f.write(content)
    
    # Ask Claude to read the file
    message = format_message(
        "Here is the content of test.txt:\n\n",
        content,
        "\n\nPlease confirm what you see in this file."
    )
    response = get_claude_response(message)
    assert "Hello World!" in response
    
    # Ask Claude to modify the file
    message = format_message(
        "Current content of test.txt:\n\n",
        content,
        "\n\nPlease help me add a second line saying 'How are you?' to this file."
    )
    response = get_claude_response(message)
    
    # Extract tool call from response and apply it
    tool = FileTool()
    tool.run({
        "operation": "write",
        "path": test_file,
        "content": "Hello World!\nHow are you?"
    })
    
    # Verify the modification
    with open(test_file, "r") as f:
        updated_content = f.read()
    assert "Hello World!" in updated_content
    assert "How are you?" in updated_content

@pytest.mark.integration
@pytest.mark.llm
def test_error_handling_integration(test_dir):
    """Test LLM handles file errors appropriately"""
    message = format_message(
        "I tried to read a file called nonexistent.txt but got an error saying the file does not exist. ",
        "Can you help me understand what this means?"
    )
    response = get_claude_response(message)
    assert any(phrase in response.lower() for phrase in [
        "not found",
        "does not exist",
        "cannot find",
        "missing"
    ])

@pytest.mark.integration
@pytest.mark.llm
def test_complex_operations(test_dir):
    """Test LLM can handle complex file operations"""
    # Create multiple files
    files = {
        "data.txt": "1,2,3\n4,5,6",
        "config.ini": "[settings]\nmode=test",
        "log.txt": "ERROR: test failed\nINFO: test started"
    }
    
    file_contents = []
    for name, content in files.items():
        path = os.path.join(test_dir, name)
        with open(path, "w") as f:
            f.write(content)
        file_contents.append(f"File: {name}\nContent:\n{content}\n")
    
    # Ask Claude to analyze files
    separator = "=" * 40
    message = format_message(
        "Here are the contents of several files:\n\n",
        separator,
        "\n",
        "\n".join(file_contents),
        "\n",
        separator,
        "\n\nPlease tell me which files contain error messages."
    )
    response = get_claude_response(message)
    assert "log.txt" in response
    assert "ERROR" in response

# System Tests (Real LLM with Real Files)
@pytest.mark.system
@pytest.mark.llm
def test_real_file_operations():
    """Test end-to-end file operations in project directory"""
    print("\n=== Initial Claude Response ===")
    message = format_message(
        "I need to create a new file called test_output.txt with the current timestamp and some test data. ",
        "Can you help me with that?"
    )
    response = get_claude_response(message)
    print(response)
    
    # Extract tool call from response and apply it
    tool = FileTool()
    test_file = "test_output.txt"
    tool.run({
        "operation": "write",
        "path": test_file,
        "content": "Timestamp: 2024-03-14 10:00:00\nTest data: Hello from system test!"
    })
    
    print("\n=== Verifying File Creation ===")
    assert os.path.exists(test_file)
    with open(test_file, "r") as f:
        content = f.read()
    print(f"File contents: {content}")
    
    print("\n=== Claude's Analysis ===")
    message = format_message(
        "Here is the content of test_output.txt:\n\n",
        content,
        "\n\nPlease confirm if you see both a timestamp and test data in this file."
    )
    response = get_claude_response(message)
    print(response)
    
    # Cleanup
    os.remove(test_file)

@pytest.mark.system
@pytest.mark.llm
def test_project_file_handling():
    """Test handling of actual project files"""
    print("\n=== Initial Claude Response ===")
    # Get list of Python files
    tool = FileTool()
    result = tool.run({
        "operation": "list_dir",
        "path": "tests"
    })
    
    # Convert Windows paths to forward slashes in the content
    content = result["content"].replace("\\", "/")
    
    message = format_message(
        "Here are the files in the tests directory:\n\n",
        content,
        "\n\nPlease analyze these files and tell me:\n",
        "1. Which ones are test files?\n",
        "2. What types of tests do they contain?\n",
        "3. How many test files are there in total?"
    )
    response = get_claude_response(message)
    print(response)
    
    # Verify response
    assert "test" in response.lower()
    assert ".py" in response.lower()
    assert any(str(n) in response for n in range(10, 15))  # Should find around 12 test files

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 