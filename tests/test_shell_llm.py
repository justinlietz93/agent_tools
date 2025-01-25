"""
Tests for ShellTool with actual LLM interactions.
Verifies that Claude can properly understand and use the shell tool.
"""

import pytest
from tools.shell_tool import ShellTool
from tools.tool_base import ToolResult
from tools.config import Config
from anthropic import Anthropic
import subprocess
import os
from typing import Dict, Any

# Load config with API keys
config = Config()

class ConcreteShellTool(ShellTool):
    """Concrete implementation of ShellTool for testing"""
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return self.input_schema

@pytest.fixture
def shell_tool():
    """Create ShellTool instance"""
    return ConcreteShellTool()

@pytest.fixture
def restricted_shell_tool():
    """Create ShellTool instance with command whitelist"""
    return ConcreteShellTool(allowed_commands=["echo", "pwd", "ls"])

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
    
    # Don't pass tool_call_id to the run method
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
def test_shell_command_llm(shell_tool):
    """Test that Claude can use the shell tool to execute commands and understand their output."""
    tools = {shell_tool.name: shell_tool}
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # System prompt that explains the tool's purpose
    system_prompt = """You are a helpful AI assistant. Use the shell tool to execute commands.
    The tool can run shell commands and return their output. Be careful with command syntax
    and always check for errors in the output."""

    # Ask Claude to run a simple command
    user_message = "Please use the shell tool to print 'Hello from Claude' using the echo command."

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
    assert len(tool_calls) > 0, "LLM should have used the shell tool"

    tool_call = tool_calls[0]
    assert tool_call.name == "shell", "LLM should use the shell tool"
    assert "echo" in tool_call.input.get("command", ""), "LLM should use echo command"

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

    # Verify we got the expected output
    content = result["content"]["content"] if isinstance(result["content"], dict) else result["content"]
    assert "Hello from Claude" in content, "Should find the echoed text"

    # Feed the result back to the LLM
    print("\nSending messages to LLM:")
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response.content[0].text},
        {"role": "user", "content": f"The command output was: {content}. Please acknowledge this output."}
    ]
    print("Messages:", messages)
    
    final_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
        temperature=0
    )

    # Debug: Print final response
    print("\nFinal LLM Response:")
    print("Response content:", final_response.content)
    
    # Verify LLM understands the output - handle empty responses
    assert len(final_response.content) > 0, "LLM should provide a response"
    final_text = final_response.content[0].text if hasattr(final_response.content[0], "text") else str(final_response.content[0])
    print("Final text:", final_text)

    # Verify the response mentions the command output
    assert any(
        (hasattr(block, "text") and "Hello from Claude" in block.text) or
        (not hasattr(block, "text") and "Hello from Claude" in str(block))
        for block in final_response.content
    ), "LLM should acknowledge the command output"

@pytest.mark.llm
def test_shell_command_error_llm(shell_tool):
    """Test that Claude can handle shell command errors appropriately."""
    tools = {shell_tool.name: shell_tool}
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    system_prompt = """You are a helpful AI assistant. Use the shell tool to execute commands.
    When a command fails, you should understand the error and explain what went wrong."""

    # Ask Claude to run an invalid command
    user_message = "Please run a command that doesn't exist to see how errors are handled."

    response = get_claude_response(system_prompt, user_message, list(tools.values()))

    # Debug: Print response content
    print("\nLLM Response Content:")
    for block in response.content:
        print(f"Block type: {block.type}")
        if hasattr(block, "text"):
            print(f"Text: {block.text}")

    tool_calls = get_tool_calls(response)
    assert len(tool_calls) > 0, "LLM should attempt to use the shell tool"

    result = execute_tool_call(tool_calls[0], tools)

    # Debug: Print result
    print("\nTool Result:")
    print(result)

    content = result["content"]["content"] if isinstance(result["content"], dict) else result["content"]
    assert content.startswith("Error:"), "Should get an error message"

    # Have Claude explain the error
    final_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response.content[0].text},
            {"role": "assistant", "content": f"The command resulted in this error: {content}"}
        ],
        temperature=0
    )

    # Debug: Print final response
    print("\nFinal LLM Response:")
    print(final_response.content[0].text)

    # Verify LLM understands and explains the error
    assert "error" in final_response.content[0].text.lower(), "LLM should acknowledge the error"
    assert "command" in final_response.content[0].text.lower(), "LLM should mention the command"

def test_basic_command_execution_llm(shell_tool):
    """Test basic command execution"""
    result = shell_tool.run({"command": "echo 'Hello World'"})
    assert isinstance(result, dict)
    assert "content" in result
    assert "type" in result
    assert "Hello World" in result["content"]

def test_command_whitelist_llm(restricted_shell_tool):
    """Test command whitelist enforcement"""
    # Allowed command
    result = restricted_shell_tool.run({"command": "echo 'test'"})
    assert isinstance(result, dict)
    assert "content" in result
    assert "test" in result["content"]
    
    # Disallowed command
    result = restricted_shell_tool.run({"command": "rm -rf /"})
    assert isinstance(result, dict)
    assert result["content"].startswith("Error:")
    assert "not in the allowed list" in result["content"]

def test_timeout_handling_llm(shell_tool):
    """Test command timeout handling"""
    # Command that should timeout
    timeout_cmd = "timeout 10" if os.name != "nt" else "ping -n 10 127.0.0.1"
    result = shell_tool.run({"command": timeout_cmd, "timeout": 1})
    assert isinstance(result, dict)
    assert result["content"].startswith("Error:")
    assert any(phrase in result["content"].lower() for phrase in ["timeout", "timed out"])

def test_working_directory_llm(shell_tool, tmp_path):
    """Test working directory handling"""
    # Create a test file in a temporary directory
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    
    # Run command in temporary directory
    result = shell_tool.run({
        "command": "type test.txt" if os.name == "nt" else "cat test.txt",
        "working_dir": str(tmp_path)
    })
    assert isinstance(result, dict)
    assert "test content" in result["content"]

def test_error_handling_llm(shell_tool):
    """Test error handling for invalid commands"""
    result = shell_tool.run({"command": "nonexistentcommand"})
    assert isinstance(result, dict)
    assert result["content"].startswith("Error:")
    assert any(phrase in result["content"].lower() for phrase in ["not found", "not recognized"])

def test_command_validation_llm(shell_tool):
    """Test command validation"""
    # Empty command
    result = shell_tool.run({"command": ""})
    assert isinstance(result, dict)
    assert result["content"].startswith("Error:")
    assert "required" in result["content"].lower()
    
    # None command
    result = shell_tool.run({"command": None})
    assert isinstance(result, dict)
    assert result["content"].startswith("Error:")
    assert "required" in result["content"].lower()

def test_output_capture_llm(shell_tool):
    """Test capturing both stdout and stderr"""
    # Command with stdout
    result = shell_tool.run({"command": "echo 'stdout test'"})
    assert isinstance(result, dict)
    assert "stdout test" in result["content"]
    
    # Command with stderr - using a command that fails and writes to stderr
    if os.name == "nt":
        stderr_cmd = "python -c \"import sys; sys.stderr.write('error message'); sys.exit(1)\""
    else:
        stderr_cmd = "python3 -c \"import sys; sys.stderr.write('error message'); sys.exit(1)\""
        
    result = shell_tool.run({"command": stderr_cmd})
    assert isinstance(result, dict)
    assert result["content"].startswith("Error:")
    assert "error message" in result["content"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 