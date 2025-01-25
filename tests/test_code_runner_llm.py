"""
Tests for CodeRunnerTool with actual LLM interactions.
Verifies that Claude can properly understand and use the code runner tool.
"""

import pytest
import os
from anthropic import Anthropic
from tools.code_runner_tool import CodeRunnerTool
from tools.config import Config
import shutil
import json

# Load config with API keys
config = Config()

@pytest.fixture
def client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return Anthropic(api_key=api_key)

@pytest.fixture
def code_runner():
    return CodeRunnerTool()

@pytest.fixture
def tool_definition():
    return {
        "name": "code_runner",
        "description": "Runs code in various languages with support for multiple files, arguments, and environment variables.",
        "input_schema": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["path", "content"]
                    }
                },
                "language": {"type": "string"},
                "main_file": {"type": "string"},
                "args": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "env": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                },
                "timeout": {"type": "number"}
            },
            "required": ["files", "language", "main_file"]
        }
    }

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
def test_basic_python_execution_llm(client, code_runner, tool_definition):
    """Test that Claude can properly use the code runner tool for basic Python execution"""
    
    # 1. Make initial request to Claude
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[{
            "role": "user",
            "content": "Please write and run a Python program that prints 'Hello from Python!'"
        }]
    )
    
    # 2. Verify Claude attempts to use the tool
    assert response.stop_reason == "tool_use"
    
    tool_use = None
    for block in response.content:
        if isinstance(block, dict) and block["type"] == "tool_use":
            tool_use = block
            break
    
    assert tool_use is not None
    assert tool_use["name"] == "code_runner"
    
    # 3. Run the tool with Claude's input
    tool_result = code_runner.run(tool_use["input"])
    
    # 4. Send result back to Claude
    followup = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": tool_result["content"]
                }]
            }
        ]
    )
    
    # 5. Verify Claude properly handled the result
    assert "Hello from Python!" in tool_result["content"]
    assert any("successfully" in block["text"] for block in followup.content if block["type"] == "text")

@pytest.mark.llm
def test_python_with_args_llm(client, code_runner, tool_definition):
    """Test that Claude can use the code runner tool with command line arguments"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[{
            "role": "user", 
            "content": "Write and run a Python program that prints out its command line arguments. Pass it the arguments 'first' and 'second'."
        }]
    )
    
    assert response.stop_reason == "tool_use"
    
    tool_use = next((block for block in response.content if isinstance(block, dict) and block["type"] == "tool_use"), None)
    assert tool_use is not None
    assert tool_use["name"] == "code_runner"
    assert "args" in tool_use["input"]
    assert tool_use["input"]["args"] == ["first", "second"]
    
    tool_result = code_runner.run(tool_use["input"])
    
    followup = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": tool_result["content"]
                }]
            }
        ]
    )
    
    assert "first" in tool_result["content"]
    assert "second" in tool_result["content"]
    assert any("arguments" in block["text"].lower() for block in followup.content if block["type"] == "text")

@pytest.mark.llm
def test_python_with_dependencies_llm(client, code_runner, tool_definition):
    """Test that Claude can create and run a project with dependencies"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[{
            "role": "user",
            "content": "Create and run a Python script that uses the 'requests' library to get the status of github.com"
        }]
    )
    
    assert response.stop_reason == "tool_use"
    
    tool_use = next((block for block in response.content if isinstance(block, dict) and block["type"] == "tool_use"), None)
    assert tool_use is not None
    assert tool_use["name"] == "code_runner"
    
    tool_result = code_runner.run(tool_use["input"])
    
    followup = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": tool_result["content"]
                }]
            }
        ]
    )
    
    assert "status" in tool_result["content"].lower()
    assert any("github" in block["text"].lower() for block in followup.content if block["type"] == "text")

@pytest.mark.llm
def test_typescript_project_llm(client, code_runner, tool_definition):
    """Test that Claude can create and run a TypeScript project"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[{
            "role": "user",
            "content": "Create and run a TypeScript program that demonstrates basic type usage"
        }]
    )
    
    assert response.stop_reason == "tool_use"
    
    tool_use = next((block for block in response.content if isinstance(block, dict) and block["type"] == "tool_use"), None)
    assert tool_use is not None
    assert tool_use["name"] == "code_runner"
    
    tool_result = code_runner.run(tool_use["input"])
    
    followup = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": tool_result["content"]
                }]
            }
        ]
    )
    
    assert len(tool_result["content"]) > 0
    assert any("type" in block["text"].lower() for block in followup.content if block["type"] == "text")

@pytest.mark.llm
def test_error_handling_llm(client, code_runner, tool_definition):
    """Test that Claude properly handles execution errors"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[{
            "role": "user",
            "content": "Run this invalid Python code: 'print(undefined_variable)'"
        }]
    )
    
    assert response.stop_reason == "tool_use"
    
    tool_use = next((block for block in response.content if isinstance(block, dict) and block["type"] == "tool_use"), None)
    assert tool_use is not None
    assert tool_use["name"] == "code_runner"
    
    tool_result = code_runner.run(tool_use["input"])
    
    followup = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": tool_result["content"]
                }]
            }
        ]
    )
    
    assert "Error:" in tool_result["content"]
    assert any("error" in block["text"].lower() for block in followup.content if block["type"] == "text")

@pytest.mark.llm
def test_timeout_handling_llm(client, code_runner, tool_definition):
    """Test that Claude can handle timeouts properly"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[{
            "role": "user",
            "content": "Create a Python script that runs forever and should timeout after 2 seconds"
        }]
    )
    
    assert response.stop_reason == "tool_use"
    
    tool_use = next((block for block in response.content if isinstance(block, dict) and block["type"] == "tool_use"), None)
    assert tool_use is not None
    assert tool_use["name"] == "code_runner"
    
    tool_result = code_runner.run(tool_use["input"])
    
    followup = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": tool_result["content"]
                }]
            }
        ]
    )
    
    assert any(phrase in tool_result["content"].lower() for phrase in ["timed out", "timeout"])
    assert any("timeout" in block["text"].lower() for block in followup.content if block["type"] == "text")

@pytest.mark.llm
@pytest.mark.skipif(shutil.which('go') is None, reason="Go not installed")
def test_go_project_llm(client, code_runner, tool_definition):
    """Test that Claude can create and run a Go project"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[{
            "role": "user",
            "content": """Create and run a Go program that:
            1. Creates a simple struct type
            2. Has a method on that struct
            3. Uses the method in main()"""
        }]
    )
    
    assert response.stop_reason == "tool_use"
    
    tool_use = next((block for block in response.content if isinstance(block, dict) and block["type"] == "tool_use"), None)
    assert tool_use is not None
    assert tool_use["name"] == "code_runner"
    
    tool_result = code_runner.run(tool_use["input"])
    
    followup = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": tool_result["content"]
                }]
            }
        ]
    )
    
    assert len(tool_result["content"]) > 0
    assert any("struct" in block["text"].lower() for block in followup.content if block["type"] == "text")

@pytest.mark.llm
@pytest.mark.skipif(shutil.which('cargo') is None, reason="Rust not installed")
def test_rust_project_llm(client, code_runner, tool_definition):
    """Test that Claude can create and run a Rust project"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[{
            "role": "user",
            "content": """Create and run a Rust program that:
            1. Defines a custom struct
            2. Implements a trait for it
            3. Uses Result for error handling"""
        }]
    )
    
    assert response.stop_reason == "tool_use"
    
    tool_use = next((block for block in response.content if isinstance(block, dict) and block["type"] == "tool_use"), None)
    assert tool_use is not None
    assert tool_use["name"] == "code_runner"
    
    tool_result = code_runner.run(tool_use["input"])
    
    followup = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": tool_result["content"]
                }]
            }
        ]
    )
    
    assert len(tool_result["content"]) > 0
    assert any("trait" in block["text"].lower() for block in followup.content if block["type"] == "text")

@pytest.mark.llm
def test_python_with_build_args_llm(client, code_runner, tool_definition):
    """Test that Claude can use build arguments properly"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[{
            "role": "user",
            "content": """Create a Python script that:
            1. Accepts a --config argument
            2. Prints different output for dev vs prod
            Run it with --config=prod"""
        }]
    )
    
    assert response.stop_reason == "tool_use"
    
    tool_use = next((block for block in response.content if isinstance(block, dict) and block["type"] == "tool_use"), None)
    assert tool_use is not None
    assert tool_use["name"] == "code_runner"
    
    tool_result = code_runner.run(tool_use["input"])
    
    followup = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": tool_result["content"]
                }]
            }
        ]
    )
    
    assert "prod" in tool_result["content"].lower()
    assert any("production" in block["text"].lower() for block in followup.content if block["type"] == "text")

@pytest.mark.llm
def test_python_with_env_vars_llm(client, code_runner, tool_definition):
    """Test that Claude can use environment variables properly"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[{
            "role": "user",
            "content": """Create a Python script that:
            1. Reads ENV and API_KEY from environment variables
            2. Uses 'dev' and 'none' as default values
            3. Prints both values
            Set ENV=prod and API_KEY=test123"""
        }]
    )
    
    assert response.stop_reason == "tool_use"
    
    tool_use = next((block for block in response.content if isinstance(block, dict) and block["type"] == "tool_use"), None)
    assert tool_use is not None
    assert tool_use["name"] == "code_runner"
    
    tool_result = code_runner.run(tool_use["input"])
    
    followup = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": tool_result["content"]
                }]
            }
        ]
    )
    
    assert all(val in tool_result["content"].lower() for val in ["prod", "test123"])
    assert any("environment" in block["text"].lower() for block in followup.content if block["type"] == "text")

@pytest.mark.llm
def test_multi_file_project_llm(client, code_runner, tool_definition):
    """Test that Claude can handle multi-file projects"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[{
            "role": "user",
            "content": """Create and run a Python project with:
            1. A main.py that imports from utils.py
            2. A utils.py with helper functions
            3. A config.py with settings
            Make them work together to print a formatted message."""
        }]
    )
    
    assert response.stop_reason == "tool_use"
    
    tool_use = next((block for block in response.content if isinstance(block, dict) and block["type"] == "tool_use"), None)
    assert tool_use is not None
    assert tool_use["name"] == "code_runner"
    
    tool_result = code_runner.run(tool_use["input"])
    
    followup = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[tool_definition],
        messages=[
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": tool_result["content"]
                }]
            }
        ]
    )
    
    assert len(tool_result["content"]) > 0
    assert any("files" in block["text"].lower() for block in followup.content if block["type"] == "text")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 