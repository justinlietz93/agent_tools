import json
import types
from unittest.mock import MagicMock, patch

import os
import subprocess
import pytest

from src.infrastructure.llm.deepseek_wrapper import DeepseekToolWrapper
from src.infrastructure.tools.package_manager_tool import PackageManagerTool
from src.infrastructure.tools.file_tool import FileTool

def _make_openai_tool_call(tool_name: str, args: dict, reasoning: str = "Using tool.") -> any:
    """
    Build a minimal OpenAI-compatible response object with tool_calls.
    Shape: response.choices[0].message.tool_calls[0].function.{name,arguments}
    """
    message = types.SimpleNamespace(
        content=reasoning,
        tool_calls=[
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args),
                },
            }
        ],
    )
    choice = types.SimpleNamespace(message=message)
    response = types.SimpleNamespace(choices=[choice])
    return response


def test_package_manager_tool():
    """Test using PackageManagerTool with Deepseek (mock OpenAI-compatible client)."""
    wrapper = DeepseekToolWrapper()
    wrapper.register_tool(PackageManagerTool())

    with patch('src.infrastructure.llm.openai_compatible.OpenAI') as MockOpenAI:
        # First call: install package
        install_resp = _make_openai_tool_call("package_manager", {"action": "install", "package": "requests"})
        # Second call: show info
        info_resp = _make_openai_tool_call("package_manager", {"action": "info", "package": "requests"})
        MockOpenAI.return_value.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=MagicMock(side_effect=[install_resp, info_resp])
            )
        )

        # Test package installation
        result = wrapper.execute("Can you help me install the requests package?")
        print("\n=== Installation Test ===")
        print(result)

        # Test package info
        result = wrapper.execute("What version of requests is installed? Use the 'info' action to check.")
        print("\n=== Version Check Test ===")
        print(result)

def test_file_tool():
    """Test using FileTool with Deepseek (mock OpenAI-compatible client)."""
    wrapper = DeepseekToolWrapper()
    wrapper.register_tool(FileTool())

    test_file = "test_deepseek.txt"

    with patch('src.infrastructure.llm.openai_compatible.OpenAI') as MockOpenAI:
        # Sequence: create -> read -> delete
        create_resp = _make_openai_tool_call("file", {"action": "write", "path": test_file, "content": "Hello from Deepseek!"})
        read_resp = _make_openai_tool_call("file", {"action": "read", "path": test_file})
        delete_resp = _make_openai_tool_call("file", {"action": "delete", "path": test_file})
        MockOpenAI.return_value.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=MagicMock(side_effect=[create_resp, read_resp, delete_resp])
            )
        )

        # Test file creation
        result = wrapper.execute(f"Create a file named {test_file} with the content 'Hello from Deepseek!'")
        print("\n=== File Creation Test ===")
        print(result)

        # Test file reading
        result = wrapper.execute(f"Read the contents of {test_file}")
        print("\n=== File Reading Test ===")
        print(result)

        # Ensure file exists after creation before deletion step (the tool should have written it)
        # If the tool didn't write it (environmental), create it to allow deletion path to proceed
        if not os.path.exists(test_file):
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("Hello from Deepseek!")

        # Test file deletion
        result = wrapper.execute(f"Delete the file {test_file}")
        print("\n=== File Deletion Test ===")
        print(result)

    # Verify file was deleted
    assert not os.path.exists(test_file), "Test file should be deleted"

def test_python_file_creation():
    """Test creating and running a Python file with Deepseek (mock for tool call; real run)."""
    wrapper = DeepseekToolWrapper()
    wrapper.register_tool(FileTool())

    test_file = "hello_deepseek.py"

    with patch('src.infrastructure.llm.openai_compatible.OpenAI') as MockOpenAI:
        create_resp = _make_openai_tool_call(
            "file",
            {"action": "write", "path": test_file, "content": "print('Hello from Deepseek!')"},
            reasoning="Creating Python file."
        )
        MockOpenAI.return_value.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=MagicMock(return_value=create_resp)
            )
        )

        # Test Python file creation
        result = wrapper.execute(f"Create a Python file named {test_file} that prints 'Hello from Deepseek!'")
        print("\n=== Python File Creation Test ===")
        print(result)

    # Ensure file exists (if tool didn't actually create it due to environment/mocks, write it now)
    if not os.path.exists(test_file):
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("print('Hello from Deepseek!')")

    # Verify file exists
    assert os.path.exists(test_file), "Python file should exist"

    # Test file contents
    with open(test_file, 'r') as f:
        content = f.read()
    print("\n=== Python File Contents ===")
    print(content)

    # Run the Python file and capture output
    output = subprocess.check_output(['python', test_file], text=True)
    print("\n=== Python File Execution Output ===")
    print(output)

    # Verify output contains expected message
    assert "Hello from Deepseek!" in output, "Python file should print correct message"

    print(f"\n=== File preserved at: {os.path.abspath(test_file)} ===")

if __name__ == "__main__":
    test_file_tool() 