"""Tests for the FileTool."""

import os
import shutil
import pytest
import anthropic
from tools.file_tool import FileTool
from tools.tool_base import Tool

@pytest.fixture
def test_dir():
    """Create and clean up a test directory."""
    test_dir = "test_files"
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    shutil.rmtree(test_dir)

@pytest.fixture
def file_tool():
    """Create a FileTool instance."""
    return FileTool()

@pytest.fixture
def claude_client():
    """Create an Anthropic client."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return anthropic.Client(api_key=api_key)

def get_claude_response(client, system_prompt, user_message, tools):
    """Get a response from Claude with tool definitions."""
    tool_configs = []
    for tool in tools.values():
        tool_config = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema
        }
        tool_configs.append(tool_config)
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        tools=tool_configs
    )
    return response

def get_tool_calls_from_response(response):
    """Extract tool calls from Claude's response."""
    tool_calls = []
    for content in response.content:
        if content.type == "tool_use":
            tool_calls.append(content)
    return tool_calls

def execute_tool_call(tool_call, tools: dict[str, Tool]) -> dict:
    """Execute a tool call from Claude."""
    tool = tools.get(tool_call.name)
    if not tool:
        raise ValueError(f"Tool {tool_call.name} not found")
    
    result = tool.run(tool_call.input)
    return {
        "type": "tool_response",
        "tool_use_id": tool_call.id,
        "content": result["content"]
    }

def test_claude_file_operations(test_dir, file_tool, claude_client):
    """Test Claude using the file tool for operations."""
    tools = {"file": file_tool}
    test_file = os.path.join(test_dir, "claude_test.txt")
    
    # Have Claude create and write to a file
    system_prompt = (
        "You are a helpful AI that uses tools to manage files. "
        "The file tool supports these operations:\n"
        "- write: Create or overwrite a file (requires path and content)\n"
        "- read: Read entire file content (requires path)\n"
        "- read_lines: Read specific lines (requires path, start_line, and end_line)\n"
        "- edit_lines: Edit specific lines (requires path, start_line, end_line, and content)\n\n"
        "Always use the exact operation name and provide all required input_schema.\n"
        "When given multiple steps, you must complete ALL steps in sequence.\n\n"
        "EXAMPLE OF REQUIRED SEQUENCE:\n"
        "When asked to modify lines 2-3 in test.txt, you MUST make these TWO tool calls:\n\n"
        "1. FIRST call read_lines:\n"
        "   {'operation': 'read_lines', 'path': 'test.txt', 'start_line': 2, 'end_line': 3}\n\n"
        "2. THEN call edit_lines:\n"
        "   {'operation': 'edit_lines', 'path': 'test.txt', 'start_line': 2, 'end_line': 3, 'content': 'new content'}\n\n"
        "IMPORTANT: Both tool calls are REQUIRED. Making only the read_lines call is not enough.\n"
        "You must ALWAYS make BOTH calls in this exact sequence when modifying lines."
    )
    user_message = f"Please create a file at {test_file} with some test content and then read it back."
    
    response = get_claude_response(claude_client, system_prompt, user_message, tools)
    
    # Execute Claude's tool calls
    tool_calls = get_tool_calls_from_response(response)
    print("\nFirst tool calls:")
    for tool_call in tool_calls:
        print(f"Tool call: {tool_call}")
        result = execute_tool_call(tool_call, tools)
        print(f"Result: {result}")
        assert result["type"] == "tool_response"
        assert "Successfully" in result["content"] or len(result["content"]) > 0
    
    assert os.path.exists(test_file)
    
    # Have Claude modify specific lines
    user_message = (
        f"You MUST make BOTH of these tool calls in sequence:\n\n"
        f"1. First make this EXACT tool call to read the lines:\n"
        f"  {{'operation': 'read_lines', 'path': '{test_file}', 'start_line': 2, 'end_line': 3}}\n\n"
        f"After I show you the result, then make this EXACT tool call to edit the lines:\n"
        f"  {{'operation': 'edit_lines', 'path': '{test_file}', 'start_line': 2, 'end_line': 3, 'content': 'Modified by Claude\\nModified by Claude'}}\n\n"
        f"IMPORTANT: You must make BOTH tool calls in this exact order. Do not skip either call."
    )

    # First get Claude to read the lines
    response = get_claude_response(claude_client, system_prompt, user_message, tools)
    
    print("\nFirst response:")
    print(response.content)
    
    tool_calls = get_tool_calls_from_response(response)
    assert len(tool_calls) == 1, "Claude should only make the read_lines call first"
    assert tool_calls[0].input["operation"] == "read_lines"
    
    result = execute_tool_call(tool_calls[0], tools)
    print(f"\nRead result: {result}")
    
    # Now have Claude edit the lines
    user_message = (
        f"Now that you have read the lines, please make the edit_lines call to modify them:\n\n"
        f"  {{'operation': 'edit_lines', 'path': '{test_file}', 'start_line': 2, 'end_line': 3, 'content': 'Modified by Claude\\nModified by Claude'}}"
    )
    
    response = get_claude_response(claude_client, system_prompt, user_message, tools)
    
    print("\nSecond response:")
    print(response.content)
    
    tool_calls = get_tool_calls_from_response(response)
    assert len(tool_calls) == 1, "Claude should only make the edit_lines call"
    assert tool_calls[0].input["operation"] == "edit_lines"
    
    result = execute_tool_call(tool_calls[0], tools)
    print(f"\nEdit result: {result}")
    
    # Verify the file was modified
    with open(test_file, "r") as f:
        content = f.read()
        assert "Modified by Claude" in content, f"File content should contain 'Modified by Claude' but was: {content}"

def test_claude_directory_operations(test_dir, file_tool, claude_client):
    """Test Claude using the file tool for directory operations."""
    tools = {"file": file_tool}
    test_dir_name = os.path.join(test_dir, "claude_dir")
    
    # Have Claude create a directory structure and some files
    system_prompt = "You are a helpful AI that uses tools to manage files and directories."
    user_message = f"Please create a directory at {test_dir_name}, add some nested directories and files in it, then show me the directory structure."
    
    response = get_claude_response(claude_client, system_prompt, user_message, tools)
    
    tool_calls = get_tool_calls_from_response(response)
    for tool_call in tool_calls:
        result = execute_tool_call(tool_call, tools)
        assert result["type"] == "tool_response"
    
    assert os.path.exists(test_dir_name)
    assert os.path.isdir(test_dir_name)
    
    # Have Claude move and copy files
    user_message = f"Please copy one of the files you created to a new location and then move another one."
    
    response = get_claude_response(claude_client, system_prompt, user_message, tools)
    
    tool_calls = get_tool_calls_from_response(response)
    for tool_call in tool_calls:
        result = execute_tool_call(tool_call, tools)
        assert result["type"] == "tool_response"
        assert "Successfully" in result["content"]

def test_claude_error_handling(test_dir, file_tool, claude_client):
    """Test how Claude handles file operation errors."""
    tools = {"file": file_tool}
    non_existent = os.path.join(test_dir, "does_not_exist.txt")
    
    # Have Claude try to read a non-existent file
    system_prompt = "You are a helpful AI that uses tools to manage files."
    user_message = f"Please try to read the file at {non_existent} and handle any errors appropriately."
    
    response = get_claude_response(claude_client, system_prompt, user_message, tools)
    
    tool_calls = get_tool_calls_from_response(response)
    for tool_call in tool_calls:
        result = execute_tool_call(tool_call, tools)
        assert result["type"] == "tool_response"
        assert "Error" in result["content"]
    
    # Have Claude handle missing input_schema
    user_message = "Please try to move a file without specifying the destination."
    
    response = get_claude_response(claude_client, system_prompt, user_message, tools)
    
    tool_calls = get_tool_calls_from_response(response)
    for tool_call in tool_calls:
        result = execute_tool_call(tool_call, tools)
        assert result["type"] == "tool_response"
        assert "Error" in result["content"]

def test_basic_operations(test_dir, file_tool):
    """Test basic read/write operations."""
    test_file = os.path.join(test_dir, "test.txt")
    
    # Test write
    result = file_tool.run({
        "operation": "write",
        "path": test_file,
        "content": "Hello\nWorld\n"
    })
    assert result["type"] == "tool_response"
    assert "Successfully wrote" in result["content"]
    assert os.path.exists(test_file)
    
    # Test read
    result = file_tool.run({
        "operation": "read",
        "path": test_file
    })
    assert result["type"] == "tool_response"
    assert result["content"] == "Hello\nWorld\n"
    
    # Test append
    result = file_tool.run({
        "operation": "append",
        "path": test_file,
        "content": "More content"
    })
    assert "Successfully appended" in result["content"]
    
    # Verify content
    result = file_tool.run({
        "operation": "read",
        "path": test_file
    })
    assert result["content"] == "Hello\nWorld\nMore content"

def test_line_operations(test_dir, file_tool):
    """Test line-level operations."""
    test_file = os.path.join(test_dir, "lines.txt")
    
    # Create test file
    content = "\n".join(f"Line {i}" for i in range(1, 6))
    file_tool.run({
        "operation": "write",
        "path": test_file,
        "content": content
    })
    
    # Test read_lines
    result = file_tool.run({
        "operation": "read_lines",
        "path": test_file,
        "start_line": 2,
        "end_line": 4
    })
    assert "2: Line 2" in result["content"]
    assert "3: Line 3" in result["content"]
    assert "4: Line 4" in result["content"]
    
    # Test edit_lines
    result = file_tool.run({
        "operation": "edit_lines",
        "path": test_file,
        "start_line": 3,
        "end_line": 4,
        "content": "New Line 3\nNew Line 4"
    })
    assert "Successfully edited" in result["content"]
    
    # Verify changes
    result = file_tool.run({
        "operation": "read",
        "path": test_file
    })
    assert "New Line 3" in result["content"]
    assert "New Line 4" in result["content"]
    assert "Line 5" in result["content"]

def test_directory_operations(test_dir, file_tool):
    """Test directory operations."""
    subdir = os.path.join(test_dir, "subdir")
    nested = os.path.join(subdir, "nested")
    
    # Test mkdir
    result = file_tool.run({
        "operation": "mkdir",
        "path": subdir
    })
    assert "Successfully created" in result["content"]
    assert os.path.isdir(subdir)
    
    # Test recursive mkdir
    result = file_tool.run({
        "operation": "mkdir",
        "path": nested,
        "recursive": True
    })
    assert "Successfully created" in result["content"]
    assert os.path.isdir(nested)
    
    # Create some files
    for i in range(3):
        file_tool.run({
            "operation": "write",
            "path": os.path.join(subdir, f"file{i}.txt"),
            "content": f"Content {i}"
        })
    
    # Test list_dir
    result = file_tool.run({
        "operation": "list_dir",
        "path": test_dir
    })
    assert "DIR  subdir" in result["content"]
    
    # Test recursive list
    result = file_tool.run({
        "operation": "list_dir",
        "path": test_dir,
        "recursive": True
    })
    assert "file0.txt" in result["content"]
    assert "file1.txt" in result["content"]
    assert "file2.txt" in result["content"]
    assert "nested" in result["content"]

def test_copy_move_operations(test_dir, file_tool):
    """Test copy and move operations."""
    src_file = os.path.join(test_dir, "source.txt")
    dest_file = os.path.join(test_dir, "dest.txt")
    src_dir = os.path.join(test_dir, "src_dir")
    dest_dir = os.path.join(test_dir, "dest_dir")
    
    # Create test file and directory
    file_tool.run({
        "operation": "write",
        "path": src_file,
        "content": "Test content"
    })
    file_tool.run({
        "operation": "mkdir",
        "path": src_dir
    })
    file_tool.run({
        "operation": "write",
        "path": os.path.join(src_dir, "nested.txt"),
        "content": "Nested content"
    })
    
    # Test file copy
    result = file_tool.run({
        "operation": "copy",
        "path": src_file,
        "dest": dest_file
    })
    assert "Successfully copied" in result["content"]
    assert os.path.exists(dest_file)
    
    # Test directory copy
    result = file_tool.run({
        "operation": "copy",
        "path": src_dir,
        "dest": dest_dir,
        "recursive": True
    })
    assert "Successfully copied" in result["content"]
    assert os.path.exists(os.path.join(dest_dir, "nested.txt"))
    
    # Test move
    moved_file = os.path.join(test_dir, "moved.txt")
    result = file_tool.run({
        "operation": "move",
        "path": src_file,
        "dest": moved_file
    })
    assert "Successfully moved" in result["content"]
    assert not os.path.exists(src_file)
    assert os.path.exists(moved_file)

def test_delete_operations(test_dir, file_tool):
    """Test delete operations."""
    test_file = os.path.join(test_dir, "to_delete.txt")
    test_dir_del = os.path.join(test_dir, "dir_to_delete")
    nested = os.path.join(test_dir_del, "nested")
    
    # Create test files and directories
    file_tool.run({
        "operation": "write",
        "path": test_file,
        "content": "Delete me"
    })
    file_tool.run({
        "operation": "mkdir",
        "path": nested,
        "recursive": True
    })
    file_tool.run({
        "operation": "write",
        "path": os.path.join(nested, "file.txt"),
        "content": "Nested file"
    })
    
    # Test file deletion
    result = file_tool.run({
        "operation": "delete",
        "path": test_file
    })
    assert "Successfully deleted" in result["content"]
    assert not os.path.exists(test_file)
    
    # Test recursive directory deletion
    result = file_tool.run({
        "operation": "delete",
        "path": test_dir_del,
        "recursive": True
    })
    assert "Successfully deleted" in result["content"]
    assert not os.path.exists(test_dir_del) 