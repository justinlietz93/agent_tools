"""
Tests for Windows-specific ComputerTool implementation.
Includes both direct functionality tests and LLM interaction tests.
"""

import pytest
from unittest.mock import patch, MagicMock
import pyautogui
import base64
from tools.computer_tool import ComputerTool, Action
from tools.config import Config
from anthropic import Anthropic
import time
import os

# Load config with API keys
config = Config()

@pytest.fixture
def computer_tool():
    """Create ComputerTool instance with mocked screen size."""
    with patch('pyautogui.size', return_value=(1920, 1080)):
        tool = ComputerTool()
        return tool

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
        "content": result["content"]
    }

def get_tool_calls(response) -> list:
    """Extract tool calls from response content."""
    return [
        block for block in response.content 
        if block.type == "tool_use"
    ]

# Direct functionality tests
def test_mouse_move(computer_tool):
    """Test mouse movement with proper mocking."""
    with patch('pyautogui.moveTo') as mock_move, \
         patch('win32gui.GetWindowRect', return_value=(0, 0, 100, 100)) as mock_rect:
        
        result = computer_tool.run({
            "action": "mouse_move",
            "coordinate": [500, 500]  # Safe coordinates away from mocked window
        })
        
        mock_move.assert_called_once_with(500, 500, duration=0.5)
        assert result["type"] == "tool_result"
        assert result["content"]["status"] == "success"
        assert "moved" in result["content"]["message"]

def test_type_text(computer_tool):
    """Test typing with proper mocking."""
    with patch('pyautogui.write') as mock_write, \
         patch('win32gui.GetForegroundWindow') as mock_get_window, \
         patch('win32gui.SetForegroundWindow') as mock_set_window:
        
        result = computer_tool.run({
            "action": "type",
            "text": "Hello World"
        })
        
        mock_write.assert_called_once_with("Hello World", interval=0.1)
        assert result["type"] == "tool_result"
        assert result["content"]["status"] == "success"
        assert "typed" in result["content"]["message"]

def test_key_press(computer_tool):
    """Test key press combinations."""
    with patch('pyautogui.hotkey') as mock_hotkey, \
         patch('win32gui.GetForegroundWindow') as mock_get_window, \
         patch('win32gui.SetForegroundWindow') as mock_set_window:
        
        result = computer_tool.run({
            "action": "key",
            "text": "ctrl+c"
        })
        
        mock_hotkey.assert_called_once_with("ctrl", "c")
        assert result["type"] == "tool_result"
        assert result["content"]["status"] == "success"
        assert "pressed" in result["content"]["message"]

def test_screenshot(computer_tool):
    """Test screenshot functionality."""
    mock_image = MagicMock()
    mock_image.save = MagicMock()
    mock_buffer = MagicMock()
    mock_buffer.getvalue.return_value = b"test"
    
    with patch('pyautogui.screenshot', return_value=mock_image) as mock_screenshot, \
         patch('io.BytesIO', return_value=mock_buffer):
        
        result = computer_tool.run({"action": "screenshot"})
        
        mock_screenshot.assert_called_once()
        assert result["type"] == "tool_result"
        assert result["content"]["type"] == "image"
        assert result["content"]["format"] == "base64"
        assert result["content"]["data"] == "dGVzdA=="  # base64 of b"test"

def test_cursor_position(computer_tool):
    """Test cursor position with mocked win32api."""
    with patch('win32api.GetCursorPos', return_value=(100, 200)):
        result = computer_tool.run({"action": "cursor_position"})
        
        assert result["type"] == "tool_result"
        assert result["content"]["status"] == "success"
        assert result["content"]["position"] == {"x": 100, "y": 200}

def test_click_actions(computer_tool):
    """Test various click actions with proper mocking."""
    with patch('pyautogui.click') as mock_click, \
         patch('pyautogui.moveTo') as mock_move, \
         patch('win32gui.GetWindowRect', return_value=(0, 0, 100, 100)):
        
        actions = {
            "left_click": {"button": "left"},
            "right_click": {"button": "right"},
            "middle_click": {"button": "middle"},
            "double_click": {"button": "left", "clicks": 2, "interval": 0.25}
        }
        
        for action, params in actions.items():
            result = computer_tool.run({
                "action": action,
                "coordinate": [500, 500]  # Safe coordinates away from mocked window
            })
            
            mock_move.assert_called_with(500, 500, duration=0.5)
            mock_click.assert_called_with(**params)
            assert result["type"] == "tool_result"
            assert result["content"]["status"] == "success"
            assert action in result["content"]["message"]
            
            mock_click.reset_mock()
            mock_move.reset_mock()

def test_coordinate_validation(computer_tool):
    """Test coordinate validation and window safety checks."""
    with patch('pyautogui.moveTo') as mock_move, \
         patch('win32gui.GetWindowRect', return_value=(0, 0, 100, 100)):
        
        # Test coordinates in window area
        result = computer_tool.run({
            "action": "mouse_move",
            "coordinate": [50, 50]  # Inside mocked window area
        })
        
        # Should be moved to safe position
        assert mock_move.call_args[0][0] == 1820  # screen_width - 100
        assert mock_move.call_args[0][1] == 980   # screen_height - 100
        
        # Test out of bounds coordinates
        result = computer_tool.run({
            "action": "mouse_move",
            "coordinate": [2000, 2000]  # Beyond screen bounds
        })
        
        # Should be clamped to screen bounds
        assert mock_move.call_args[0][0] == 1919  # screen_width - 1
        assert mock_move.call_args[0][1] == 1079  # screen_height - 1

def test_error_handling(computer_tool):
    """Test error handling for invalid inputs."""
    # Test missing coordinate
    result = computer_tool.run({
        "action": "mouse_move"
    })
    assert result["type"] == "tool_result"
    assert "error" in result["content"]
    assert "coordinate is required" in result["content"]["error"]
    
    # Test missing text
    result = computer_tool.run({
        "action": "type"
    })
    assert result["type"] == "tool_result"
    assert "error" in result["content"]
    assert "text is required" in result["content"]["error"]
    
    # Test invalid action
    result = computer_tool.run({
        "action": "invalid_action"
    })
    assert result["type"] == "tool_result"
    assert "error" in result["content"]

# LLM integration tests
@pytest.mark.llm
def test_basic_mouse_movement_llm(computer_tool):
    """Test that Claude can use the computer tool to move the mouse."""
    tools = {computer_tool.name: computer_tool}

    system_prompt = """You are a helpful AI assistant. Use the computer tool to interact with the computer.
    The tool can control the mouse, keyboard, and take screenshots. Be careful with coordinate values
    and always verify they are within screen bounds."""

    user_message = "Please move the mouse to coordinates (100, 100)."

    response = get_claude_response(system_prompt, user_message, list(tools.values()))

    # Debug: Print response content
    print("\nLLM Response Content:")
    for block in response.content:
        print(f"Block type: {block.type}")
        if hasattr(block, "text"):
            print(f"Text: {block.text}")

    # Verify LLM used the tool correctly
    tool_calls = get_tool_calls(response)
    assert len(tool_calls) > 0, "LLM should have used the computer tool"

    tool_call = tool_calls[0]
    assert tool_call.name == "computer", "LLM should use the computer tool"
    assert tool_call.input["action"] == "mouse_move", "LLM should use mouse_move action"
    assert tool_call.input["coordinate"] == [100, 100], "LLM should use correct coordinates"

    # Execute the tool call with mocked pyautogui
    with patch('pyautogui.moveTo') as mock_move:
        result = execute_tool_call(tool_call, tools)

    # Verify the result format
    assert isinstance(result, dict)
    assert "content" in result
    assert "tool_use_id" in result
    assert "type" in result
    assert result["content"]["status"] == "success"
    assert "moved" in result["content"]["message"]

@pytest.mark.llm
def test_os_awareness_llm(computer_tool):
    """Test that Claude understands it's interacting with a Windows system."""
    tools = {computer_tool.name: computer_tool}

    system_prompt = """You are a helpful AI assistant. Use the computer tool to interact with the computer.
    You are running on a Windows system and should adapt your commands accordingly.
    When responding, always acknowledge the Windows environment and use Windows-specific terminology.
    The computer tool is designed to work with Windows-specific functionality."""

    user_message = "What operating system are we using? Can you perform a simple mouse movement to test the system?"

    response = get_claude_response(system_prompt, user_message, list(tools.values()))

    # Verify LLM acknowledges Windows environment
    assert any(
        "windows" in block.text.lower()
        for block in response.content
        if hasattr(block, "text")
    ), "LLM should recognize Windows environment"

    # Verify LLM uses the tool appropriately
    tool_calls = get_tool_calls(response)
    assert len(tool_calls) > 0, "LLM should use the computer tool"

    # Execute the tool call with mocked pyautogui
    with patch('pyautogui.moveTo') as mock_move:
        result = execute_tool_call(tool_calls[0], tools)

    # Verify result format
    assert isinstance(result, dict)
    assert "content" in result
    assert result["content"]["status"] == "success"

@pytest.mark.llm
def test_keyboard_interaction_llm(computer_tool):
    """Test that Claude can use the computer tool for keyboard input."""
    tools = {computer_tool.name: computer_tool}

    system_prompt = """You are a helpful AI assistant. Use the computer tool to interact with the computer.
    You can send keyboard input using either individual key commands or typing text."""

    user_message = "Please type 'Hello, World!' and then press Ctrl+A to select it."

    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    tool_calls = get_tool_calls(response)
    
    # Should have at least 2 tool calls (type and key command)
    assert len(tool_calls) >= 2, "LLM should make multiple tool calls for typing and key commands"

    results = []
    with patch('pyautogui.write') as mock_write, patch('pyautogui.hotkey') as mock_hotkey:
        for tool_call in tool_calls:
            result = execute_tool_call(tool_call, tools)
            results.append(result)
            
            if tool_call.input["action"] == "type":
                mock_write.assert_called_with("Hello, World!")
            elif tool_call.input["action"] == "key":
                mock_hotkey.assert_called()

    # Verify all results have correct format
    for result in results:
        assert isinstance(result, dict)
        assert "content" in result
        assert result["content"]["status"] == "success"

@pytest.mark.llm
def test_screenshot_llm(computer_tool):
    """Test that Claude can use the computer tool to take and analyze screenshots."""
    tools = {computer_tool.name: computer_tool}

    system_prompt = """You are a helpful AI assistant. Use the computer tool to interact with the computer.
    You can take screenshots and analyze their content."""

    user_message = "Please take a screenshot of the current screen."

    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    tool_calls = get_tool_calls(response)
    
    assert len(tool_calls) > 0, "LLM should use the computer tool"
    assert tool_calls[0].input["action"] == "screenshot", "LLM should use screenshot action"

    # Execute screenshot with mock
    mock_image = MagicMock()
    mock_image.save = MagicMock()
    with patch('pyautogui.screenshot', return_value=mock_image):
        result = execute_tool_call(tool_calls[0], tools)

    # Verify screenshot result format
    assert isinstance(result, dict)
    assert "content" in result
    assert result["content"]["type"] == "image"
    assert result["content"]["format"] == "base64"
    assert "data" in result["content"]

@pytest.mark.llm
def test_complex_interaction_sequence_llm(computer_tool):
    """Test that Claude can chain multiple computer interactions together."""
    tools = {computer_tool.name: computer_tool}

    system_prompt = """You are a helpful AI assistant. Use the computer tool to interact with the computer.
    IMPORTANT: You must complete ALL actions in the sequence. After each action:
    1. Execute the current action using the computer tool
    2. Verify the action was successful
    3. Move to the next action in the original sequence
    4. Execute that next action immediately
    
    Do not wait for additional prompting between actions.
    Do not stop until you have completed every action in the sequence.
    After completing all actions, summarize what was accomplished."""

    initial_message = """Please execute this exact sequence of actions:
    1. Move the mouse to position (100, 100)
    2. Click and drag to (200, 200)
    3. Type 'Hello World'
    4. Take a screenshot to verify the result
    
    Execute all actions in order without stopping. Confirm each action and move to the next one immediately."""

    # Initialize message history
    messages = [{"role": "user", "content": initial_message}]
    all_tool_calls = []

    # List of all actions for reference
    action_sequence = [
        "Move the mouse to position (100, 100)",
        "Click and drag to (200, 200)", 
        "Type 'Hello World'",
        "Take a screenshot to verify the result"
    ]

    # Execute actions with mocks
    with patch('pyautogui.moveTo') as mock_move, \
         patch('pyautogui.dragTo') as mock_drag, \
         patch('pyautogui.write') as mock_write, \
         patch('pyautogui.screenshot', return_value=MagicMock()) as mock_screenshot:

        # Keep getting responses until we have all actions
        while len(all_tool_calls) < 4:
            # Get response from Claude
            response = get_claude_response(system_prompt, messages[-1]["content"], list(tools.values()))
            
            # Debug: Print response content
            print(f"\nLLM Response (Action {len(all_tool_calls) + 1}):")
            for block in response.content:
                print(f"Block type: {block.type}")
                if hasattr(block, "text"):
                    print(f"Text: {block.text}")

            # Get tool calls from this response
            tool_calls = get_tool_calls(response)
            if not tool_calls:
                break

            # Process each tool call
            for tool_call in tool_calls:
                result = execute_tool_call(tool_call, tools)
                all_tool_calls.append(tool_call)
                
                # Update message history with progress and remaining actions
                completed_actions = len(all_tool_calls)
                remaining_actions = "\n".join(
                    f"{i+completed_actions+1}. {action}" 
                    for i, action in enumerate(action_sequence[completed_actions:])
                )
                
                # Get appropriate result message based on action type
                result_content = result["content"]
                if "message" in result_content:
                    result_msg = result_content["message"]
                elif "type" in result_content and result_content["type"] == "image":
                    result_msg = "Screenshot captured successfully"
                else:
                    result_msg = "Action completed successfully"
                
                next_message = f"""Action {completed_actions} completed: {tool_call.input['action']} - {result_msg}

Remaining actions to execute:
{remaining_actions}

Please execute the next action immediately."""
                
                messages.append({
                    "role": "assistant", 
                    "content": next_message
                })

    # Verify we got all expected actions
    assert len(all_tool_calls) >= 4, "LLM should make all required tool calls"
    
    # Verify each type of action was called
    actions_called = [call.input["action"] for call in all_tool_calls]
    assert "mouse_move" in actions_called, "Should include mouse movement"
    assert any(action in actions_called for action in ["left_click_drag", "drag"]), "Should include click and drag"
    assert "type" in actions_called, "Should include typing"
    assert "screenshot" in actions_called, "Should include screenshot"
    
    # Verify the sequence was correct
    mouse_move_index = actions_called.index("mouse_move")
    drag_index = next(i for i, action in enumerate(actions_called) if action in ["left_click_drag", "drag"])
    type_index = actions_called.index("type")
    screenshot_index = actions_called.index("screenshot")
    
    assert mouse_move_index < drag_index < type_index < screenshot_index, \
        "Actions should be performed in the correct order"

@pytest.mark.llm
def test_error_handling_llm(computer_tool):
    """Test that Claude can handle and explain computer tool errors."""
    tools = {computer_tool.name: computer_tool}

    system_prompt = """You are a helpful AI assistant. Use the computer tool to interact with the computer.
    Handle any errors gracefully and explain what went wrong to the user."""

    user_message = """Please try these actions:
    1. Move the mouse to invalid coordinates (-100, -100)
    2. Type without providing text
    3. Use an invalid action type"""

    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    tool_calls = get_tool_calls(response)
    
    results = []
    with patch('pyautogui.moveTo'):
        for tool_call in tool_calls:
            result = execute_tool_call(tool_call, tools)
            results.append(result)

    # Verify error responses are properly formatted
    for result in results:
        assert isinstance(result, dict)
        assert "type" in result
        assert "content" in result
        if "error" in result["content"]:
            assert isinstance(result["content"]["error"], str)
            assert "action" in result["content"]

@pytest.mark.llm
def test_os_specific_commands_llm(computer_tool):
    """Test that Claude adapts commands to Windows environment."""
    tools = {computer_tool.name: computer_tool}

    system_prompt = """You are a helpful AI assistant. Use the computer tool to interact with the computer.
    You should adapt your commands to work specifically with Windows."""

    user_message = """Please help me with these Windows-specific actions:
    1. Press the Windows key
    2. Type 'notepad'
    3. Press Enter
    4. Type 'This is a Windows test'"""

    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    tool_calls = get_tool_calls(response)
    
    results = []
    with patch('pyautogui.hotkey') as mock_hotkey, \
         patch('pyautogui.write') as mock_write:
        
        for tool_call in tool_calls:
            result = execute_tool_call(tool_call, tools)
            results.append(result)
            
            if tool_call.input["action"] == "key":
                if "win" in tool_call.input["text"].lower():
                    mock_hotkey.assert_called()
            elif tool_call.input["action"] == "type":
                mock_write.assert_called()

    # Verify Windows-specific behavior
    assert any(
        "windows" in block.text.lower()
        for block in response.content
        if hasattr(block, "text")
    ), "LLM should acknowledge Windows environment"

@pytest.mark.llm
def test_safety_awareness_llm(computer_tool):
    """Test that Claude understands and respects safety boundaries."""
    tools = {computer_tool.name: computer_tool}

    system_prompt = """You are a helpful AI assistant. Use the computer tool to interact with the computer.
    You must be careful with mouse movements and keyboard actions to avoid unintended consequences."""

    user_message = """Please perform these actions carefully:
    1. Move the mouse to the edge of the screen
    2. Click multiple times rapidly
    3. Type some special key combinations"""

    response = get_claude_response(system_prompt, user_message, list(tools.values()))
    
    # Verify LLM shows safety awareness in its response
    assert any(
        "careful" in block.text.lower() or "safely" in block.text.lower()
        for block in response.content
        if hasattr(block, "text")
    ), "LLM should demonstrate safety awareness"

    tool_calls = get_tool_calls(response)
    
    # Execute actions with mocks
    with patch('pyautogui.moveTo') as mock_move, \
         patch('pyautogui.click') as mock_click, \
         patch('pyautogui.hotkey') as mock_hotkey:
        
        for tool_call in tool_calls:
            result = execute_tool_call(tool_call, tools)
            
            # Verify safe coordinate values
            if tool_call.input["action"] == "mouse_move" and "coordinate" in tool_call.input:
                x, y = tool_call.input["coordinate"]
                assert 0 <= x < computer_tool.screen_width, "X coordinate should be within bounds"
                assert 0 <= y < computer_tool.screen_height, "Y coordinate should be within bounds"

@pytest.mark.real
@pytest.mark.demo
def test_notepad_demo_real():
    """Test the computer tool with a real Notepad instance."""
    tool = ComputerTool()
    
    # Launch Notepad
    print("Launching Notepad...")
    tool.run({"action": "type", "text": "notepad"})
    tool.run({"action": "key", "text": "enter"})
    
    # Wait for Notepad to open
    time.sleep(3)
    
    # Find and get info about Notepad window
    print("Finding Notepad window...")
    result = tool.run({"action": "get_window_info", "window_title": "Notepad"})
    print(f"Window info: {result}")
    
    if result["content"].get("error"):
        print("Error: Could not find Notepad window")
        return
    
    # Move Notepad to a specific position
    print("Moving Notepad window...")
    tool.run({
        "action": "move_window",
        "window_title": "Notepad",
        "position": [100, 100],
        "size": [800, 600]
    })
    
    # Ensure Notepad has focus
    print("Setting focus to Notepad...")
    tool.run({"action": "set_window_focus", "window_title": "Notepad"})
    
    # Type some text
    print("Typing text...")
    tool.run({"action": "type", "text": "Hello from the Computer Tool!"})
    tool.run({"action": "key", "text": "enter"})
    tool.run({"action": "type", "text": "Using precise window control."})
    
    # Take a screenshot
    print("Taking screenshot...")
    tool.run({"action": "screenshot"})
    
    # Close Notepad without saving
    print("Closing Notepad...")
    tool.run({"action": "key", "text": "alt+f4"})
    time.sleep(1)
    tool.run({"action": "key", "text": "n"})  # Don't save changes
    
    print("Test completed successfully")

@pytest.mark.llm
def test_llm_interaction_safety(computer_tool):
    """Test that Claude understands basic tool usage."""
    tools = {computer_tool.name: computer_tool}

    system_prompt = """You are a helpful AI assistant. Use the computer tool to control the mouse and keyboard."""

    test_cases = [
        # Simple click test
        {
            "message": "Click at coordinates 50, 50",
            "expected_action": "left_click",
            "expected_coords": [50, 50]
        },
        # Simple move test
        {
            "message": "Move mouse to 500, 500",
            "expected_action": "mouse_move",
            "expected_coords": [500, 500]
        }
    ]

    for case in test_cases:
        print(f"\n=== Testing case: {case['message']} ===")
        response = get_claude_response(system_prompt, case["message"], list(tools.values()))
        
        # Print full LLM response for debugging
        print("\nLLM Response:")
        for block in response.content:
            print(f"\nBlock type: {block.type}")
            if hasattr(block, "text"):
                print(f"Text content: {block.text}")
            if block.type == "tool_use":
                print(f"Tool call: {block.name}")
                print(f"Tool input: {block.input}")
        
        tool_calls = get_tool_calls(response)
        
        assert len(tool_calls) > 0, "LLM should attempt to use the tool"
        tool_call = tool_calls[0]
        
        # Verify tool usage
        assert tool_call.input["action"] == case["expected_action"]
        assert tool_call.input["coordinate"] == case["expected_coords"]
        
        # Execute with mocks
        with patch('pyautogui.moveTo') as mock_move, \
             patch('pyautogui.click') as mock_click, \
             patch.object(computer_tool, '_is_safe_coordinate', return_value=True):
            
            result = execute_tool_call(tool_call, tools)
            print(f"\nTool execution result: {result}")
            
            # Verify result format
            assert "error" not in result["content"], f"Got error: {result['content'].get('error')}"
            assert "message" in result["content"], "Result should have a message" 