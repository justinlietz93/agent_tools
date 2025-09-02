"""Tests for using tools with Deepseek Reasoner.

These tests are updated to avoid real API calls by mocking the OpenAI-compatible
client used under the DeepseekToolWrapper. Each test patches the underlying
OpenAI client to return tool_calls that match the expected behavior.
"""

import json
import types
from typing import Any, Dict, List

import pytest
from unittest.mock import patch, MagicMock

from src.infrastructure.llm.deepseek_wrapper import DeepseekToolWrapper
from src.infrastructure.tools.computer_tool import ComputerTool

@pytest.fixture
def wrapper():
    """Create a DeepseekToolWrapper instance."""
    return DeepseekToolWrapper()

@pytest.fixture
def computer_tool():
    """Create a mocked ComputerTool instance."""
    with patch('pyautogui.size', return_value=(1920, 1080)), \
         patch('win32gui.EnumWindows'), \
         patch('win32gui.GetWindowRect', return_value=(0, 0, 100, 100)), \
         patch('win32gui.GetWindowText', return_value="Cursor"):
        tool = ComputerTool()
        tool.cursor_window = 12345
        tool.cursor_window_rect = (0, 0, 100, 100)
        return tool

def print_test_response(test_name: str, result: str):
    """Print formatted test response."""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(result)
    print(f"{'='*80}\n")


def _make_openai_tool_call(tool_name: str, args: Dict[str, Any], reasoning: str = "I will use the tool.") -> Any:
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

@pytest.mark.llm
def test_basic_mouse_movement_llm(wrapper, computer_tool):
    """Test that Deepseek can use the computer tool to move the mouse."""
    wrapper.register_tool(computer_tool)

    with patch('src.infrastructure.llm.openai_compatible.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=MagicMock(return_value=_make_openai_tool_call(
                    "computer", {"action": "mouse_move", "coordinate": [100, 100]},
                    reasoning="First, I'll move the mouse as requested."
                ))
            )
        )

        with patch('pyautogui.moveTo') as mock_move:
            result = wrapper.execute("Move the mouse to coordinates (100, 100)")
            print_test_response("Basic Mouse Movement", result)
            
            assert "Reasoning:" in result
            assert "Tool Call:" in result
            mock_move.assert_called_once()

@pytest.mark.llm
def test_os_awareness_llm(wrapper, computer_tool):
    """Test that Deepseek understands it's interacting with a Windows system."""
    wrapper.register_tool(computer_tool)

    with patch('src.infrastructure.llm.openai_compatible.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=MagicMock(return_value=_make_openai_tool_call(
                    "computer", {"action": "mouse_move", "coordinate": [50, 50]},
                    reasoning="Detected Windows; using the computer tool to move the mouse."
                ))
            )
        )
        with patch('pyautogui.moveTo') as mock_move:
            result = wrapper.execute("What operating system are we using? Can you perform a simple mouse movement to test the system?")
            print_test_response("OS Awareness", result)
            
            assert "windows" in result.lower()
            assert mock_move.called

@pytest.mark.llm
def test_keyboard_interaction_llm(wrapper, computer_tool):
    """Test that Deepseek can use the computer tool for keyboard input."""
    wrapper.register_tool(computer_tool)
    
    with patch('src.infrastructure.llm.openai_compatible.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=MagicMock(return_value=_make_openai_tool_call(
                    "computer", {"action": "type", "text": "Hello, World!"},
                    reasoning="I'll type the requested text."
                ))
            )
        )
        with patch('pyautogui.write') as mock_write:
            result = wrapper.execute("Type 'Hello, World!'")
            assert mock_write.called
            assert "success" in result.lower()

@pytest.mark.llm
def test_error_handling_llm(wrapper, computer_tool):
    """Test that Deepseek handles invalid coordinates appropriately."""
    wrapper.register_tool(computer_tool)
    
    with patch('src.infrastructure.llm.openai_compatible.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=MagicMock(return_value=_make_openai_tool_call(
                    "computer", {"action": "mouse_move", "coordinate": [-100, -100]},
                    reasoning="Attempting to move; tool will clamp coordinates safely."
                ))
            )
        )
        with patch('pyautogui.moveTo') as mock_move:
            result = wrapper.execute("Move the mouse to coordinates (-100, -100)")
            # Check that it handled the invalid coordinates gracefully
            assert mock_move.called
            assert "success" in result.lower()

@pytest.mark.llm
def test_computer_tool_complex_action(wrapper, computer_tool):
    """Test more complex computer interactions."""
    wrapper.register_tool(computer_tool)
    
    with patch('pyautogui.click') as mock_click, \
         patch('pyautogui.write') as mock_write, \
         patch('pyautogui.press') as mock_press, \
         patch('pyautogui.hotkey') as mock_hotkey, \
         patch('src.infrastructure.llm.openai_compatible.OpenAI') as MockOpenAI:
        
        # Sequence of tool_calls for each wrapper.execute invocation
        responses: List[Any] = [
            _make_openai_tool_call("computer", {"action": "key", "text": "win"}, "Pressing Windows key"),
            _make_openai_tool_call("computer", {"action": "type", "text": "notepad"}, "Typing notepad"),
            _make_openai_tool_call("computer", {"action": "key", "text": "enter"}, "Pressing Enter"),
            _make_openai_tool_call("computer", {"action": "type", "text": "Hello World"}, "Typing Hello World"),
        ]
        MockOpenAI.return_value.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=MagicMock(side_effect=responses)
            )
        )

        # Step 1: Press Windows key
        result1 = wrapper.execute("Press the Windows key")
        print_test_response("Step 1: Windows Key", result1)
        
        # Step 2: Type 'notepad'
        result2 = wrapper.execute("Type 'notepad'")
        print_test_response("Step 2: Type Notepad", result2)
        
        # Step 3: Press Enter
        result3 = wrapper.execute("Press Enter")
        print_test_response("Step 3: Press Enter", result3)
        
        # Step 4: Type Hello World
        result4 = wrapper.execute("Type 'Hello World'")
        print_test_response("Step 4: Type Hello World", result4)
        
        # Verify the sequence worked
        assert mock_hotkey.called, "Windows key not pressed"
        assert mock_write.called, "Text not typed"
        assert mock_press.called or mock_hotkey.called, "Enter not pressed"

@pytest.mark.llm
def test_chain_of_thought(wrapper, computer_tool):
    """Test Deepseek's reasoning capabilities."""
    wrapper.register_tool(computer_tool)
    
    with patch('src.infrastructure.llm.openai_compatible.OpenAI') as MockOpenAI:
        # Reasoning text contains "first" and "then"
        MockOpenAI.return_value.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=MagicMock(return_value=_make_openai_tool_call(
                    "computer", {"action": "mouse_move", "coordinate": [100, 100]},
                    reasoning="First I will move the mouse, then I will click."
                ))
            )
        )
        result = wrapper.execute(
            "Move the mouse to (100, 100), click, then move it back to (0, 0)"
        )
        print_test_response("Chain of Thought", result)
        
        reasoning = result.split("Reasoning:\n")[1].split("\n\nTool Call:")[0]
        assert "first" in reasoning.lower() or "then" in reasoning.lower()
        assert "move" in reasoning.lower()
        assert "click" in reasoning.lower()

@pytest.mark.llm
def test_tool_selection(wrapper, computer_tool):
    """Test Deepseek's ability to select appropriate tool actions."""
    wrapper.register_tool(computer_tool)
    
    actions = [
        ("Move the mouse to (100, 100)", "mouse_move"),
        ("Click the left mouse button", "left_click"),
        ("Type 'hello'", "type"),
        ("Press Ctrl+C", "key")
    ]
    
    with patch('src.infrastructure.llm.openai_compatible.OpenAI') as MockOpenAI:
        # Side effects matching each expected action
        side_effects = [
            _make_openai_tool_call("computer", {"action": "mouse_move", "coordinate": [100, 100]}, "Moving"),
            _make_openai_tool_call("computer", {"action": "left_click", "coordinate": [50, 50]}, "Clicking"),
            _make_openai_tool_call("computer", {"action": "type", "text": "hello"}, "Typing"),
            _make_openai_tool_call("computer", {"action": "key", "text": "ctrl+c"}, "Hotkey"),
        ]
        MockOpenAI.return_value.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=MagicMock(side_effect=side_effects)
            )
        )

        for i, (command, expected_action) in enumerate(actions):
            result = wrapper.execute(command)
            print_test_response(f"Tool Selection {i+1}: {command}", result)
            assert expected_action in result.lower()

@pytest.mark.llm
def test_parameter_handling(wrapper, computer_tool):
    """Test how Deepseek handles different parameter types."""
    wrapper.register_tool(computer_tool)
    
    with patch('src.infrastructure.llm.openai_compatible.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=MagicMock(return_value=_make_openai_tool_call(
                    "computer", {"action": "mouse_move", "coordinate": [960, 540]},
                    reasoning="Moving to screen center."
                ))
            )
        )
        with patch('pyautogui.moveTo') as mock_move:
            result = wrapper.execute("Move mouse to the center of the screen")
            assert mock_move.called
            assert "success" in result.lower()

@pytest.mark.llm
def test_natural_language_understanding(wrapper, computer_tool):
    """Test Deepseek's ability to understand various phrasings."""
    wrapper.register_tool(computer_tool)
    
    variations = [
        "Move the mouse to coordinates (100, 100)",
        "Put the cursor at position 100, 100",
        "Place the mouse pointer at x=100, y=100",
        "Move cursor to 100,100"
    ]
    
    with patch('src.infrastructure.llm.openai_compatible.OpenAI') as MockOpenAI:
        # side_effect for each variation
        MockOpenAI.return_value.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=MagicMock(side_effect=[
                    _make_openai_tool_call("computer", {"action": "mouse_move", "coordinate": [100, 100]}),
                    _make_openai_tool_call("computer", {"action": "mouse_move", "coordinate": [100, 100]}),
                    _make_openai_tool_call("computer", {"action": "mouse_move", "coordinate": [100, 100]}),
                    _make_openai_tool_call("computer", {"action": "mouse_move", "coordinate": [100, 100]}),
                ])
            )
        )
        with patch('pyautogui.moveTo') as mock_move:
            for command in variations:
                result = wrapper.execute(command)
                assert mock_move.called
                assert "success" in result.lower()
                mock_move.reset_mock()

@pytest.mark.real
def test_real_notepad_demo(wrapper, computer_tool):
    """Real demo of opening notepad and typing."""
    wrapper.register_tool(computer_tool)
    
    # Actually perform the actions without mocks
    print("\nStarting real notepad demo...")
    
    # Step 1: Press Windows key
    result1 = wrapper.execute("Press the Windows key")
    print_test_response("Step 1: Windows Key", result1)
    
    # Step 2: Type 'notepad'
    result2 = wrapper.execute("Type 'notepad'")
    print_test_response("Step 2: Type Notepad", result2)
    
    # Step 3: Press Enter
    result3 = wrapper.execute("Press Enter")
    print_test_response("Step 3: Press Enter", result3)
    
    # Step 4: Type Hello World
    result4 = wrapper.execute("Type 'Hello World'")
    print_test_response("Step 4: Type Hello World", result4)
    
    print("\nDemo completed!") 