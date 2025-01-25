"""
Windows-specific implementation of ComputerTool for screen, keyboard, and mouse interaction.
Uses pyautogui for cross-platform control and win32api for Windows-specific functionality.
"""

import base64
import io
import os
import time
from enum import Enum
from typing import Dict, Any, Literal, Optional, Tuple, Union

import pyautogui
import win32api
import win32con
import win32gui
from PIL import Image

from .tool_base import Tool

# Configure pyautogui
pyautogui.FAILSAFE = True  # Move mouse to corner to abort
pyautogui.PAUSE = 0.5  # Increase delay between actions for stability

class Action(str, Enum):
    KEY = "key"
    TYPE = "type"
    MOUSE_MOVE = "mouse_move"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    MIDDLE_CLICK = "middle_click"
    DOUBLE_CLICK = "double_click"
    SCREENSHOT = "screenshot"
    CURSOR_POSITION = "cursor_position"
    LEFT_CLICK_DRAG = "left_click_drag"
    FIND_WINDOW = "find_window"
    MOVE_WINDOW = "move_window"
    SET_WINDOW_FOCUS = "set_window_focus"
    GET_WINDOW_INFO = "get_window_info"

class ComputerTool(Tool):
    """Tool for interacting with the computer's screen, keyboard, and mouse."""
    
    name = "computer"
    description = """A tool for controlling the computer's mouse, keyboard, and windows.
    
    Available actions:
    1. Mouse Movement:
       - action: "mouse_move"
       - coordinate: [x, y] (required)
       Example: {"action": "mouse_move", "coordinate": [500, 500]}
    
    2. Mouse Clicks:
       - action: "left_click", "right_click", "middle_click", or "double_click"
       - coordinate: [x, y] (optional)
       Example: {"action": "left_click", "coordinate": [500, 500]}
    
    3. Keyboard Input:
       - action: "type" for text, "key" for key combinations
       - text: string (required)
       Examples: 
       - {"action": "type", "text": "Hello World"}
       - {"action": "key", "text": "ctrl+c"}
    
    4. Screen Capture:
       - action: "screenshot"
       Example: {"action": "screenshot"}
    
    5. Cursor Position:
       - action: "cursor_position"
       Example: {"action": "cursor_position"}

    6. Window Control:
       - action: "find_window"
       - title: window title to find (required)
       Example: {"action": "find_window", "title": "Notepad"}

       - action: "move_window"
       - window_title: title of window to move (required)
       - position: [x, y] (required)
       - size: [width, height] (optional)
       Example: {"action": "move_window", "window_title": "Notepad", "position": [0, 0], "size": [800, 600]}

       - action: "set_window_focus"
       - window_title: title of window to focus (required)
       Example: {"action": "set_window_focus", "window_title": "Notepad"}

       - action: "get_window_info"
       - window_title: title of window to get info for (required)
       Example: {"action": "get_window_info", "window_title": "Notepad"}"""
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        """Get the input schema for the tool."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [e.value for e in Action],
                    "description": "The action to perform"
                },
                "text": {
                    "type": "string",
                    "description": "Text to type or key command to send"
                },
                "coordinate": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Screen coordinates [x, y]"
                },
                "window_title": {
                    "type": "string",
                    "description": "Title of window to control"
                },
                "position": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Window position [x, y]"
                },
                "size": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Window size [width, height]"
                }
            },
            "required": ["action"]
        }

    def __init__(self):
        """Initialize the tool with screen dimensions."""
        super().__init__()
        self.screen_width, self.screen_height = pyautogui.size()

    def _validate_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """Validate and adjust coordinates to be within screen bounds."""
        x = max(0, min(x, self.screen_width - 1))
        y = max(0, min(y, self.screen_height - 1))
        return x, y

    def _take_screenshot(self) -> str:
        """Take a screenshot and return as base64 string."""
        screenshot = pyautogui.screenshot()
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def find_and_move_window(self, title_substring: str) -> Optional[int]:
        """Find a window by title substring and move it to the primary monitor."""
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title_substring.lower() in title.lower():
                    windows.append(hwnd)
            return True
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        
        if not windows:
            return None
            
        hwnd = windows[0]
        try:
            # Get current window placement
            placement = win32gui.GetWindowPlacement(hwnd)
            rect = win32gui.GetWindowRect(hwnd)
            
            # Calculate window size
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            
            # Move to primary monitor (0,0 is top-left)
            new_x = 100
            new_y = 100
            
            # Set window to normal state (not minimized/maximized)
            placement = list(placement)
            placement[1] = win32con.SW_NORMAL
            win32gui.SetWindowPlacement(hwnd, tuple(placement))
            
            # Move window
            win32gui.MoveWindow(hwnd, new_x, new_y, width, height, True)
            
            # Bring to front and focus
            win32gui.SetForegroundWindow(hwnd)
            
            return hwnd
        except Exception as e:
            print(f"Error moving window: {e}")
            return None

    def find_window_by_title(self, title: str) -> Optional[int]:
        """Find a window by its title."""
        try:
            hwnd = win32gui.FindWindow(None, title)
            if hwnd and win32gui.IsWindowVisible(hwnd):
                return hwnd
            # Try partial match
            def callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
                    if title.lower() in window_title.lower():
                        windows.append(hwnd)
            windows = []
            win32gui.EnumWindows(callback, windows)
            return windows[0] if windows else None
        except Exception as e:
            print(f"Error finding window: {e}")
            return None

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the requested computer interaction action."""
        tool_use_id = input.get("tool_use_id", "")
        
        try:
            action = Action(input["action"])
            text = input.get("text")
            coordinate = input.get("coordinate")
            window_title = input.get("window_title")
            position = input.get("position")
            size = input.get("size")

            # Window management actions
            if action == Action.FIND_WINDOW:
                if not window_title:
                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": {
                            "error": "window_title is required for find_window action",
                            "action": action.value
                        }
                    }
                hwnd = self.find_window_by_title(window_title)
                if hwnd:
                    rect = win32gui.GetWindowRect(hwnd)
                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": {
                            "status": "success",
                            "action": action.value,
                            "window_handle": hwnd,
                            "position": [rect[0], rect[1]],
                            "size": [rect[2] - rect[0], rect[3] - rect[1]],
                            "title": win32gui.GetWindowText(hwnd),
                            "class": win32gui.GetClassName(hwnd)
                        }
                    }
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": {
                        "error": f"Window with title '{window_title}' not found",
                        "action": action.value
                    }
                }

            elif action == Action.MOVE_WINDOW:
                if not window_title or not position:
                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": {
                            "error": "window_title and position are required for move_window action",
                            "action": action.value
                        }
                    }
                hwnd = self.find_window_by_title(window_title)
                if hwnd:
                    if not size:
                        rect = win32gui.GetWindowRect(hwnd)
                        size = [rect[2] - rect[0], rect[3] - rect[1]]
                    win32gui.MoveWindow(hwnd, position[0], position[1], size[0], size[1], True)
                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": {
                            "status": "success",
                            "action": action.value,
                            "message": f"Moved window to ({position[0]}, {position[1]}) with size ({size[0]}, {size[1]})"
                        }
                    }
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": {
                        "error": f"Window with title '{window_title}' not found",
                        "action": action.value
                    }
                }

            elif action == Action.SET_WINDOW_FOCUS:
                if not window_title:
                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": {
                            "error": "window_title is required for set_window_focus action",
                            "action": action.value
                        }
                    }
                hwnd = self.find_window_by_title(window_title)
                if hwnd:
                    # Ensure window is not minimized
                    if win32gui.IsIconic(hwnd):
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    win32gui.SetForegroundWindow(hwnd)
                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": {
                            "status": "success",
                            "action": action.value,
                            "message": f"Set focus to window '{window_title}'"
                        }
                    }
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": {
                        "error": f"Window with title '{window_title}' not found",
                        "action": action.value
                    }
                }

            elif action == Action.GET_WINDOW_INFO:
                if not window_title:
                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": {
                            "error": "window_title is required for get_window_info action",
                            "action": action.value
                        }
                    }
                hwnd = self.find_window_by_title(window_title)
                if hwnd:
                    rect = win32gui.GetWindowRect(hwnd)
                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": {
                            "status": "success",
                            "action": action.value,
                            "window_handle": hwnd,
                            "position": [rect[0], rect[1]],
                            "size": [rect[2] - rect[0], rect[3] - rect[1]],
                            "title": win32gui.GetWindowText(hwnd),
                            "class": win32gui.GetClassName(hwnd),
                            "visible": win32gui.IsWindowVisible(hwnd),
                            "enabled": win32gui.IsWindowEnabled(hwnd),
                            "minimized": win32gui.IsIconic(hwnd),
                            "foreground": hwnd == win32gui.GetForegroundWindow()
                        }
                    }
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": {
                        "error": f"Window with title '{window_title}' not found",
                        "action": action.value
                    }
                }

            # Existing actions
            if action in [Action.MOUSE_MOVE, Action.LEFT_CLICK_DRAG]:
                if not coordinate:
                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": {
                            "error": "coordinate is required for mouse movement",
                            "action": action.value
                        }
                    }
                x, y = self._validate_coordinates(coordinate[0], coordinate[1])
                
                if action == Action.MOUSE_MOVE:
                    pyautogui.moveTo(x, y, duration=0.5)  # Slower movement
                else:  # LEFT_CLICK_DRAG
                    pyautogui.dragTo(x, y, button='left', duration=0.5)
                
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": {
                        "status": "success",
                        "action": action.value,
                        "message": f"Mouse {'moved' if action == Action.MOUSE_MOVE else 'dragged'} to ({x}, {y})"
                    }
                }

            elif action in [Action.KEY, Action.TYPE]:
                if not text:
                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": {
                            "error": "text is required for keyboard actions",
                            "action": action.value
                        }
                    }
                
                # Store current window focus
                active_window = win32gui.GetForegroundWindow()
                
                if action == Action.KEY:
                    pyautogui.hotkey(*text.split('+'))
                else:  # TYPE
                    pyautogui.write(text, interval=0.1)  # Slower typing
                
                # Restore window focus
                if active_window:
                    win32gui.SetForegroundWindow(active_window)
                
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": {
                        "status": "success",
                        "action": action.value,
                        "message": f"{'Keys pressed' if action == Action.KEY else 'Text typed'}: {text}"
                    }
                }

            elif action == Action.SCREENSHOT:
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": {
                        "status": "success",
                        "action": action.value,
                        "type": "image",
                        "format": "base64",
                        "data": self._take_screenshot()
                    }
                }

            elif action == Action.CURSOR_POSITION:
                x, y = win32api.GetCursorPos()
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": {
                        "status": "success",
                        "action": action.value,
                        "position": {"x": x, "y": y},
                        "message": f"Cursor position: ({x}, {y})"
                    }
                }

            else:  # Click actions
                if coordinate:
                    x, y = self._validate_coordinates(coordinate[0], coordinate[1])
                    pyautogui.moveTo(x, y, duration=0.5)

                click_params = {
                    Action.LEFT_CLICK: {"button": "left"},
                    Action.RIGHT_CLICK: {"button": "right"},
                    Action.MIDDLE_CLICK: {"button": "middle"},
                    Action.DOUBLE_CLICK: {"button": "left", "clicks": 2, "interval": 0.25}
                }[action]

                pyautogui.click(**click_params)
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": {
                        "status": "success",
                        "action": action.value,
                        "message": f"Mouse {action.value} performed"
                    }
                }

        except Exception as e:
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,  # Now tool_use_id is always defined
                "content": {
                    "error": str(e),
                    "action": input.get("action"),
                    "traceback": str(e.__traceback__)
                }
            } 