"""Basic examples of using Deepseek Reasoner."""

from tools.deepseek_wrapper import DeepseekToolWrapper
from tools.computer_tool import ComputerTool
import time

def print_response(step: str, response: str):
    """Print formatted response from Deepseek."""
    print(f"\n{'='*80}")
    print(f"STEP: {step}")
    print(f"{'='*80}")
    print(response)
    print(f"{'='*80}\n")

def setup_tools():
    """Initialize the wrapper and tools."""
    wrapper = DeepseekToolWrapper()
    computer = ComputerTool()
    wrapper.register_tool(computer)
    return wrapper

def mouse_examples(wrapper):
    """Examples of mouse control."""
    result = wrapper.execute("Move the mouse to coordinates (100, 100)")
    print_response("Basic Mouse Movement", result)
    time.sleep(0.5)
    
    result = wrapper.execute("Move the mouse right by 50 pixels")
    print_response("Relative Mouse Movement", result)
    time.sleep(0.5)
    
    result = wrapper.execute("Double click")
    print_response("Mouse Click", result)

def keyboard_examples(wrapper):
    """Examples of keyboard input."""
    result = wrapper.execute("Type 'Hello from Deepseek!'")
    print_response("Basic Typing", result)
    
    result = wrapper.execute("Press Ctrl+A to select all")
    print_response("Key Combination", result)

def window_examples(wrapper):
    """Examples of window management."""
    result = wrapper.execute("Focus the Chrome window")
    print_response("Window Focus", result)
    
    result = wrapper.execute("Move the current window to (0, 0)")
    print_response("Window Movement", result)

def notepad_example(wrapper):
    """Example of opening Notepad and typing."""
    result = wrapper.execute("Press the Windows key")
    print_response("Open Start Menu", result)
    
    result = wrapper.execute("Type 'notepad'")
    print_response("Search for Notepad", result)
    
    result = wrapper.execute("Press Enter")
    print_response("Launch Notepad", result)
    
    result = wrapper.execute("Type 'This is an example of automated typing.'")
    print_response("Type in Notepad", result)

if __name__ == "__main__":
    try:
        print("Initializing tools...")
        wrapper = setup_tools()
        if not wrapper:
            print("Failed to initialize tools. Exiting.")
            exit(1)
        
        print("\nStarting examples in 3 seconds...")
        time.sleep(3)
        
        mouse_examples(wrapper)
        keyboard_examples(wrapper)
        window_examples(wrapper)
        notepad_example(wrapper)
        
        print("\nExamples completed!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Check your .env file has DEEPSEEK_API_KEY")
        print("2. Run 'pip install -r requirements.txt'")
        print("3. Try running terminal as administrator") 