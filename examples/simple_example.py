"""Simple example of using Deepseek Reasoner."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def main():
    """Run a simple demonstration."""
    try:
        print("Initializing Deepseek...")
        wrapper = DeepseekToolWrapper()
        computer = ComputerTool()
        wrapper.register_tool(computer)
        
        print("\nStarting demo in 3 seconds...")
        time.sleep(3)
        
        # Move mouse
        result = wrapper.execute("Move the mouse to coordinates (100, 100)")
        print_response("Moving Mouse", result)
        time.sleep(1)
        
        # Type text
        result = wrapper.execute("Type 'Hello from Deepseek!'")
        print_response("Typing Text", result)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. Your .env file has DEEPSEEK_API_KEY")
        print("2. All dependencies are installed")

if __name__ == "__main__":
    main() 