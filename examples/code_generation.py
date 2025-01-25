"""Example of using Deepseek to generate and run a Python program."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.deepseek_wrapper import DeepseekToolWrapper
from tools.file_tool import FileTool
from tools.computer_tool import ComputerTool
from tools.package_manager_tool import PackageManagerTool
import time

def print_response(step: str, response: str):
    """Print formatted response from Deepseek."""
    print(f"\n{'='*80}")
    print(f"STEP: {step}")
    print(f"{'='*80}")
    print(response)
    print(f"{'='*80}\n")

def main():
    """Generate and run a Pomodoro timer application using Deepseek."""
    try:
        print("Initializing Deepseek...")
        wrapper = DeepseekToolWrapper()
        file_tool = FileTool()
        computer = ComputerTool()
        package_manager = PackageManagerTool()
        
        wrapper.register_tool(file_tool)
        wrapper.register_tool(computer)
        wrapper.register_tool(package_manager)
        
        # Let Deepseek handle dependency installation
        print("\nChecking and installing dependencies...")
        result = wrapper.execute(
            "Check if plyer package is installed, if not, install it"
        )
        print_response("Dependency Check", result)
        
        prompt = """
        Create a Python program that implements a Pomodoro timer with these features:
        1. GUI using tkinter (standard library)
        2. 25-minute work timer
        3. 5-minute break timer
        4. Start, pause, and reset buttons
        5. Desktop notifications using plyer (already installed)
        6. Progress bar showing time remaining
        7. Option to customize timer durations
        8. Save settings between sessions
        
        Make it a single file that's well-documented and handles errors gracefully.
        """
        
        # Generate the program
        print("\nGenerating program...")
        result = wrapper.execute(
            f"Create a new file at examples/pomodoro_timer.py with this Python program: {prompt}"
        )
        print_response("Program Generation", result)
        
        # Run the program
        print("\nRunning the program...")
        result = wrapper.execute(
            "Run the Python script at examples/pomodoro_timer.py"
        )
        print_response("Program Execution", result)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. Your .env file has DEEPSEEK_API_KEY")
        print("2. You're in the correct directory")
        print("3. Your virtual environment is activated")

if __name__ == "__main__":
    main() 