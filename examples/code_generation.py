"""Example of using Deepseek to generate and run a Python program."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.deepseek_wrapper import DeepseekToolWrapper
from tools.file_tool import FileTool
from tools.shell_tool import ShellTool
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
        shell_tool = ShellTool()
        package_manager = PackageManagerTool()
        
        # Register tools with their input_schema
        wrapper.register_tool(file_tool)
        wrapper.register_tool(shell_tool)
        wrapper.register_tool(package_manager)
        
        prompt = """
        Create a Python program that implements a Pomodoro timer with these features:
        1. GUI using tkinter (standard library)
        2. 25-minute work timer
        3. 5-minute break timer
        4. Start, pause, and reset buttons
        5. Desktop notifications (use any appropriate package)
        6. Progress bar showing time remaining
        7. Option to customize timer durations
        8. Save settings between sessions
        
        Make it a single file that's well-documented and handles errors gracefully.
        Include required package imports at the top of the file.
        
        Use the file tool to create this program at examples/pomodoro_timer.py
        """
        
        # Generate the program
        print("\nGenerating program...")
        result = wrapper.execute(prompt)
        print_response("Program Generation", result)
        
        # Check and install dependencies
        print("\nChecking for required packages...")
        result = wrapper.execute(
            """Use the file tool to read examples/pomodoro_timer.py, 
            then use the package manager tool to install any imported packages that aren't standard library."""
        )
        print_response("Package Installation", result)
        
        # Run the program
        print("\nRunning the program...")
        result = wrapper.execute(
            """Use the shell tool to run the Python program at examples/pomodoro_timer.py.
            The shell tool can execute python commands directly."""
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