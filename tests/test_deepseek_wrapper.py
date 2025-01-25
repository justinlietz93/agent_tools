import pytest
from tools.deepseek_wrapper import DeepseekToolWrapper
from tools.package_manager_tool import PackageManagerTool
from tools.file_tool import FileTool
import os
import subprocess

def test_package_manager_tool():
    """Test using PackageManagerTool with Deepseek."""
    wrapper = DeepseekToolWrapper()
    wrapper.register_tool(PackageManagerTool())
    
    # Test package installation
    result = wrapper.execute("Can you help me install the requests package?")
    print("\n=== Installation Test ===")
    print(result)
    
    # Test package info
    result = wrapper.execute("What version of requests is installed? Use the 'info' action to check.")
    print("\n=== Version Check Test ===")
    print(result)

def test_file_tool():
    """Test using FileTool with Deepseek."""
    wrapper = DeepseekToolWrapper()
    wrapper.register_tool(FileTool())
    
    test_file = "test_deepseek.txt"
    
    # Test file creation
    result = wrapper.execute(f"Create a file named {test_file} with the content 'Hello from Deepseek!'")
    print("\n=== File Creation Test ===")
    print(result)
    
    # Test file reading
    result = wrapper.execute(f"Read the contents of {test_file}")
    print("\n=== File Reading Test ===")
    print(result)
    
    # Test file deletion
    result = wrapper.execute(f"Delete the file {test_file}")
    print("\n=== File Deletion Test ===")
    print(result)
    
    # Verify file was deleted
    assert not os.path.exists(test_file), "Test file should be deleted"

def test_python_file_creation():
    """Test creating and running a Python file with Deepseek."""
    wrapper = DeepseekToolWrapper()
    wrapper.register_tool(FileTool())
    
    test_file = "hello_deepseek.py"
    
    # Test Python file creation
    result = wrapper.execute(f"Create a Python file named {test_file} that prints 'Hello from Deepseek!'")
    print("\n=== Python File Creation Test ===")
    print(result)
    
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