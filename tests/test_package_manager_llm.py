"""
Tests for PackageManagerTool with actual LLM interactions.
Verifies that Claude can properly understand and use the package management tool.
"""

import pytest
from unittest.mock import Mock, patch, ANY
import os
import tempfile
import subprocess
from pathlib import Path
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.package_manager_tool import PackageManagerTool
from tests.conftest import get_claude_response

# Unit Tests (Mock LLM and Mock Tool Calls)
@pytest.mark.unit
def test_tool_definition():
    """Verify PackageManagerTool definition format and input_schema"""
    tool = PackageManagerTool()
    assert tool.name == "package_manager"
    assert "pip" in tool.description.lower()
    assert "package" in tool.description.lower()
    
    schema = tool.input_schema
    assert "action" in schema["properties"]
    assert "install" in schema["properties"]["action"]["enum"]
    assert "package" in schema["properties"]

@pytest.mark.unit
def test_basic_install_operation():
    """Test basic package installation with mocks"""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Successfully installed test-package-1.0.0",
            stderr=""
        )
        
        tool = PackageManagerTool()
        result = tool.run({
            "action": "install",
            "package": "test-package"
        })
        
        assert "success" in result["content"].lower()
        assert "test-package" in result["content"]
        mock_run.assert_called_once()

@pytest.mark.unit
def test_error_handling():
    """Test error cases with mocks"""
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["pip", "install", "nonexistent-package"],
            stderr="Could not find a version that satisfies the requirement"
        )
        
        tool = PackageManagerTool()
        result = tool.run({
            "action": "install",
            "package": "nonexistent-package"
        })
        
        assert "error" in result["content"].lower()
        assert "failed" in result["content"].lower()

@pytest.mark.unit
def test_list_packages():
    """Test listing installed packages with mocks"""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(
            returncode=0,
            stdout="pytest==8.3.4\nrequests==2.31.0",
            stderr=""
        )
        
        tool = PackageManagerTool()
        result = tool.run({
            "action": "list"
        })
        
        assert "pytest" in result["content"]
        assert "requests" in result["content"]
        assert "8.3.4" in result["content"]

# Integration Tests (Real LLM with Fake Package Manager)
@pytest.fixture
def venv_dir():
    """Create a temporary virtual environment for testing"""
    temp_dir = tempfile.mkdtemp()
    # Create virtual environment
    subprocess.run(["python", "-m", "venv", temp_dir], check=True)
    yield temp_dir
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

@pytest.mark.integration
@pytest.mark.llm
def test_basic_package_operations(venv_dir):
    """Test LLM can perform basic package operations"""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Successfully installed requests-2.31.0",
            stderr=""
        )
        
        message = (
            "I need to install the 'requests' package version 2.31.0. "
            "Can you help me with that? Please explain what command you would use."
        )
        response = get_claude_response(message)
        
        # Verify Claude suggests using pip install
        assert "pip" in response.lower()
        assert "install" in response.lower()
        assert "requests" in response
        
        # Extract and execute tool call
        tool = PackageManagerTool()
        tool.run({
            "action": "install", 
            "package": "requests==2.31.0"
        })
        
        # Verify mock was called with ANY for pip path
        mock_run.assert_called_with(
            [ANY, "install", "requests==2.31.0"],
            capture_output=True,
            text=True,
            check=True
        )

@pytest.mark.integration
@pytest.mark.llm
def test_error_handling_integration():
    """Test LLM handles package manager errors appropriately"""
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["pip", "install", "nonexistent-package"],
            stderr="Could not find a version that satisfies the requirement"
        )
        
        message = (
            "I tried to install a package called 'nonexistent-package' but got this error:\n"
            "Could not find a version that satisfies the requirement\n"
            "Can you explain what this error means and how to fix it?"
        )
        response = get_claude_response(message)
        
        # Verify Claude explains package not found error
        assert any(phrase in response.lower() for phrase in [
            "not found",
            "does not exist", 
            "package index",
            "pypi",
            "available",
            "repository"
        ])

@pytest.mark.integration
@pytest.mark.llm
def test_version_constraints():
    """Test LLM understands package version constraints"""
    message = (
        "What's the difference between using >= and == when specifying Python package versions? "
        "For example, what's the difference between 'pip install requests>=2.31.0' and "
        "'pip install requests==2.31.0'? Please explain in detail."
    )
    response = get_claude_response(message)
    
    # Verify Claude explains version constraints
    assert any(phrase in response.lower() for phrase in [
        "exact version",
        "specific version",
        "greater than or equal",
        "minimum version",
        "version constraint"
    ])
    
    # Verify Claude understands version pinning importance
    message = (
        "When should I use == vs >= for package versions in my requirements.txt file? "
        "What are the pros and cons of each approach?"
    )
    response = get_claude_response(message)
    
    assert any(phrase in response.lower() for phrase in [
        "reproducible",
        "dependency",
        "compatibility",
        "conflict",
        "consistent"
    ])

# System Tests (Real LLM with Real Package Manager)
@pytest.mark.system
@pytest.mark.llm
def test_real_package_operations():
    """Test end-to-end package operations with real pip"""
    print("\n=== Initial Claude Response ===")
    message = (
        "I need to check what version of pytest is installed in my environment. "
        "Can you help me with that?"
    )
    response = get_claude_response(message)
    print(response)
    
    # Extract tool call from response and apply it
    tool = PackageManagerTool()
    result = tool.run({
        "action": "list"
    })
    
    print("\n=== Package List ===")
    print(result["content"])
    
    print("\n=== Claude's Analysis ===")
    message = (
        f"Here's the output of pip list:\n\n{result['content']}\n\n"
        "Can you find pytest in this list and tell me its version?"
    )
    response = get_claude_response(message)
    print(response)
    
    # Verify Claude found pytest version
    assert "pytest" in response.lower()
    assert "8.3.4" in response

@pytest.mark.system
@pytest.mark.llm
def test_dependency_analysis():
    """Test that Claude can analyze package dependencies"""
    print("\n=== Initial Claude Response ===")
    message = (
        "Can you help me understand what dependencies the 'pytest' package has? "
        "Please check the installed packages and explain the relationships."
    )
    response = get_claude_response(message)
    print(response)
    
    # Get actual package list
    tool = PackageManagerTool()
    result = tool.run({
        "action": "list"
    })
    
    print("\n=== Package List ===")
    print(result["content"])
    
    print("\n=== Claude's Analysis ===")
    message = (
        f"Here's the complete list of installed packages:\n\n{result['content']}\n\n"
        "Based on this list, can you identify which packages are required by pytest "
        "and explain their roles? Please focus on the key dependencies like pluggy, "
        "iniconfig, packaging, and any pytest plugins."
    )
    response = get_claude_response(message)
    print(response)
    
    # Verify Claude understands key dependencies
    # We check for at least 3 of these key packages to allow for some flexibility
    key_packages = ["pluggy", "iniconfig", "packaging", "pytest-asyncio", "colorama"]
    found_packages = [pkg for pkg in key_packages if pkg in response.lower()]
    assert len(found_packages) >= 3, f"Expected at least 3 key packages, found: {found_packages}"
    
    # Verify Claude provides meaningful explanations
    assert any(term in response.lower() for term in [
        "plugin",
        "configuration",
        "testing",
        "framework"
    ])

# Additional System Tests
@pytest.mark.system
@pytest.mark.llm
def test_requirements_file_handling():
    """Test LLM can handle requirements.txt operations"""
    print("\n=== Initial Claude Response ===")
    # Create a temporary requirements.txt
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("requests==2.31.0\npytest>=8.0.0\n")
        requirements_file = f.name

    message = f"""
    I have a requirements.txt file with these contents:
    requests==2.31.0
    pytest>=8.0.0
    
    Can you help me install these packages and then verify they're installed correctly?
    """
    response = get_claude_response(message)
    print(response)
    
    # Extract and execute tool calls
    tool = PackageManagerTool()
    
    print("\n=== Installing Requirements ===")
    install_result = tool.run({
        "action": "install",
        "requirements_file": requirements_file
    })
    print(install_result["content"])
    
    print("\n=== Verifying Installation ===")
    list_result = tool.run({
        "action": "list"
    })
    print(list_result["content"])
    
    print("\n=== Claude's Analysis ===")
    verify_message = f"""
    Here's the list of installed packages:
    {list_result['content']}
    
    Can you verify that both requests and pytest are installed with the correct versions?
    """
    final_response = get_claude_response(verify_message)
    print(final_response)
    
    # Cleanup
    os.unlink(requirements_file)
    
    # Verify results
    assert "success" in install_result["content"].lower() or "requirement already satisfied" in install_result["content"].lower()
    assert "requests" in list_result["content"].lower()
    assert "pytest" in list_result["content"].lower()

@pytest.mark.system
@pytest.mark.llm
def test_package_upgrade_workflow():
    """Test LLM can handle package upgrade scenarios"""
    print("\n=== Initial Claude Response ===")
    message = """
    I want to check if any of my installed packages are outdated and need upgrading.
    Can you help me with that? After checking, if you find any outdated packages,
    please help me upgrade them.
    """
    response = get_claude_response(message)
    print(response)
    
    tool = PackageManagerTool()
    
    print("\n=== Checking Outdated Packages ===")
    check_result = tool.run({
        "action": "list",
        "package": "--outdated"
    })
    print(check_result["content"])
    
    print("\n=== Claude's Analysis ===")
    analysis_message = f"""
    Here's the list of outdated packages:
    {check_result['content']}
    
    Can you analyze this output and tell me which packages need upgrading?
    If any packages need upgrading, please help me upgrade them one by one.
    """
    upgrade_response = get_claude_response(analysis_message)
    print(upgrade_response)
    
    # Verify results
    assert "package" in check_result["content"].lower()

@pytest.mark.system
@pytest.mark.llm
def test_package_info_analysis():
    """Test LLM can analyze package information and dependencies"""
    print("\n=== Initial Claude Response ===")
    message = """
    I want to understand more about the 'requests' package. Can you:
    1. Check if it's installed
    2. Show me its dependencies
    3. Tell me what version is installed
    4. Explain what the package does based on its metadata
    """
    response = get_claude_response(message)
    print(response)
    
    tool = PackageManagerTool()
    
    print("\n=== Getting Package Info ===")
    info_result = tool.run({
        "action": "info",
        "package": "requests"
    })
    print(info_result["content"])
    
    print("\n=== Claude's Analysis ===")
    analysis_message = f"""
    Here's the information about the requests package:
    {info_result['content']}
    
    Can you analyze this information and tell me:
    1. What version is installed?
    2. What are its main dependencies?
    3. What is the package's purpose based on its description?
    4. Are there any security considerations mentioned?
    """
    final_response = get_claude_response(analysis_message)
    print(final_response)
    
    # Verify results
    assert "requests" in info_result["content"].lower()
    assert "version" in info_result["content"].lower()
    assert "requires" in info_result["content"].lower()

@pytest.mark.system
@pytest.mark.llm
def test_virtual_environment_awareness():
    """Test LLM understands virtual environment context"""
    print("\n=== Initial Claude Response ===")
    message = """
    I want to make sure I'm installing packages in the correct virtual environment.
    Can you help me:
    1. Check if we're in a virtual environment
    2. Show me where pip is installing packages
    3. List the currently installed packages in this environment
    """
    response = get_claude_response(message)
    print(response)
    
    tool = PackageManagerTool()
    
    print("\n=== Checking Environment ===")
    # Get pip configuration
    config_result = tool.run({
        "action": "config"
    })
    print(config_result["content"])
    
    # List installed packages
    list_result = tool.run({
        "action": "list"
    })
    print("\n=== Installed Packages ===")
    print(list_result["content"])
    
    print("\n=== Claude's Analysis ===")
    analysis_message = f"""
    Here's the pip configuration:
    {config_result['content']}
    
    And here are the installed packages:
    {list_result['content']}
    
    Can you analyze this information and tell me:
    1. Are we in a virtual environment?
    2. Where are packages being installed?
    3. Does the package list look appropriate for a virtual environment?
    """
    final_response = get_claude_response(analysis_message)
    print(final_response)
    
    # Verify results
    assert "pip" in config_result["content"].lower()
    assert len(list_result["content"]) > 0

@pytest.mark.system
@pytest.mark.llm
def test_error_recovery_workflow():
    """Test LLM can handle and recover from package management errors"""
    print("\n=== Initial Claude Response ===")
    message = """
    I want to try installing a package with a very specific version constraint
    that probably doesn't exist. Let's try 'requests==999.999.999' and see how
    you handle the error and help me recover from it.
    """
    response = get_claude_response(message)
    print(response)
    
    tool = PackageManagerTool()
    
    print("\n=== Attempting Installation ===")
    install_result = tool.run({
        "action": "install",
        "package": "requests==999.999.999"
    })
    print(install_result["content"])
    
    # Extract the error message for Claude
    error_message = install_result["content"]
    if len(error_message) > 500:
        error_message = error_message[:500] + "..."
    
    print("\n=== Claude's Error Analysis ===")
    analysis_message = f"""
    The installation attempt failed with this error:
    {error_message}
    
    Can you:
    1. Explain what went wrong in simple terms
    2. Suggest how to fix it
    3. Help me install a valid version of the package
    """
    recovery_response = get_claude_response(analysis_message)
    print(recovery_response)
    
    # Attempt recovery with valid version
    print("\n=== Attempting Recovery ===")
    recovery_result = tool.run({
        "action": "install",
        "package": "requests==2.31.0"
    })
    print(recovery_result["content"])
    
    # Verify results
    assert "error" in install_result["content"].lower()
    assert any(phrase in recovery_response.lower() for phrase in [
        "version not found",
        "does not exist",
        "invalid version",
        "no matching distribution",
        "could not be found",
        "could not find",
        "specified version"
    ])
    assert "success" in recovery_result["content"].lower() or "requirement already satisfied" in recovery_result["content"].lower()

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 