import pytest
import os
import json
from tools.code_runner_tool import CodeRunnerTool

@pytest.fixture
def code_runner():
    return CodeRunnerTool()

def test_basic_python_execution(code_runner):
    files = [{
        "path": "test.py",
        "content": "print('Hello from Python!')"
    }]
    result = code_runner.run({
        "files": files,
        "language": "python",
        "main_file": "test.py"
    })
    assert result["type"] == "tool_response"
    assert result["content"] == "Hello from Python!"

def test_python_with_args(code_runner):
    files = [{
        "path": "args_test.py",
        "content": "import sys\nprint(f'Args: {sys.argv[1:]}')"
    }]
    result = code_runner.run({
        "files": files,
        "language": "python",
        "main_file": "args_test.py",
        "args": ["arg1", "arg2"]
    })
    assert result["type"] == "tool_response"
    assert result["content"] == "Args: ['arg1', 'arg2']"

def test_python_with_env(code_runner):
    files = [{
        "path": "env_test.py",
        "content": "import os\nprint(f'TEST_VAR = {os.environ.get(\"TEST_VAR\")}')"
    }]
    result = code_runner.run({
        "files": files,
        "language": "python",
        "main_file": "env_test.py",
        "env": {"TEST_VAR": "test_value"}
    })
    assert result["type"] == "tool_response"
    assert result["content"] == "TEST_VAR = test_value"

def test_execution_timeout(code_runner):
    files = [{
        "path": "timeout_test.py",
        "content": "import time\nwhile True: time.sleep(1)"
    }]
    result = code_runner.run({
        "files": files,
        "language": "python",
        "main_file": "timeout_test.py",
        "timeout": 2
    })
    assert result["type"] == "tool_response"
    assert "Execution timed out after 2 seconds" in result["content"]

def test_file_not_found(code_runner):
    files = []
    result = code_runner.run({
        "files": files,
        "language": "python",
        "main_file": "nonexistent.py"
    })
    assert result["type"] == "tool_response"
    assert "Error:" in result["content"]

def test_invalid_language(code_runner):
    files = [{
        "path": "test.py",
        "content": "print('test')"
    }]
    result = code_runner.run({
        "files": files,
        "language": "invalid",
        "main_file": "test.py"
    })
    assert result["type"] == "tool_response"
    assert "Unsupported language: invalid" in result["content"]

def test_invalid_extension(code_runner):
    files = [{
        "path": "test.txt",
        "content": "print('test')"
    }]
    result = code_runner.run({
        "files": files,
        "language": "python",
        "main_file": "test.txt"
    })
    assert result["type"] == "tool_response"
    assert "File extension not valid for python" in result["content"]

def test_complex_python_project(code_runner):
    files = [
        {
            "path": "requirements.txt",
            "content": "requests==2.31.0"
        },
        {
            "path": "main.py",
            "content": """
import requests
response = requests.get('https://api.github.com/status')
print(f'GitHub API Status: {response.status_code}')
"""
        }
    ]
    result = code_runner.run({
        "files": files,
        "language": "python",
        "main_file": "main.py"
    })
    assert result["type"] == "tool_response"
    assert "GitHub API Status:" in result["content"]

def test_typescript_project(code_runner):
    files = [
        {
            "path": "package.json",
            "content": """{
  "name": "test-project",
  "version": "1.0.0",
  "scripts": {
    "build": "tsc",
    "start": "ts-node src/index.ts"
  },
  "dependencies": {
    "typescript": "^5.0.0",
    "ts-node": "^10.9.0"
  }
}"""
        },
        {
            "path": "tsconfig.json",
            "content": """{
  "compilerOptions": {
    "target": "es5",
    "module": "commonjs",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  }
}"""
        },
        {
            "path": "src/index.ts",
            "content": "console.log('Hello from TypeScript!');"
        }
    ]
    result = code_runner.run({
        "files": files,
        "language": "typescript",
        "main_file": "src/index.ts"
    })
    assert result["type"] == "tool_response"
    assert "Hello from TypeScript!" in result["content"]

@pytest.mark.xfail(reason="Go not installed")
def test_go_project(code_runner):
    files = [
        {
            "path": "go.mod",
            "content": """module example.com/test
go 1.16"""
        },
        {
            "path": "main.go",
            "content": """package main
import "fmt"
func main() {
    fmt.Println("Hello from Go!")
}"""
        }
    ]
    result = code_runner.run({
        "files": files,
        "language": "go",
        "main_file": "main.go"
    })
    assert result["type"] == "tool_response"
    assert "Hello from Go!" in result["content"]

@pytest.mark.xfail(reason="Rust not installed")
def test_rust_project(code_runner):
    files = [
        {
            "path": "Cargo.toml",
            "content": """[package]
name = "test"
version = "0.1.0"
edition = "2021"
"""
        },
        {
            "path": "src/main.rs",
            "content": """fn main() {
    println!("Hello from Rust!");
}"""
        }
    ]
    result = code_runner.run({
        "files": files,
        "language": "rust",
        "main_file": "src/main.rs"
    })
    assert result["type"] == "tool_response"
    assert "Hello from Rust!" in result["content"]

def test_project_with_build_args(code_runner):
    """Test a project with custom build arguments"""
    files = [
        {
            "path": "main.py",
            "content": """
import os
print(f"Build type: {os.getenv('BUILD_TYPE', 'unknown')}")
"""
        }
    ]
    
    result = code_runner.run({
        "files": files,
        "language": "python",
        "main_file": "main.py",
        "build_args": {"args": ["--config", "release"]},
        "env": {"BUILD_TYPE": "release"}
    })
    
    assert result["type"] == "tool_response"
    assert "Build type: release" in result["content"]

def test_project_with_timeout(code_runner):
    """Test project execution with timeout"""
    files = [
        {
            "path": "main.py",
            "content": """
import time
time.sleep(2)
print("Done")
"""
        }
    ]
    
    result = code_runner.run({
        "files": files,
        "language": "python",
        "main_file": "main.py",
        "timeout": 1
    })
    
    assert result["type"] == "tool_response"
    assert "Error: Execution timed out after 1 seconds" in result["content"]

def test_invalid_project(code_runner):
    """Test error handling for invalid project"""
    files = [
        {
            "path": "main.py",
            "content": """
invalid python code
"""
        }
    ]
    
    result = code_runner.run({
        "files": files,
        "language": "python",
        "main_file": "main.py"
    })
    
    assert result["type"] == "tool_response"
    assert "Error: Execution failed with code" in result["content"] 