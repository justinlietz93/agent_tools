# selfprompter/tools/code_runner_tool.py

import subprocess
import os
import json
import shutil
import tempfile
import signal
import sys
import psutil
from typing import Dict, Any, Optional, List, Literal, Union
from pathlib import Path
from tools.tool_base import Tool

def kill_proc_tree(pid, including_parent=True):
    """Kill a process tree (including grandchildren) with signal.SIGTERM"""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        if including_parent:
            try:
                parent.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(children + ([parent] if including_parent else []), timeout=1)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass

def run_with_timeout(cmd: str, cwd: str, timeout: int, env: Dict[str, str] = None) -> subprocess.CompletedProcess:
    """Run a command with timeout and proper cleanup on Windows"""
    process = None
    try:
        # Merge environment variables with system environment
        merged_env = os.environ.copy()
        if env:
            merged_env.update({k: str(v) for k, v in env.items() if v is not None})

        if os.name == 'nt':
            # On Windows, use absolute path for python
            python_exe = sys.executable
            if cmd.startswith('python '):
                cmd = f'"{python_exe}" {cmd[7:]}'
            
            # Create startupinfo to properly configure Windows process
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=merged_env,
                shell=True,
                startupinfo=startupinfo
            )
        else:
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=merged_env,
                shell=True,
                preexec_fn=os.setsid
            )

        stdout, stderr = process.communicate(timeout=timeout)
        if process.returncode != 0:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout="",
                stderr=f"Error: Execution failed with code {process.returncode}\n{stderr}"
            )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr
        )
    except subprocess.TimeoutExpired:
        if process:
            kill_proc_tree(process.pid)
            process.kill()
        raise
    except Exception as e:
        if process:
            kill_proc_tree(process.pid)
            process.kill()
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr=f"Error: Execution failed with code 1: {str(e)}"
        )

class CodeRunnerTool(Tool):
    """
    Advanced tool for executing complex code projects in various languages.
    Supports multi-file projects, dependencies, and build steps.
    Follows Anthropic Claude tool use standards.
    """

    name: Literal["code_runner"] = "code_runner"

    def __init__(self, working_dir: str = "./", timeout: int = 30):
        """
        Initialize with working directory and default timeout.
        
        Args:
            working_dir: Base directory for code execution
            timeout: Default timeout in seconds
        """
        self.working_dir = os.path.abspath(working_dir)
        self.timeout = min(timeout, 3600)  # Max 1 hour timeout
        self.language_configs = {
            "python": {
                "file_ext": ".py",
                "run_cmd": "python",
                "install_cmd": "pip install -r requirements.txt",
                "package_file": "requirements.txt"
            },
            "typescript": {
                "file_ext": ".ts",
                "run_cmd": "npx ts-node",
                "install_cmd": "npm install",
                "package_file": "package.json"
            },
            "go": {
                "file_ext": ".go", 
                "run_cmd": "go run",
                "install_cmd": "go mod download",
                "package_file": "go.mod"
            },
            "rust": {
                "file_ext": ".rs",
                "run_cmd": "cargo run",
                "install_cmd": "cargo build",
                "package_file": "Cargo.toml"
            }
        }

    @property
    def description(self) -> str:
        return """Executes code files in various programming languages (Python, TypeScript, Go, Rust).
        Supports multi-file projects, package dependencies, and build steps.
        Required input_schema:
        - files: List of files with paths and contents
        - language: Programming language (python, typescript, go, rust) 
        - main_file: Path to the main file to execute
        Optional input_schema:
        - args: Command line arguments
        - env: Environment variables
        - timeout: Maximum execution time in seconds
        - build_args: Additional build arguments"""

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["path", "content"]
                    }
                },
                "language": {"type": "string"},
                "main_file": {"type": "string"},
                "args": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "env": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                },
                "timeout": {"type": "integer"},
                "build_args": {"type": "string"}
            },
            "required": ["files", "language", "main_file"]
        }

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a complex code project.
        
        Args:
            input: Dictionary containing:
                files: List of files to create (path and content)
                language: Programming language
                main_file: Entry point file
                args: Command line arguments
                env: Environment variables
                timeout: Execution timeout
            
        Returns:
            Dictionary containing execution output or error message
        """
        try:
            files = input.get("files", [])
            language = input.get("language", "").lower()
            main_file = input.get("main_file", "")
            args = input.get("args", [])
            env_vars = input.get("env", {})
            timeout = min(input.get("timeout", self.timeout), 3600)
            build_args = input.get("build_args", "")

            # Validate language
            if language not in self.language_configs:
                return self._error(f"Unsupported language: {language}")

            config = self.language_configs[language]
            
            # Validate file extension
            if not main_file.endswith(config["file_ext"]):
                return self._error(f"File extension not valid for {language}")

            # Create temporary project directory
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Create all files
                    for file_info in files:
                        file_path = os.path.join(temp_dir, file_info["path"])
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        with open(file_path, "w") as f:
                            f.write(file_info["content"])

                    # Install dependencies if package file exists
                    package_file = os.path.join(temp_dir, config["package_file"])
                    if os.path.exists(package_file):
                        try:
                            print(f"Installing dependencies with command: {config['install_cmd']}")
                            install_result = run_with_timeout(
                                config["install_cmd"],
                                temp_dir,
                                timeout,
                                env_vars
                            )
                            print(f"Install result: {install_result.stdout}\n{install_result.stderr}")
                            if install_result.returncode != 0:
                                return self._error(f"Dependency installation failed: {install_result.stderr}")
                        except subprocess.TimeoutExpired:
                            return self._error(f"Dependency installation timed out after {timeout} seconds")

                    # Run build step if needed
                    if "build_cmd" in config:
                        build_cmd = f"{config['build_cmd']} {build_args}".strip()
                        try:
                            print(f"Running build command: {build_cmd}")
                            build_result = run_with_timeout(
                                build_cmd,
                                temp_dir,
                                timeout,
                                env_vars
                            )
                            print(f"Build result: {build_result.stdout}\n{build_result.stderr}")
                            if build_result.returncode != 0:
                                return self._error(f"Build failed: {build_result.stderr}")
                        except subprocess.TimeoutExpired:
                            return self._error(f"Build step timed out after {timeout} seconds")

                    # Execute the main file
                    cmd = f"{config['run_cmd']} {main_file}"
                    if args:
                        cmd += f" {' '.join(args)}"

                    env = os.environ.copy()
                    env.update(env_vars)

                    try:
                        print(f"Running command: {cmd}")
                        result = run_with_timeout(
                            cmd,
                            temp_dir,
                            timeout,
                            env
                        )
                        print(f"Execution result: {result.stdout}\n{result.stderr}")
                        if result.returncode != 0:
                            return self._error(f"Execution failed: {result.stderr}")

                        output = result.stdout.strip() or "Execution completed successfully (no output)"
                        return self._success(output)

                    except subprocess.TimeoutExpired:
                        return self._error(f"Execution timed out after {timeout} seconds")

                except Exception as e:
                    return self._error(f"Error: {str(e)}")

        except Exception as e:
            return self._error(str(e))

    def _success(self, content: str) -> Dict[str, Any]:
        return {
            "type": "tool_response",
            "tool_use_id": "",
            "content": content
        }
        
    def _error(self, message: str) -> Dict[str, Any]:
        return {
            "type": "tool_response",
            "tool_use_id": "",
            "content": f"Error: {message}"
        }
