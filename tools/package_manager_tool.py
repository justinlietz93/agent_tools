# selfprompter/tools/package_manager_tool.py

import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, TypedDict

from .tool_base import Tool
from .config import Config

class PackageManagerOptions(TypedDict):
    """Options for package manager operations."""
    virtual_env: Optional[str]
    index_url: Optional[str]
    extra_index_url: Optional[str]
    trusted_host: Optional[str]
    require_virtualenv: bool
    no_deps: bool
    pre: bool

class PackageManagerTool(Tool):
    """Tool for managing Python packages using pip."""

    @property
    def name(self) -> str:
        return "package_manager"

    @property
    def description(self) -> str:
        return "Manages Python packages using pip"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Package management action to perform",
                    "enum": ["install", "uninstall", "list", "check", "info", "upgrade", "freeze", "config"]
                },
                "package": {
                    "type": "string",
                    "description": "Package name with optional version specification"
                },
                "requirements_file": {
                    "type": "string",
                    "description": "Path to requirements.txt file"
                }
            },
            "required": ["action"]
        }

    def __init__(self, options: Optional[PackageManagerOptions] = None):
        self.options = options or {}
        self.pip_cmd = self._get_pip_cmd()

    def _get_pip_cmd(self) -> str:
        """Get the appropriate pip command based on environment."""
        if Config.PACKAGE_MANAGER_CONFIG["pip_command"]:
            return Config.PACKAGE_MANAGER_CONFIG["pip_command"]
        
        venv = self.options.get('virtual_env') or os.environ.get('VIRTUAL_ENV')
        if Config.PACKAGE_MANAGER_CONFIG["use_module_pip"]:
            return sys.executable  # Will be used with -m pip
        
        if venv:
            if os.name == 'nt':  # Windows
                return os.path.join(venv, 'Scripts', 'pip.exe')
            return os.path.join(venv, 'bin', 'pip')
        return 'pip'

    def _run_pip(self, *args: str) -> Dict[str, Any]:
        """Run pip command and return result."""
        try:
            if Config.PACKAGE_MANAGER_CONFIG["use_module_pip"]:
                cmd = [sys.executable, "-m", "pip"] + list(args)
            else:
                cmd = [self.pip_cmd] + list(args)
            
            if index_url := self.options.get('index_url'):
                cmd.extend(['--index-url', index_url])
            if extra_index := self.options.get('extra_index_url'):
                cmd.extend(['--extra-index-url', extra_index])
            if trusted_host := self.options.get('trusted_host'):
                cmd.extend(['--trusted-host', trusted_host])
            if self.options.get('require_virtualenv'):
                cmd.append('--require-virtualenv')
            if self.options.get('no_deps'):
                cmd.append('--no-deps')
            if self.options.get('pre'):
                cmd.append('--pre')

            result = self._try_run_command(cmd)
            if "Error" not in result["content"]:
                return result
            
            # If failed, try python -m pip
            cmd = [sys.executable, "-m", "pip"] + list(args)
            return self._try_run_command(cmd)
        except Exception as e:
            return {"content": f"Error: Error running pip: {str(e)}"}

    def _try_run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Try running a command and return result."""
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return {"content": result.stdout}

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute package management action."""
        action = input.get("action", "install")
        package = input.get("package", "")
        requirements_file = input.get("requirements_file")

        if action == "install":
            if requirements_file:
                return self._run_pip('install', '-r', requirements_file)
            elif package:
                return self._run_pip('install', package)
            else:
                return {"content": "Error: Either package or requirements_file must be specified for install"}
        
        elif action == "uninstall":
            if not package:
                return {"content": "Error: Package name is required for uninstall"}
            return self._run_pip('uninstall', '-y', package)
        
        elif action == "list":
            return self._run_pip('list')
        
        elif action == "check":
            if not package:
                return {"content": "Error: Package name is required for check"}
            return self._run_pip('show', package)
        
        elif action == "info":
            if not package:
                return {"content": "Error: Package name is required for info"}
            return self._run_pip('show', package)
        
        elif action == "upgrade":
            if not package:
                return {"content": "Error: Package name is required for upgrade"}
            return self._run_pip('install', '--upgrade', package)
        
        elif action == "freeze":
            return self._run_pip('freeze')
        
        elif action == "config":
            return self._run_pip('-V')
        
        else:
            return {"content": f"Error: Unsupported action: {action}"}

    def install(self, package_spec: str) -> Dict[str, Any]:
        """Install a package."""
        return self._run_pip('install', package_spec)

    def uninstall(self, package_name: str) -> Dict[str, Any]:
        """Uninstall a package."""
        return self._run_pip('uninstall', '-y', package_name)

    def list(self) -> Dict[str, Any]:
        """List installed packages."""
        return self._run_pip('list')

    def check(self, package_name: str) -> Dict[str, Any]:
        """Check if a package is installed."""
        return self._run_pip('show', package_name)

    def info(self, package_name: str) -> Dict[str, Any]:
        """Get package information."""
        return self._run_pip('show', package_name)

    def install_requirements(self, requirements_file: str) -> Dict[str, Any]:
        """Install packages from requirements file."""
        return self._run_pip('install', '-r', requirements_file)

    def upgrade(self, package_name: str) -> Dict[str, Any]:
        """Upgrade a package to latest version."""
        return self._run_pip('install', '--upgrade', package_name)

    def freeze(self) -> Dict[str, Any]:
        """Export installed packages as requirements."""
        return self._run_pip('freeze')

    def check_outdated(self) -> Dict[str, Any]:
        """Check for outdated packages."""
        return self._run_pip('list', '--outdated')

    def config(self) -> Dict[str, Any]:
        """Get pip configuration and location."""
        return self._run_pip('-V')

    def cache_info(self) -> Dict[str, Any]:
        """Get pip cache information."""
        return self._run_pip('cache', 'info')

    def cache_clear(self) -> Dict[str, Any]:
        """Clear pip cache."""
        return self._run_pip('cache', 'purge')

    def wheel_info(self, wheel_file: str) -> Dict[str, Any]:
        """Get information about a wheel file."""
        return self._run_pip('show', '-f', wheel_file)
