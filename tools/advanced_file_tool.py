# selfprompter/tools/advanced_file_tool.py

import os
import shutil
from typing import Dict, Any, Optional, List, Union
from .tool_base import Tool, ToolResult

class AdvancedFileTool(Tool):
    """
    Advanced file and directory operations tool.
    Follows Anthropic Claude tool use standards.
    """

    def __init__(self, repo_root: str = "./"):
        """
        repo_root is the top-level directory in which files can be manipulated.
        Default is current directory.
        """
        self.repo_root = os.path.abspath(repo_root)
        os.makedirs(self.repo_root, exist_ok=True)

    @property
    def name(self) -> str:
        return "advanced_file_operations"

    @property
    def description(self) -> str:
        return (
            "Advanced file and directory management tool that can read, write, edit files "
            "and manage directories. Supports partial file reads, in-place edits, directory "
            "creation/deletion, and file moving/renaming operations. All operations are "
            "restricted to a safe root directory."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "read_chunk", "write", "append", "edit", "mkdir", "rmdir", "delete_file", "move"],
                    "description": "The operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "Target file or directory path"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write/append/edit operations)"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Starting line number for chunk read or edit operations"
                },
                "end_line": {
                    "type": "integer",
                    "description": "Ending line number for edit operations"
                },
                "num_lines": {
                    "type": "integer",
                    "description": "Number of lines to read for chunk operations"
                },
                "src": {
                    "type": "string",
                    "description": "Source path for move operations"
                },
                "dest": {
                    "type": "string",
                    "description": "Destination path for move operations"
                }
            },
            "required": ["operation"],
            "additionalProperties": False
        }

    def _safe_path(self, user_path: str) -> str:
        """Ensure path is within repo_root directory."""
        normalized = os.path.normpath(user_path)
        full_path = os.path.join(self.repo_root, normalized)
        if not full_path.startswith(self.repo_root):
            raise ValueError("Path is outside the repository root")
        return full_path

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the requested file or directory operation.
        
        Args:
            input: Dictionary containing operation-specific input_schema:
                operation: The operation to perform
                path: Target file or directory path
                content: Content to write (for write/append/edit operations)
                start_line: Starting line number for chunk read or edit operations
                end_line: Ending line number for edit operations
                num_lines: Number of lines to read for chunk operations
                src: Source path for move operations
                dest: Destination path for move operations
            
        Returns:
            Dictionary containing operation result or error message
        """
        try:
            operation = input.get("operation", "").lower()
            path = input.get("path", "")
            content = input.get("content", "")
            start_line = input.get("start_line")
            end_line = input.get("end_line")
            num_lines = input.get("num_lines")
            src = input.get("src")
            dest = input.get("dest")

            if operation in ["read", "read_chunk", "write", "append", "edit"]:
                result = self._handle_file_operations(
                    operation=operation,
                    path=path,
                    content=content,
                    start_line=start_line,
                    end_line=end_line,
                    num_lines=num_lines
                )
            elif operation == "mkdir":
                result = self._mkdir(path)
            elif operation == "rmdir":
                result = self._rmdir(path)
            elif operation == "delete_file":
                result = self._delete_file(path)
            elif operation == "move":
                if not src or not dest:
                    return {
                        "type": "tool_response",
                        "content": "Error: 'move' operation requires both 'src' and 'dest' input_schema"
                    }
                result = self._move(src, dest)
            else:
                return {
                    "type": "tool_response",
                    "content": f"Error: Unknown operation '{operation}'"
                }

            return {
                "type": "tool_response",
                "content": result
            }

        except Exception as e:
            return {
                "type": "tool_response",
                "content": f"Error: {str(e)}"
            }

    def _handle_file_operations(
        self,
        operation: str,
        path: str,
        content: str = "",
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        num_lines: Optional[int] = None
    ) -> str:
        filepath = self._safe_path(path)

        if operation == "read":
            return self._read_file(filepath)
        elif operation == "read_chunk":
            if not start_line:
                raise ValueError("'read_chunk' requires 'start_line'")
            return self._read_chunk(filepath, start_line, num_lines or 50)
        elif operation == "write":
            return self._write_file(filepath, content)
        elif operation == "append":
            return self._append_file(filepath, content)
        elif operation == "edit":
            if not start_line:
                raise ValueError("'edit' requires 'start_line'")
            return self._edit_file(filepath, start_line, end_line or start_line, content)
        else:
            raise ValueError(f"Unexpected operation '{operation}'")

    def _read_file(self, filepath: str) -> str:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File '{filepath}' not found")
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def _read_chunk(self, filepath: str, start_line: int, num_lines: int) -> str:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File '{filepath}' not found")
        
        lines_out = []
        with open(filepath, "r", encoding="utf-8") as f:
            all_lines = f.readlines()

        actual_start = max(0, start_line - 1)
        actual_end = min(len(all_lines), actual_start + num_lines)
        chunk = all_lines[actual_start:actual_end]
        
        for i, line in enumerate(chunk, start=actual_start+1):
            lines_out.append(f"{i}: {line.rstrip()}")
        return "\n".join(lines_out)

    def _write_file(self, filepath: str, content: str) -> str:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to '{filepath}'"

    def _append_file(self, filepath: str, content: str) -> str:
        with open(filepath, "a", encoding="utf-8") as f:
            if content:
                f.write("\n" + content)
        return f"Successfully appended to '{filepath}'"

    def _edit_file(self, filepath: str, start_line: int, end_line: int, new_content: str) -> str:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File '{filepath}' not found")

        with open(filepath, "r", encoding="utf-8") as f:
            original = f.readlines()

        start_idx = max(0, start_line - 1)
        end_idx = min(len(original), end_line)
        
        edited = (
            original[:start_idx] +
            [line + "\n" for line in new_content.split("\n")] +
            original[end_idx:]
        )
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(edited)

        return f"Successfully edited lines {start_line} through {end_line} in '{filepath}'"

    def _mkdir(self, path: str) -> str:
        dirpath = self._safe_path(path)
        os.makedirs(dirpath, exist_ok=True)
        return f"Successfully created directory '{dirpath}'"

    def _rmdir(self, path: str) -> str:
        dirpath = self._safe_path(path)
        if not os.path.isdir(dirpath):
            raise NotADirectoryError(f"'{dirpath}' is not a directory or does not exist")
        shutil.rmtree(dirpath)
        return f"Successfully removed directory '{dirpath}'"

    def _delete_file(self, path: str) -> str:
        filepath = self._safe_path(path)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File '{filepath}' not found")
        os.remove(filepath)
        return f"Successfully deleted '{filepath}'"

    def _move(self, src: str, dest: str) -> str:
        src_path = self._safe_path(src)
        dest_path = self._safe_path(dest)
        
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source path '{src_path}' does not exist")
            
        shutil.move(src_path, dest_path)
        return f"Successfully moved '{src_path}' to '{dest_path}'"