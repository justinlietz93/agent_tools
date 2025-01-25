# selfprompter/tools/doc_check_tool.py

"""
Documentation checking tool that validates markdown files and monitors external documentation sites.

This tool provides functionality to:
1. Check local documentation files for formatting, completeness, and broken links
2. Monitor external documentation sites for updates
3. Validate required sections and structure
4. Ensure consistent formatting

Example usage:
    # Check local documentation
    result = doc_check_tool.run(
        tool_call_id="check_docs",
        path="docs/api.md",
        check_type="all"
    )

    # Monitor external sites
    result = doc_check_tool.run(
        tool_call_id="monitor_sites",
        check_type="sites"
    )
"""

import os
import re
import requests
from typing import Dict, Any, List, Optional, Union
from .tool_base import Tool, ToolResult

class DocCheckTool(Tool):
    """
    Tool for checking and validating documentation.
    Follows Anthropic Claude tool use standards.
    """

    def __init__(self, docs_root: str = "./docs", default_sites: List[str] = None):
        """
        Initialize with docs directory root and optional default sites.
        
        Args:
            docs_root: Root directory containing documentation
            default_sites: List of default documentation sites to check
        """
        self.docs_root = os.path.abspath(docs_root)
        os.makedirs(self.docs_root, exist_ok=True)
        self.default_sites = default_sites or [
            "https://docs.anthropic.com/",
            "https://api-docs.deepseek.com/",
            "https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-pro",
            "https://platform.openai.com/docs/api-reference/introduction",
            "https://google.com",
            "https://github.com/justinlietz93?tab=repositories"
        ]

    @property
    def name(self) -> str:
        return "documentation_check"

    @property
    def description(self) -> str:
        return (
            "Validates documentation files and checks external documentation sites for updates. "
            "Can analyze local files for completeness, broken links, and formatting issues, "
            "as well as monitor external documentation sites for changes."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to documentation file or directory to check"
                },
                "check_type": {
                    "type": "string",
                    "description": "Type of check to perform",
                    "enum": ["completeness", "links", "formatting", "sites", "all"],
                    "default": "all"
                },
                "required_sections": {
                    "type": "array",
                    "description": "List of required section names",
                    "items": {
                        "type": "string"
                    }
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to check subdirectories recursively",
                    "default": False
                },
                "sites": {
                    "type": "array",
                    "description": "List of documentation sites to check",
                    "items": {
                        "type": "string"
                    }
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return per site",
                    "default": 2000
                }
            },
            "required": ["check_type"],
            "additionalProperties": False
        }

    @property
    def input_schema(self) -> Dict[str, Any]:
        """Get the input schema for the tool."""
        return self.input_schema

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check documentation files and sites.
        
        Args:
            input: Dictionary containing:
                check_type: Type of check to perform
                path: Path to documentation file or directory (for local checks)
                required_sections: List of required section names
                recursive: Whether to check subdirectories
                sites: List of documentation sites to check
                max_chars: Maximum characters to return per site
            
        Returns:
            Dictionary containing check results or error message
        """
        try:
            check_type = input.get("check_type", "all").lower()
            
            if check_type in ["sites", "all"]:
                sites_results = self._check_sites(
                    input.get("sites", self.default_sites),
                    input.get("max_chars", 2000)
                )
                if check_type == "sites":
                    return self.format_result(input.get("tool_use_id", ""), sites_results)

            if check_type != "sites":
                path = input.get("path")
                if not path:
                    return self.format_error(input.get("tool_use_id", ""), "Path is required for local documentation checks")

                full_path = os.path.join(self.docs_root, path)
                if not os.path.exists(full_path):
                    return self.format_error(input.get("tool_use_id", ""), f"Path '{path}' not found")

                required_sections = input.get("required_sections", [])
                recursive = input.get("recursive", False)

                if os.path.isfile(full_path):
                    results = self._check_file(full_path, check_type, required_sections)
                else:
                    results = self._check_directory(full_path, check_type, required_sections, recursive)

                if check_type == "all":
                    results = f"{sites_results}\n\nLocal Documentation Results:\n{self._format_results(results)}"
                else:
                    results = self._format_results(results)

                return self.format_result(input.get("tool_use_id", ""), results)

        except Exception as e:
            return self.format_error(input.get("tool_use_id", ""), str(e))

    def _check_sites(self, sites: List[str], max_chars: int) -> str:
        """Check documentation sites for updates."""
        results = []
        for url in sites:
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    results.append(f"{url} -> ERROR status {resp.status_code}")
                    continue
                    
                text = resp.text
                if len(text) > max_chars:
                    text = text[:max_chars] + " ...[truncated]"
                results.append(f"{url}\n{text}")

            except requests.RequestException as e:
                results.append(f"{url} -> ERROR: {str(e)}")

        return "External Documentation Sites Results:\n" + "\n\n".join(results)

    def _check_file(self, filepath: str, check_type: str, required_sections: List[str]) -> Dict[str, Any]:
        """Check a single documentation file."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        issues = []
        
        if check_type in ["completeness", "all"]:
            missing_sections = self._check_required_sections(content, required_sections)
            if missing_sections:
                issues.append(f"Missing required sections: {', '.join(missing_sections)}")

        if check_type in ["links", "all"]:
            broken_links = self._check_links(content)
            if broken_links:
                issues.append(f"Broken links found: {', '.join(broken_links)}")

        if check_type in ["formatting", "all"]:
            formatting_issues = self._check_formatting(content)
            if formatting_issues:
                issues.extend(formatting_issues)

        return {
            "file": filepath,
            "issues": issues,
            "status": "pass" if not issues else "fail"
        }

    def _check_directory(self, dirpath: str, check_type: str, required_sections: List[str], recursive: bool) -> List[Dict[str, Any]]:
        """Check all documentation files in a directory."""
        results = []
        
        for root, _, files in os.walk(dirpath):
            if not recursive and root != dirpath:
                continue
                
            for file in files:
                if file.endswith((".md", ".rst", ".txt")):
                    filepath = os.path.join(root, file)
                    results.append(self._check_file(filepath, check_type, required_sections))

        return results

    def _check_required_sections(self, content: str, required_sections: List[str]) -> List[str]:
        """Check for required sections in content."""
        missing = []
        for section in required_sections:
            pattern = rf"#+ {section}"
            if not re.search(pattern, content, re.IGNORECASE):
                missing.append(section)
        return missing

    def _check_links(self, content: str) -> List[str]:
        """Check for broken links in content."""
        broken = []
        link_pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
        
        for match in re.finditer(link_pattern, content):
            link = match.group(2)
            if link.startswith(("http://", "https://")):
                continue  # Skip external links
            
            if not os.path.exists(os.path.join(self.docs_root, link)):
                broken.append(link)
                
        return broken

    def _check_formatting(self, content: str) -> List[str]:
        """Check for common formatting issues."""
        issues = []
        
        # Check for consistent header formatting
        if re.search(r"#[^#\s]", content):
            issues.append("Incorrect header formatting (missing space after #)")
            
        # Check for trailing whitespace
        if re.search(r"[ \t]+$", content, re.MULTILINE):
            issues.append("Lines with trailing whitespace found")
            
        # Check for multiple consecutive blank lines
        if re.search(r"\n{3,}", content):
            issues.append("Multiple consecutive blank lines found")
            
        return issues

    def _format_results(self, results: Union[Dict[str, Any], List[Dict[str, Any]]]) -> str:
        """Format check results into readable string."""
        if isinstance(results, dict):
            results = [results]
            
        output = []
        for result in results:
            output.append(f"File: {result['file']}")
            output.append(f"Status: {result['status']}")
            if result['issues']:
                output.append("Issues:")
                for issue in result['issues']:
                    output.append(f"  - {issue}")
            output.append("")
            
        return "\n".join(output)
