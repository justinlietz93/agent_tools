# selfprompter/tools/requests_tool.py

import requests
from typing import Dict, Any, Optional
from .tool_base import Tool, ToolResult

class RequestsTool(Tool):
    """
    Tool for making HTTP requests.
    Follows Anthropic Claude tool use standards.
    """

    def __init__(self, default_headers: Optional[Dict[str, str]] = None):
        """
        Initialize with optional default headers.
        
        Args:
            default_headers: Default headers to include in all requests
        """
        self.default_headers = default_headers or {}

    @property
    def name(self) -> str:
        return "http_request"

    @property
    def description(self) -> str:
        return (
            "Makes HTTP requests to specified URLs. Supports GET, POST, PUT, DELETE methods. "
            "Can send custom headers and data. Returns response content and status."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to send the request to"
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method to use",
                    "enum": ["GET", "POST", "PUT", "DELETE"],
                    "default": "GET"
                },
                "headers": {
                    "type": "object",
                    "description": "Request headers",
                    "additionalProperties": {
                        "type": "string"
                    }
                },
                "data": {
                    "type": "string",
                    "description": "Request body data"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds",
                    "minimum": 1,
                    "maximum": 60,
                    "default": 30
                }
            },
            "required": ["url"],
            "additionalProperties": False
        }

    def run(self, tool_call_id: str, **kwargs) -> ToolResult:
        """
        Execute an HTTP request.
        
        Args:
            tool_call_id: Unique ID for this tool call
            url: The URL to send the request to
            method: HTTP method (default: "GET")
            headers: Request headers
            data: Request body data
            timeout: Request timeout in seconds (default: 30)
            
        Returns:
            ToolResult containing response data or error message
        """
        try:
            url = kwargs.get("url")
            if not url:
                raise ValueError("URL is required")

            method = kwargs.get("method", "GET").upper()
            if method not in ["GET", "POST", "PUT", "DELETE"]:
                raise ValueError(f"Unsupported HTTP method: {method}")

            headers = {**self.default_headers, **(kwargs.get("headers") or {})}
            data = kwargs.get("data")
            timeout = min(max(1, kwargs.get("timeout", 30)), 60)

            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=data,
                timeout=timeout
            )

            response.raise_for_status()
            
            result = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text
            }

            return self.format_result(tool_call_id, str(result))

        except requests.RequestException as e:
            return self.format_error(tool_call_id, f"Request failed: {str(e)}")
        except Exception as e:
            return self.format_error(tool_call_id, str(e))
