# selfprompter/tools/web_browser_tool.py

import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional
from .tool_base import Tool, ToolResult

class WebBrowserTool(Tool):
    """
    Tool for fetching and parsing web page content.
    Follows Anthropic Claude tool use standards.
    """

    def __init__(self, user_agent: Optional[str] = None):
        """
        Initialize with optional custom user agent.
        
        Args:
            user_agent: Custom User-Agent string for requests
        """
        self.user_agent = user_agent or "AnthropicClaudeTool/1.0"
        self.headers = {"User-Agent": self.user_agent}

    @property
    def name(self) -> str:
        return "web_browser"

    @property
    def description(self) -> str:
        return (
            "Fetches and parses web page content. Can extract text content, follow links, "
            "and handle different content types. Returns cleaned and formatted page content."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from"
                },
                "extract_type": {
                    "type": "string",
                    "description": "Type of content to extract",
                    "enum": ["text", "links", "title"],
                    "default": "text"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds",
                    "minimum": 1,
                    "maximum": 30,
                    "default": 10
                }
            },
            "required": ["url"],
            "additionalProperties": False
        }

    def run(self, tool_call_id: str, **kwargs) -> ToolResult:
        """
        Fetch and parse web page content.
        
        Args:
            tool_call_id: Unique ID for this tool call
            url: The URL to fetch
            extract_type: Type of content to extract (default: "text")
            timeout: Request timeout in seconds (default: 10)
            
        Returns:
            ToolResult containing parsed content or error message
        """
        try:
            url = kwargs.get("url")
            if not url:
                raise ValueError("URL is required")

            extract_type = kwargs.get("extract_type", "text").lower()
            timeout = min(max(1, kwargs.get("timeout", 10)), 30)

            response = requests.get(url, headers=self.headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            if extract_type == "text":
                content = self._extract_text(soup)
            elif extract_type == "links":
                content = self._extract_links(soup)
            elif extract_type == "title":
                content = self._extract_title(soup)
            else:
                raise ValueError(f"Invalid extract_type: {extract_type}")

            return self.format_result(tool_call_id, content)

        except Exception as e:
            return self.format_error(tool_call_id, str(e))

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract main text content from page."""
        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.decompose()

        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text

    def _extract_links(self, soup: BeautifulSoup) -> str:
        """Extract all links from page."""
        links = []
        for link in soup.find_all("a"):
            href = link.get("href")
            text = link.get_text().strip()
            if href and text:
                links.append(f"{text}: {href}")

        return "\n".join(links) if links else "No links found"

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title = soup.find("title")
        return title.get_text() if title else "No title found"
