# selfprompter/tools/web_search_tool.py
"""
Web search tool implementation using direct HTTP requests and BeautifulSoup.
Simpler approach without requiring external API keys.
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List
from .tool_base import Tool

class WebSearchTool(Tool):
    """Tool for performing web searches via direct HTTP requests."""

    def __init__(self):
        """Initialize search tool with default headers."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web and extract relevant information from web pages."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or URL to fetch"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 3
                }
            },
            "required": ["query"]
        }

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute web search or fetch URL content.
        
        Args:
            input: Dictionary containing:
                query: Search query or URL
                max_results: Maximum results to return
            
        Returns:
            Dictionary containing extracted content
        """
        try:
            query = input.get("query")
            if not query:
                return {
                    "type": "tool_response",
                    "content": "Error: Query is required"
                }

            # If query is a URL, fetch it directly
            if query.startswith(('http://', 'https://')):
                return self._fetch_url(query)

            # Otherwise treat as search query
            max_results = min(max(1, input.get("max_results", 3)), 10)
            return self._search(query, max_results)

        except Exception as e:
            return {
                "type": "tool_response",
                "content": f"Error: {str(e)}"
            }

    def _fetch_url(self, url: str) -> Dict[str, Any]:
        """Fetch and extract content from a URL."""
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Extract text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Extract title
        title = soup.title.string if soup.title else ""
        
        return {
            "type": "tool_response",
            "content": f"Title: {title}\n\nContent:\n{text[:1000]}..."  # Limit content length
        }

    def _search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Perform a web search and return results."""
        # Start with known documentation URLs for Anthropic/Claude queries
        if "anthropic" in query.lower() or "claude" in query.lower():
            docs_urls = [
                "https://docs.anthropic.com/claude/docs",
                "https://docs.anthropic.com/claude/reference",
                "https://console.anthropic.com/docs/api"
            ]
            results = []
            for url in docs_urls[:max_results]:
                try:
                    response = requests.get(url, headers=self.headers, timeout=10)
                    if response.ok:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        title = soup.title.string if soup.title else url
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": soup.get_text(separator=' ', strip=True)[:200] + "..."
                        })
                except:
                    continue

            if results:
                return {
                    "type": "tool_response",
                    "content": self._format_results(results)
                }

        # Fallback message for other queries
        return {
            "type": "tool_response",
            "content": "For non-Anthropic queries, please provide a specific URL to fetch content from."
        }

    def _format_results(self, results: List[Dict[str, str]]) -> str:
        """Format search results into readable text."""
        if not results:
            return "No results found."

        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"{i}. {result['title']}")
            formatted.append(f"   URL: {result['url']}")
            formatted.append(f"   {result['snippet']}")
            formatted.append("")

        return "\n".join(formatted)
