import pytest

from src.infrastructure.tools.mcp_tools.doc_check_tool import DocCheckTool
from src.infrastructure.mcp.context7_client import Context7MCPClient


@pytest.mark.integration
def test_context7_doc_completeness_integration():
    """
    Non-flaky integration test for the Context7-first DocCheckTool (library-centric).

    Behavior:
    - If Context7 MCP is not reachable, skip the test.
    - Run 'completeness' check against the Context7 MCP library docs.
    - Assert standardized output shape (success or error depending on provider support).
    """
    client = Context7MCPClient()
    if not client.health_check(timeout=2.0):
        pytest.skip("Context7 MCP not reachable")

    # Instantiate DocCheckTool with client DI
    tool = DocCheckTool(context7_client=client)

    # Execute a library-centric completeness check
    result = tool.run({
        "check_type": "completeness",
        "library": "/upstash/context7",  # Context7 MCP repository docs
        "required_sections": ["Context7"],  # heuristic section name
        "tokens": 2000
    })

    # Validate standardized output shape
    assert isinstance(result, dict)
    assert result.get("check_type") == "completeness"
    assert result.get("status") in {"success", "error"}  # provider decides success
    assert isinstance(result.get("summary"), str)
    assert isinstance(result.get("data"), dict)
    assert isinstance(result.get("base_url"), str)