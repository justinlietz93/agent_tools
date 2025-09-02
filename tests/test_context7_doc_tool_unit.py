import pytest

from src.infrastructure.tools.mcp_tools.doc_check_tool import DocCheckTool


class FakeContext7Client:
    def __init__(self, healthy: bool = True, base_url: str = "http://localhost:8080"):
        self._healthy = healthy
        self.base_url = base_url
        self.calls = []

    def health_check(self, timeout: float = 2.0) -> bool:
        self.calls.append(("health_check", {"timeout": timeout}))
        return self._healthy

    def resolve_library_id(self, library_name: str) -> str:
        self.calls.append(("resolve_library_id", {"library_name": library_name}))
        name = (library_name or "").strip()
        if not name:
            raise ValueError("empty library name")
        # Simulate ID resolution: pass through IDs, prefix simple names
        return name if name.startswith("/") else f"/mock/{name}"

    def get_docs(self, library_id: str, topic=None, tokens: int = 4000):
        self.calls.append(("get_docs", {"library_id": library_id, "topic": topic, "tokens": tokens}))
        if not library_id:
            return {"status": "error", "error": "missing library_id", "base_url": self.base_url}
        # Return simple text containing headings so completeness can find them
        return {
            "status": "ok",
            "text": "# Title\n\n## Introduction\n\nSome docs...\n\n## API Reference\n\nMore docs...",
            "base_url": self.base_url,
        }


def test_health_check_false_returns_error():
    client = FakeContext7Client(healthy=False)
    tool = DocCheckTool(context7_client=client)

    res = tool.run({
        "check_type": "completeness",
        "library": "react",
        "required_sections": ["Introduction"],
    })

    assert isinstance(res, dict)
    assert res["check_type"] == "completeness"
    assert res["status"] == "error"
    assert "not reachable" in res["summary"].lower() or "unavailable" in res["summary"].lower()
    assert isinstance(res["data"], dict)
    assert res["data"].get("reason") == "unavailable"
    assert isinstance(res["base_url"], str)


def test_completeness_success_and_resolution():
    client = FakeContext7Client(healthy=True)
    tool = DocCheckTool(context7_client=client)

    res = tool.run({
        "check_type": "completeness",
        "library": "react",
        "required_sections": ["Introduction", "API Reference"],
        "tokens": 2000
    })

    assert isinstance(res, dict)
    assert res["check_type"] == "completeness"
    assert res["status"] == "success"
    assert isinstance(res["summary"], str)
    assert isinstance(res["data"], dict)
    assert res["base_url"] == client.base_url

    # Verify call order and param forwarding
    names = [c[0] for c in client.calls]
    assert "resolve_library_id" in names
    assert "get_docs" in names
    last_call = client.calls[-1]
    assert last_call[0] == "get_docs"
    assert last_call[1]["tokens"] == 2000


def test_sites_success_multiple():
    client = FakeContext7Client(healthy=True)
    tool = DocCheckTool(context7_client=client)

    res = tool.run({
        "check_type": "sites",
        "libraries": ["react", "vue", "svelte"],
        "tokens": 1000
    })

    assert isinstance(res, dict)
    assert res["check_type"] == "sites"
    assert res["status"] == "success"
    assert isinstance(res["summary"], str)
    assert isinstance(res["data"], dict)
    results = res["data"]["results"]
    assert len(results) == 3
    assert all(item.get("status") in {"ok", "error", "unavailable", "unsupported"} for item in results)


def test_unsupported_checks_return_error():
    client = FakeContext7Client(healthy=True)
    tool = DocCheckTool(context7_client=client)

    for ct in ("links", "anchors", "frontmatter"):
        res = tool.run({"check_type": ct, "library": "react"})
        assert res["status"] == "error"
        assert "not supported" in res["summary"].lower()


def test_unknown_check_type_returns_error():
    client = FakeContext7Client(healthy=True)
    tool = DocCheckTool(context7_client=client)

    res = tool.run({"check_type": "unknown", "library": "react"})
    assert res["status"] == "error"
    assert "check_type" in res["summary"].lower()