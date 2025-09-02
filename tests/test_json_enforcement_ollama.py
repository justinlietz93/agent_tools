import json
import types

import requests

from src.infrastructure.llm.ollama_wrapper import OllamaWrapper
from src.infrastructure.tools.tool_base import Tool


class DummyTool(Tool):
    def __init__(self):
        self.last_input = None

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "Dummy tool"

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"foo": {"type": "number", "description": "foo"}},
            "required": ["foo"],
        }

    def run(self, input: dict) -> dict:
        self.last_input = input
        return {"type": "tool_result", "content": "ok"}


def test_ollama_sets_format_json_and_no_openai_only_fields(monkeypatch):
    wrapper = OllamaWrapper(api_key="ollama", base_url="http://localhost:11434/v1", model="llama3.1")
    tool = DummyTool()
    wrapper.register_tool(tool)

    captured = {"url": None, "headers": None, "payload": None}

    class FakeResp:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    def fake_post(url, headers=None, data=None, **kwargs):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json.loads(data or "{}")
        # Return minimal valid /api/chat body with no tool_calls so wrapper falls back to content
        body = {"message": {"content": "No tool call made."}}
        return FakeResp(json.dumps(body))

    monkeypatch.setattr(requests, "post", fake_post)

    wrapper.execute("hi")

    assert captured["url"].endswith("/api/chat")
    payload = captured["payload"]
    assert payload["model"] == wrapper.model
    assert payload["messages"][0]["role"] == "system"
    assert isinstance(payload.get("tools"), list) and payload["tools"], "Ollama tools must be present"
    assert payload.get("format") == "json", "Ollama must set format=json when tools are present"
    # Ensure OpenAI-only fields are not present on Ollama /api/chat payload
    assert "response_format" not in payload
    assert "tool_choice" not in payload