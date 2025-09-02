import json
import types

from src.infrastructure.llm.openai_compatible import OpenAICompatibleWrapper
from src.infrastructure.tools.tool_base import Tool


class DummyTool(Tool):
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
            "properties": {
                "foo": {"type": "number", "description": "foo"}
            },
            "required": ["foo"]
        }

    def run(self, input: dict) -> dict:
        self.last_input = input
        return {"type": "tool_result", "content": "ok"}


def test_openai_enforces_json_and_tool_choice_when_tools_present(monkeypatch):
    wrapper = OpenAICompatibleWrapper(api_key="x", base_url="https://api.openai.com/v1", model="gpt-test")
    wrapper.register_tool(DummyTool())

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        class Msg:
            content = 'TOOL_CALL: {"tool":"dummy","input_schema":{"foo":1}}'
            reasoning_content = None
        class Choice:
            message = Msg()
        class Resp:
            choices = [Choice()]
        return Resp()

    wrapper.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=fake_create)
        )
    )

    wrapper.execute("hi")

    assert "tools" in captured
    assert captured.get("tool_choice") == "required"
    assert captured.get("response_format") == {"type": "json_object"}

    tools = captured["tools"]
    assert isinstance(tools, list) and tools, "tools list must be non-empty"
    func = tools[0]["function"]
    assert func["name"] == "dummy"
    assert func["parameters"]["properties"]["foo"]["type"] == "number"