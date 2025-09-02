import json
import types

from src.infrastructure.llm.openai_compatible import OpenAICompatibleWrapper
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
            "properties": {"x": {"type": "number", "description": "x value"}},
            "required": ["x"],
        }

    def run(self, input: dict) -> dict:
        self.last_input = input
        return {"type": "tool_result", "content": "ok"}


def _make_openai_message_with_tool_calls(name: str, args: dict):
    class Fn:
        def __init__(self):
            self.name = name
            self.arguments = json.dumps(args)

    class ToolCall:
        def __init__(self):
            self.function = Fn()

    class Msg:
        def __init__(self):
            self.tool_calls = [ToolCall()]
            self.content = ""  # content irrelevant when tool_calls present
            self.reasoning_content = None

    class Choice:
        def __init__(self):
            self.message = Msg()

    class Resp:
        def __init__(self):
            self.choices = [Choice()]

    return Resp()


def _make_openai_message_with_text(content: str):
    class Msg:
        def __init__(self):
            self.tool_calls = None
            self.content = content
            self.reasoning_content = None

    class Choice:
        def __init__(self):
            self.message = Msg()

    class Resp:
        def __init__(self):
            self.choices = [Choice()]

    return Resp()


def test_adapter_uses_native_tool_calls_when_available(monkeypatch):
    wrapper = OpenAICompatibleWrapper(api_key="x", base_url="https://api.openai.com/v1", model="gpt-test")
    tool = DummyTool()
    wrapper.register_tool(tool)

    def fake_create(**kwargs):
        return _make_openai_message_with_tool_calls("dummy", {"x": 7})

    wrapper.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=fake_create)
        )
    )

    out = wrapper.execute("do it")
    assert tool.last_input == {"x": 7}, "Adapter must convert native tool_calls to internal shape"
    assert "Tool Call:" in out and '"tool": "dummy"' in out, "Output must include tool call JSON"


def test_adapter_falls_back_to_text_parser_when_no_tool_calls(monkeypatch):
    wrapper = OpenAICompatibleWrapper(api_key="x", base_url="https://api.openai.com/v1", model="gpt-test")
    tool = DummyTool()
    wrapper.register_tool(tool)

    def fake_create(**kwargs):
        return _make_openai_message_with_text('Reasoning...\nTOOL_CALL: {"tool":"dummy","input_schema":{"x":1}}')

    wrapper.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=fake_create)
        )
    )

    out = wrapper.execute("do it")
    assert tool.last_input == {"x": 1}, "Fallback parser must extract TOOL_CALL JSON when no native tool_calls"
    assert "Tool Call:" in out and '"tool": "dummy"' in out