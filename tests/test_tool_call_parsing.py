import pytest

from src.infrastructure.llm.openai_compatible import OpenAICompatibleWrapper


@pytest.fixture(scope="module")
def parser():
    # Instantiate with inert values; client isn't used for the parsing method
    return OpenAICompatibleWrapper(
        api_key="test",
        base_url="http://localhost:11434/v1",
        model="dummy",
    )


class TestToolCallParsing:
    def test_fenced_json_with_language_tag_json(self, parser):
        text = (
            "Thoughts...\n"
            "```json\n"
            '{"tool":"file","input_schema":{"operation":"write","path":"demo.txt","content":"ok"}}\n'
            "```\n"
            "done."
        )
        obj = parser._extract_tool_call(text)
        assert obj is not None, "Expected fenced json tool call to parse"
        assert obj.get("tool") == "file"
        assert obj.get("input_schema", {}).get("operation") == "write"

    def test_fenced_json_without_language_tag(self, parser):
        text = (
            "Here you go:\n"
            "```\n"
            '{"tool":"file","input_schema":{"operation":"write","path":"demo.txt","content":"ok"}}\n'
            "```\n"
        )
        obj = parser._extract_tool_call(text)
        assert obj is not None, "Expected fenced json (no language) to parse"
        assert obj.get("tool") == "file"
        assert obj.get("input_schema", {}).get("operation") == "write"

    @pytest.mark.parametrize(
        "text",
        [
            # Sentinel with direct JSON (upper snake)
            'TOOL_CALL: {"tool":"file","input_schema":{"operation":"write"}}',
            # Sentinel with direct JSON (space)
            'Tool Call: {"tool":"file","input_schema":{"operation":"write"}}',
            # Sentinel without space
            'ToolCall: {"tool":"file","input_schema":{"operation":"write"}}',
            # Sentinel with fenced JSON; parser step 2 handles the fence
            (
                "Tool Call: ```json\n"
                '{"tool":"file","input_schema":{"operation":"write","path":"demo.txt"}}\n'
                "```"
            ),
        ],
    )
    def test_sentinel_wrapped_json_positive(self, parser, text):
        obj = parser._extract_tool_call(text)
        assert obj is not None, "Expected sentinel-wrapped tool call to parse"
        assert obj.get("tool") == "file"
        assert obj.get("input_schema", {}).get("operation") == "write"

    @pytest.mark.parametrize(
        "text",
        [
            # Minor malformation: trailing comma inside object
            (
                "```json\n"
                '{"tool":"file","input_schema":{"operation":"write",}}\n'
                "```"
            ),
            # Minor malformation: single quotes instead of JSON double quotes
            (
                "```json\n"
                "{'tool':'file','input_schema':{'operation':'write'}}\n"
                "```"
            ),
        ],
    )
    def test_minor_malformations_expected_behavior(self, parser, text):
        # Current implementation does not normalize trailing commas or single quotes; expect graceful None
        obj = parser._extract_tool_call(text)
        assert obj is None

    def test_multiple_json_blocks_only_one_matches_tool_call(self, parser):
        text = (
            "Intro...\n"
            "```json\n"
            '{"foo":"bar"}\n'
            "```\n"
            "Middle text\n"
            "```json\n"
            '{"tool":"file","input_schema":{"operation":"write","path":"demo.txt"}}\n'
            "```\n"
            "End."
        )
        obj = parser._extract_tool_call(text)
        assert obj is not None, "Expected to select the valid tool-call block"
        assert obj.get("tool") == "file"
        assert obj.get("input_schema", {}).get("operation") == "write"

    def test_non_tool_json_present(self, parser):
        text = (
            "```json\n"
            '{"message":"hello","value":1}\n'
            "```\n"
        )
        obj = parser._extract_tool_call(text)
        assert obj is None, "JSON without required keys should return None"

    def test_plain_text_no_json(self, parser):
        text = "No JSON here. Just plain text and no tool call structure at all."
        obj = parser._extract_tool_call(text)
        assert obj is None