import json
import unittest
from types import SimpleNamespace

from devflow.planning.serena import (
    READ_ONLY_SERENA_TOOLS,
    _langchain_tools,
    _tool_result_text,
)


class SerenaToolFilteringTests(unittest.TestCase):
    def test_exposes_only_allowlisted_read_tools(self):
        tools = [
            SimpleNamespace(
                name="find_symbol",
                description="Find symbols",
                inputSchema={"type": "object"},
            ),
            SimpleNamespace(
                name="execute_shell_command",
                description="Execute shell",
                inputSchema={"type": "object"},
            ),
            SimpleNamespace(
                name="replace_symbol_body",
                description="Edit symbol",
                inputSchema={"type": "object"},
            ),
        ]

        converted = _langchain_tools(tools)
        names = [tool["function"]["name"] for tool in converted]

        self.assertEqual(names, ["find_symbol"])
        self.assertNotIn("execute_shell_command", READ_ONLY_SERENA_TOOLS)
        self.assertNotIn("replace_symbol_body", READ_ONLY_SERENA_TOOLS)


class SerenaResultFormattingTests(unittest.TestCase):
    def test_prefers_structured_content_and_truncates(self):
        result = SimpleNamespace(
            structuredContent={"symbols": ["one", "two"]},
            content=[],
        )

        text = _tool_result_text(result, 20)

        self.assertEqual(text, json.dumps(
            {"symbols": ["one", "two"]},
            ensure_ascii=False,
        )[:20])

    def test_joins_text_blocks(self):
        result = SimpleNamespace(
            structuredContent=None,
            content=[SimpleNamespace(text="first"), SimpleNamespace(text="second")],
        )

        self.assertEqual(_tool_result_text(result, 100), "first\nsecond")
