from pathlib import Path

from src.tools.registry import ToolContext, execute_tool


def test_tool_invalid_arguments_return_structured_error():
    result = execute_tool("calculator", {"input": ""}, ToolContext("sess", None, Path("outputs")))

    assert result.error
    assert "invalid arguments" in result.error


def test_write_file_tool_writes_inside_workspace():
    result = execute_tool(
        "write_file",
        {"path": "outputs/protocol_test.md", "content": "hello"},
        ToolContext("sess", None, Path("outputs")),
    )

    assert result.error is None
    assert Path("outputs/protocol_test.md").read_text(encoding="utf-8") == "hello"
