from pathlib import Path

from src.tools.calculator import calculator
from src.tools.notes_tool import NOTES_FILE, notes_tool


def test_calculator_handles_basic_math():
    assert calculator("12 * (8 + 5)") == "12 * (8 + 5) = 156"


def test_calculator_rejects_unsafe_code():
    result = calculator("__import__('os').system('dir')")
    assert result.startswith("计算失败")


def test_notes_tool_writes_to_outputs_directory():
    result = notes_tool("hello workflow agent")
    assert "笔记已保存" in result
    assert Path(NOTES_FILE).read_text(encoding="utf-8") == "hello workflow agent\n"
