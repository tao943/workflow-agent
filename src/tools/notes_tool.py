from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
NOTES_FILE = OUTPUT_DIR / "notes.md"


def notes_tool(input: str) -> str:
    """Save notes inside the project outputs directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_FILE.write_text(input.strip() + "\n", encoding="utf-8")
    return f"笔记已保存：{NOTES_FILE}"
