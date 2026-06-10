from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_json_object(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    if raw.startswith("@"):
        path = Path(raw[1:]).expanduser()
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            return None, f"Cannot read JSON args file {path}: {exc}"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON args: {exc}"
    if not isinstance(value, dict):
        return None, "Tool args must be a JSON object."
    return value, None
