from __future__ import annotations

import json
from typing import Any


def print_output(output_format: str, text: str, payload: dict[str, Any]) -> None:
    if output_format == "json":
        print(json.dumps(payload, ensure_ascii=False))
        return
    print(text)

