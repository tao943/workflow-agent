import json
import sys


def main() -> int:
    request = json.loads(sys.stdin.read() or "{}")
    args = request.get("args") or {}
    text = args.get("text")
    if not isinstance(text, str) or not text.strip():
        print(
            json.dumps(
                {
                    "status": "invalid_args",
                    "title": "Echo failed",
                    "output": "",
                    "error": "The demo_echo.echo tool requires a non-empty text argument.",
                    "metadata": {"skill": "demo_echo"},
                },
                ensure_ascii=False,
            )
        )
        return 0
    print(
        json.dumps(
            {
                "status": "success",
                "title": "Echo completed",
                "output": text,
                "metadata": {"skill": "demo_echo", "length": len(text)},
                "attachments": [],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
