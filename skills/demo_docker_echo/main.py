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
                    "title": "Docker echo failed",
                    "output": "",
                    "error": "The demo_docker_echo.echo tool requires a non-empty text argument.",
                    "metadata": {"skill": "demo_docker_echo"},
                },
                ensure_ascii=False,
            )
        )
        return 0
    print(
        json.dumps(
            {
                "status": "success",
                "title": "Docker echo completed",
                "output": text,
                "metadata": {"skill": "demo_docker_echo", "runner": "docker", "length": len(text)},
                "attachments": [],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
