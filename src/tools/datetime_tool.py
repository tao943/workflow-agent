from datetime import datetime


def datetime_tool(input: str) -> str:
    """Return the current local date and time."""
    now = datetime.now().astimezone()
    return f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
