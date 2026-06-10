from dataclasses import dataclass, field
from typing import Any

from src.config import ContextConfig
from src.memory import format_memory_for_prompt
from src.storage import Storage


@dataclass
class ContextBundle:
    system_context: str
    session_summary: str
    recent_messages: list[dict[str, str]]
    relevant_memory: str
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    estimated_chars: int = 0


class ContextManager:
    """Compatibility adapter for legacy callers.

    New runtime code should prefer ContextHarness.prepare(), which returns a
    manifest with selected memories, artifacts, budget, and retrieval reasons.
    """

    def __init__(self, storage: Storage, config: ContextConfig | None = None) -> None:
        self.storage = storage
        self.config = config or ContextConfig()

    def build(self, session_id: str, user_task: str, memory: list[dict]) -> ContextBundle:
        messages = self.storage.list_messages(session_id) if session_id else []
        keep = max(1, self.config.recent_turns * 2)
        recent_messages = messages[-keep:]
        session_summary = self.storage.compact_session(session_id, keep_tail_messages=keep) if session_id else ""
        relevant_memory = format_memory_for_prompt(user_task, memory[-self.config.memory_items :])
        tool_outputs = self._tool_outputs(session_id)
        system_context = self._clip(
            "\n\n".join(
                [
                    "会话摘要：\n" + (session_summary or "无"),
                    "最近消息：\n" + self._format_messages(recent_messages),
                    "相关记忆：\n" + relevant_memory,
                    "工具输出摘要：\n" + self._format_tool_outputs(tool_outputs),
                ]
            ),
            self.config.max_prompt_chars,
        )
        return ContextBundle(
            system_context=system_context,
            session_summary=session_summary,
            recent_messages=recent_messages,
            relevant_memory=relevant_memory,
            tool_outputs=tool_outputs,
            estimated_chars=len(system_context),
        )

    def _tool_outputs(self, session_id: str) -> list[dict[str, Any]]:
        if not session_id:
            return []
        outputs = []
        for item in self.storage.list_tool_calls(session_id)[-10:]:
            output = item.get("output") or ""
            outputs.append(
                {
                    "tool": item.get("tool_name"),
                    "output": self._clip(output, self.config.tool_output_max_chars),
                    "error": item.get("error"),
                }
            )
        return outputs

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        if not messages:
            return "无"
        return "\n".join(f"- {item['role']}: {self._clip(item['content'], 500)}" for item in messages)

    def _format_tool_outputs(self, outputs: list[dict[str, Any]]) -> str:
        if not outputs:
            return "无"
        return "\n".join(f"- {item['tool']}: {item['error'] or item['output']}" for item in outputs)

    def _clip(self, text: str, limit: int) -> str:
        return text if len(text) <= limit else f"{text[:limit]}..."
