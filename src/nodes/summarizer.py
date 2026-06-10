import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.memory import format_memory_for_prompt
from src.prompts.summarizer_prompt import SUMMARIZER_SYSTEM_PROMPT
from src.state import AgentState


def _fallback_summary(state: AgentState) -> str:
    acceptance = state.get("acceptance") or {}
    task_passed = bool(acceptance.get("passed", True)) and not state.get("error")
    status_line = "- 任务执行完成，已通过验收。" if task_passed else "- 任务流程已结束，但未通过验收。"
    lines = [
        "## 已证实结论",
        "",
        status_line,
        f"- 用户任务：{state['user_task']}",
        "",
        "## 合理推断",
        "",
        "- 当前进展可由计划状态、执行日志和工具结果推断：",
    ]
    lines.extend(
        f"- [{step['status']}] {step['description']}（工具：{step.get('tool_name') or '无'}）"
        for step in state.get("plan", [])
    )

    if acceptance and not acceptance.get("passed", True):
        lines.extend(["", "验收问题："])
        lines.extend(f"- {_sanitize_for_final(item)}" for item in acceptance.get("issues", []))
        missing_outputs = acceptance.get("missing_outputs") or []
        if missing_outputs:
            lines.extend(["", "缺失产物："])
            lines.extend(f"- {_sanitize_for_final(item)}" for item in missing_outputs)

    lines.extend(
        [
            "",
            "## 未验证假设",
            "",
            "- 没有工具结果或 evidence_id 支撑的模型文本，只能作为未验证假设。",
            "",
            "## 下一步",
            "",
            "- 如需更可靠结论，请补充工具证据或查看下方执行日志。",
            "",
            "执行日志：",
        ]
    )
    lines.extend(f"- {_sanitize_for_final(item)}" for item in state.get("execution_log", []))

    step_results = [step for step in state.get("plan", []) if step.get("result")]
    if step_results:
        lines.extend(["", "执行结果："])
        for step in step_results:
            lines.append(f"### {step['description']}")
            lines.append(_safe_summary_text(str(step["result"])))

    tool_results = state.get("tool_results", [])
    if tool_results:
        lines.extend(["", "工具结果："])
        for item in tool_results:
            tool_name = item.get("tool")
            raw_path = item.get("raw_output_path")
            suffix = f"（完整结果：{raw_path}）" if raw_path else ""
            lines.append(f"- {tool_name}: {_safe_summary_text(str(item.get('result') or item.get('error') or ''))}{suffix}")

    if state.get("error"):
        lines.extend(["", f"错误：{_sanitize_for_final(state['error'])}"])

    return "\n".join(lines)


def _safe_summary_text(text: str, limit: int = 700) -> str:
    compact = " ".join(str(text).split())
    compact = _sanitize_for_final(compact)
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _sanitize_for_final(text: str) -> str:
    return re.sub(r"\bev_(\d+)\b", r"ev-\1", str(text))


def summarizer_node(state: AgentState, llm=None) -> dict:
    if llm is None:
        final_answer = _fallback_summary(state)
    else:
        memory_text = format_memory_for_prompt(state["user_task"], state.get("memory", []))
        session_summary = state.get("session_summary") or "无"
        response = llm.invoke(
            [
                SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"用户任务：{state['user_task']}\n"
                        f"计划：{state.get('plan', [])}\n"
                        f"执行日志：{state.get('execution_log', [])}\n"
                        f"工具结果：{state.get('tool_results', [])}\n"
                        f"会话压缩摘要：{session_summary}\n"
                        f"可参考记忆：{memory_text}\n"
                        f"错误：{state.get('error')}"
                    )
                ),
            ]
        )
        final_answer = response.content

    return {"final_answer": final_answer}
