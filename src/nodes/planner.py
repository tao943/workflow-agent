import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.memory import format_memory_for_prompt
from src.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT
from src.prompts.templates import PROMPT_VERSION
from src.session import new_id
from src.state import AgentState, PlanStep
from src.storage import Storage
from src.tool_intent import required_tools_for_task, user_intends_tool
from src.tools.registry import get_tool_names


ALLOWED_TOOLS = get_tool_names()
INTENT_GATED_TOOLS = {
    "calculator",
    "datetime_tool",
    "list_files",
    "read_file",
    "search_files",
    "rag_search",
    "write_file",
    "notes_tool",
}


def _make_step(
    step_id: int,
    description: str,
    tool_name: str | None = None,
    enabled_tools: set[str] | None = None,
    user_task: str = "",
    tool_args: dict[str, Any] | None = None,
    intent: str = "answer",
    expected_output: str = "",
    done_criteria: str = "",
    tool_reason: str | None = None,
    risk_level: str = "low",
    required_evidence: list[str] | None = None,
) -> PlanStep:
    allowed_tools = enabled_tools or ALLOWED_TOOLS
    if tool_name not in allowed_tools:
        tool_name = None
    elif tool_name in INTENT_GATED_TOOLS and not user_intends_tool(user_task, tool_name):
        tool_name = None
    requires_tool = tool_name is not None
    return {
        "id": step_id,
        "description": description,
        "status": "pending",
        "tool_name": tool_name,
        "tool_args": tool_args if requires_tool else None,
        "result": None,
        "intent": intent,
        "requires_tool": requires_tool,
        "tool_reason": tool_reason if requires_tool else None,
        "expected_output": expected_output or description,
        "required_evidence": required_evidence or [],
        "risk_level": risk_level,
        "done_criteria": done_criteria or f"完成：{description}",
    }


def _fallback_plan(user_task: str, enabled_tools: set[str] | None = None, agent: str = "build") -> list[PlanStep]:
    required_tools = [
        tool
        for tool in required_tools_for_task(user_task, agent)
        if enabled_tools is None or tool in enabled_tools
    ]
    tool_descriptions = {
        "calculator": "计算数学表达式",
        "datetime_tool": "获取当前日期时间",
        "list_files": "列出项目文件结构",
        "read_file": "读取相关项目文件",
        "search_files": "搜索项目代码或文件",
        "rag_search": "检索知识库证据",
        "write_file": "写入请求的输出文件",
        "notes_tool": "保存笔记",
    }
    steps: list[PlanStep] = []
    for tool_name in required_tools:
        steps.append(
            _make_step(
                len(steps) + 1,
                tool_descriptions.get(tool_name, f"调用 {tool_name}"),
                tool_name,
                enabled_tools,
                user_task,
                intent="use_tool",
                tool_reason="用户任务明确需要该工具能力",
                expected_output=f"{tool_name} 工具结果",
                done_criteria=f"{tool_name} 调用完成并记录证据",
                required_evidence=[tool_name],
            )
        )

    evidence_tools = {"read_file", "list_files", "search_files", "rag_search"}
    if any(tool in evidence_tools for tool in required_tools):
        steps.append(
            _make_step(
                len(steps) + 1,
                "基于工具证据生成最终回答",
                None,
                enabled_tools,
                user_task,
                intent="synthesize",
                expected_output="面向用户目标的综合答案",
                done_criteria="最终答案引用或解释工具证据",
            )
        )
    elif not steps:
        steps.append(_make_step(1, "生成完整结果", None, enabled_tools, user_task))
    elif not any(tool in {"write_file", "notes_tool"} for tool in required_tools):
        steps.append(_make_step(len(steps) + 1, "整理最终结果", None, enabled_tools, user_task))

    return _compact_content_steps(steps, enabled_tools, user_task)


def _normalize_steps(
    raw_steps: list[Any],
    enabled_tools: set[str] | None = None,
    user_task: str = "",
) -> list[PlanStep]:
    normalized: list[PlanStep] = []
    max_steps = 8 if _has_dynamic_tools(enabled_tools or ALLOWED_TOOLS) else 5
    for index, raw_step in enumerate(raw_steps[:max_steps], start=1):
        intent = "answer"
        expected_output = ""
        done_criteria = ""
        tool_reason = None
        tool_args = None
        risk_level = "low"
        required_evidence: list[str] = []
        if isinstance(raw_step, str):
            description = raw_step.strip()
            tool_name = None
        elif isinstance(raw_step, dict):
            description = str(raw_step.get("description", "")).strip()
            tool_name = raw_step.get("tool_name")
            raw_tool_args = raw_step.get("tool_args")
            tool_args = raw_tool_args if isinstance(raw_tool_args, dict) else None
            intent = str(raw_step.get("intent", "answer")).strip() or "answer"
            expected_output = str(raw_step.get("expected_output", "")).strip()
            done_criteria = str(raw_step.get("done_criteria", "")).strip()
            tool_reason = raw_step.get("tool_reason")
            risk_level = str(raw_step.get("risk_level", "low")).strip() or "low"
            raw_required_evidence = raw_step.get("required_evidence")
            if isinstance(raw_required_evidence, list):
                required_evidence = [str(item) for item in raw_required_evidence if str(item).strip()]
        else:
            continue

        if description:
            normalized.append(
                _make_step(
                    index,
                    description,
                    tool_name,
                    enabled_tools,
                    user_task,
                    tool_args=tool_args if isinstance(raw_step, dict) else None,
                    intent=intent if isinstance(raw_step, dict) else "answer",
                    expected_output=expected_output if isinstance(raw_step, dict) else "",
                    done_criteria=done_criteria if isinstance(raw_step, dict) else "",
                    tool_reason=str(tool_reason).strip() if tool_reason else None,
                    risk_level=risk_level if isinstance(raw_step, dict) else "low",
                    required_evidence=required_evidence if isinstance(raw_step, dict) else None,
                )
            )

    return _compact_content_steps(normalized, enabled_tools, user_task)


def _compact_content_steps(
    steps: list[PlanStep],
    enabled_tools: set[str] | None = None,
    user_task: str = "",
) -> list[PlanStep]:
    tool_steps = [step for step in steps if step["tool_name"]]
    content_steps = [step for step in steps if not step["tool_name"]]
    if len(content_steps) < 2:
        return steps
    if not tool_steps:
        return [_make_step(1, "生成完整结果", None, enabled_tools, user_task)]

    compacted: list[PlanStep] = []
    evidence_tools = {"read_file", "list_files", "search_files", "rag_search"}
    if not any(step["tool_name"] in evidence_tools for step in tool_steps):
        compacted.append(_make_step(1, "生成完整结果", None, enabled_tools, user_task))

    max_tool_steps = 7 if _has_dynamic_tools(enabled_tools or ALLOWED_TOOLS) else 4
    for step in tool_steps[:max_tool_steps]:
        compacted.append(
            _make_step(
                len(compacted) + 1,
                step["description"],
                step["tool_name"],
                enabled_tools,
                user_task,
                tool_args=step.get("tool_args"),
                intent=step.get("intent", "use_tool"),
                expected_output=step.get("expected_output", ""),
                done_criteria=step.get("done_criteria", ""),
                tool_reason=step.get("tool_reason"),
                risk_level=step.get("risk_level", "low"),
                required_evidence=step.get("required_evidence", []),
            )
        )
    if content_steps and any(step["tool_name"] in evidence_tools for step in tool_steps):
        compacted.append(
            _make_step(
                len(compacted) + 1,
                "基于工具证据生成最终回答",
                None,
                enabled_tools,
                user_task,
                intent="synthesize",
                expected_output="面向用户目标的综合答案",
            )
        )
    return compacted


def planner_node(state: AgentState, llm=None) -> dict:
    user_task = state["user_task"]
    enabled_tools = set(state.get("enabled_tools", [])) or None
    agent = state.get("agent", "build")

    if llm is None:
        steps = _fallback_plan(user_task, enabled_tools, agent)
    else:
        memory_text = format_memory_for_prompt(user_task, state.get("memory", []))
        session_summary = state.get("session_summary") or "无"
        skill_text = state.get("skill_instructions") or "无"
        response = llm.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"用户任务：{user_task}\n\n"
                        f"当前 agent：{agent}\n"
                        f"启用技能：{state.get('enabled_skills', [])}\n"
                        f"技能说明：\n{skill_text}\n\n"
                        f"可用工具：{sorted(enabled_tools or ALLOWED_TOOLS)}\n\n"
                        f"会话压缩摘要：\n{session_summary}\n\n"
                        f"可参考记忆：\n{memory_text}"
                    )
                ),
            ]
        )
        try:
            payload = json.loads(response.content)
            steps = _normalize_steps(payload.get("steps", []), enabled_tools, user_task)
        except json.JSONDecodeError:
            steps = []

        if not steps:
            steps = _fallback_plan(user_task, enabled_tools, agent)

    if _has_dynamic_tools(enabled_tools or ALLOWED_TOOLS):
        steps = steps[:8]
    else:
        steps = steps[:5]
    _emit_prompt_event(state, "planner", PROMPT_VERSION)
    return {
        "plan": steps,
        "current_step_index": 0,
        "execution_log": [f"Planner: 生成 {len(steps)} 个结构化步骤。"],
        "tool_results": [],
        "is_complete": False,
        "final_answer": "",
        "error": None,
    }


def _has_dynamic_tools(tool_names: set[str]) -> bool:
    return any(name not in INTENT_GATED_TOOLS and not name.startswith("team_") for name in tool_names)


def _emit_prompt_event(state: AgentState, prompt_name: str, version: str) -> None:
    session_id = state.get("session_id")
    if not session_id:
        return
    Storage(state.get("storage_path") or "outputs/agent.sqlite").add_event(
        session_id,
        "prompt.used",
        {"prompt": prompt_name, "version": version, "message_id": state.get("message_id") or new_id("msg")},
    )
