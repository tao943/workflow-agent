import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.prompts.verifier_prompt import VERIFIER_SYSTEM_PROMPT
from src.state import AgentState, MAX_AGENT_STEPS, MAX_EXECUTION_ROUNDS
from src.tool_intent import required_tools_for_task


def verifier_node(state: AgentState, llm=None) -> dict:
    plan = state.get("plan", [])
    current_step_index = state.get("current_step_index", 0)
    execution_log = list(state.get("execution_log", []))
    reached_max_rounds = current_step_index >= MAX_EXECUTION_ROUNDS
    reached_max_agent_steps = state.get("agent_step_count", 0) >= MAX_AGENT_STEPS
    finished_plan = current_step_index >= len(plan)

    if finished_plan:
        acceptance = _acceptance_check(state)
        if acceptance["passed"]:
            execution_log.append("Verifier: 计划步骤已执行完毕，验收通过，进入总结。")
            error = state.get("error")
        else:
            if state.get("agent_step_count", 0) < MAX_AGENT_STEPS and _should_append_remediation(state, acceptance):
                plan.append(
                    {
                        "id": len(plan) + 1,
                        "description": acceptance["suggested_next_steps"][0] if acceptance["suggested_next_steps"] else "补救未完成的验收项",
                        "intent": "remediate",
                        "requires_tool": False,
                        "tool_name": None,
                        "tool_args": None,
                        "tool_reason": None,
                        "expected_output": "验收问题被解决",
                        "risk_level": "medium",
                        "done_criteria": "验收通过",
                        "status": "pending",
                        "result": None,
                    }
                )
                execution_log.append(f"Verifier: 验收未通过，追加补救步骤：{plan[-1]['description']}")
                return {"is_complete": False, "execution_log": execution_log, "acceptance": acceptance, "plan": plan}
            execution_log.append(f"Verifier: 计划步骤已执行完毕，验收发现问题：{'; '.join(acceptance['issues'])}")
            error = "; ".join(acceptance["issues"])
        return {"is_complete": True, "execution_log": execution_log, "acceptance": acceptance, "error": error}

    if reached_max_rounds:
        execution_log.append("Verifier: 达到最大执行轮数，强制进入总结。")
        return {"is_complete": True, "execution_log": execution_log}

    if reached_max_agent_steps:
        execution_log.append("Verifier: 达到 agent 最大循环步数，强制进入总结。")
        return {"is_complete": True, "execution_log": execution_log}

    if llm is None:
        is_complete = False
        reason = "任务尚未完成，继续执行下一步。"
    else:
        response = llm.invoke(
            [
                SystemMessage(content=VERIFIER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"用户任务：{state['user_task']}\n"
                        f"计划：{plan}\n"
                        f"当前步骤索引：{current_step_index}\n"
                        f"执行日志：{execution_log}\n"
                        f"工具结果：{state.get('tool_results', [])}"
                    )
                ),
            ]
        )
        try:
            payload = json.loads(response.content)
            is_complete = bool(payload.get("is_complete"))
            reason = str(payload.get("reason", "模型完成检查。"))
        except json.JSONDecodeError:
            is_complete = finished_plan or reached_max_rounds
            reason = "模型返回不是 JSON，使用本地规则判断。"
    execution_log.append(f"Verifier: {reason}")
    return {"is_complete": is_complete, "execution_log": execution_log}


def _should_append_remediation(state: AgentState, acceptance: dict) -> bool:
    feedback = state.get("feedback_decision") or {}
    if feedback.get("action") in {"repair", "rollback", "stop"} and _has_stable_tool_failure(state):
        return False
    if _has_rag_no_evidence(state) and _final_explains_no_evidence(state):
        return False
    suggestions = acceptance.get("suggested_next_steps") or []
    if suggestions and _has_existing_remediation(state, str(suggestions[0])):
        return False
    return True


def _has_stable_tool_failure(state: AgentState) -> bool:
    stable_error_codes = {"not_configured", "service_unavailable", "permission_denied", "invalid_args"}
    for item in state.get("tool_results", []):
        if not item.get("error"):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        if item.get("status") in {"invalid_args", "denied"}:
            return True
        if str(metadata.get("code") or "") in stable_error_codes:
            return True
    return False


def _has_rag_no_evidence(state: AgentState) -> bool:
    return any(_is_rag_no_evidence(item) for item in state.get("tool_results", []))


def _is_rag_no_evidence(item: dict) -> bool:
    if item.get("tool") != "rag_search":
        return False
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    diagnostics = metadata.get("diagnostics") if isinstance(metadata.get("diagnostics"), dict) else {}
    sources = metadata.get("sources")
    return (
        str(metadata.get("code") or "") == "no_evidence"
        or bool(diagnostics.get("no_evidence"))
        or sources == []
    )


def _has_existing_remediation(state: AgentState, description: str) -> bool:
    for step in state.get("plan", []):
        if step.get("intent") == "remediate" and step.get("description") == description:
            return True
    return False


def _acceptance_check(state: AgentState) -> dict:
    issues: list[str] = []
    missing_outputs: list[str] = []
    failed_tools: list[str] = []
    suggested_next_steps: list[str] = []
    plan = state.get("plan", [])
    user_task = state.get("user_task", "")
    enabled_tools = set(state.get("enabled_tools", []))
    tool_results = state.get("tool_results", [])
    feedback = state.get("feedback_decision") or {}

    if feedback.get("action") in {"repair", "rollback", "stop"}:
        issues.append(f"Context Supervisor: {feedback.get('reason', feedback.get('action'))}")
        suggested_next_steps.append("根据 Context Supervisor 反馈创建有限修复步骤或停止当前路径")

    for step in plan:
        if step.get("status") == "error" and not _is_expected_failure_task(user_task):
            issue = f"步骤失败：{step.get('description')}"
            issues.append(issue)
            suggested_next_steps.append(f"重试或改写步骤：{step.get('description')}")
        if step.get("status") == "pending":
            issue = f"步骤未执行：{step.get('description')}"
            issues.append(issue)
            missing_outputs.append(str(step.get("expected_output") or step.get("description")))
        if step.get("requires_tool") and step.get("tool_name") and step.get("status") == "completed":
            if not any(item.get("tool") == step.get("tool_name") for item in tool_results):
                issues.append(f"声明需要工具但没有工具记录：{step.get('tool_name')}")
                failed_tools.append(str(step.get("tool_name")))

    _check_required_tools(state, issues, missing_outputs, failed_tools, suggested_next_steps)

    latest_tool_results: dict[tuple[str, str], dict] = {}
    for item in tool_results:
        latest_tool_results[(str(item.get("step")), str(item.get("tool")))] = item

    for item in latest_tool_results.values():
        if _is_rag_no_evidence(item) and not _final_explains_no_evidence(state):
            issues.append("RAG 检索没有返回可用来源或证据，最终答案也没有明确说明无证据")
            missing_outputs.append("RAG no-evidence explanation")
            suggested_next_steps.append("明确说明未找到证据，不能自信回答")

    for item in latest_tool_results.values():
        if item.get("error") and not _is_expected_failure_task(user_task):
            issues.append(f"工具失败：{item.get('tool')}: {item.get('error')}")
            failed_tools.append(str(item.get("tool")))
            suggested_next_steps.append(f"修正 {item.get('tool')} 参数或权限后重试")

    if _expects_summary(user_task) and any(item.get("tool") == "read_file" for item in tool_results):
        if not _has_synthesis_step(plan):
            issues.append("用户要求总结文件，但计划缺少工具结果综合步骤")
            missing_outputs.append("文件摘要")
            suggested_next_steps.append("基于 read_file 结果生成摘要，而不是只返回原文")

    return {
        "passed": not issues,
        "issues": issues,
        "missing_outputs": missing_outputs,
        "failed_tools": failed_tools,
        "suggested_next_steps": suggested_next_steps,
    }


def _check_required_tools(
    state: AgentState,
    issues: list[str],
    missing_outputs: list[str],
    failed_tools: list[str],
    suggested_next_steps: list[str],
) -> None:
    user_task = state.get("user_task", "")
    enabled_tools = set(state.get("enabled_tools", []))
    required = required_tools_for_task(user_task, state.get("agent", "build"))
    if enabled_tools:
        required = [tool for tool in required if tool in enabled_tools]
    called = {str(item.get("tool")) for item in state.get("tool_results", [])}
    for tool_name in required:
        if tool_name not in called:
            issues.append(f"任务意图要求调用工具但缺少记录：{tool_name}")
            missing_outputs.append(f"{tool_name} 工具结果")
            failed_tools.append(tool_name)
            suggested_next_steps.append(f"补充调用 {tool_name} 完成用户目标")


def _expects_summary(user_task: str) -> bool:
    lowered = user_task.lower()
    return any(keyword in lowered for keyword in ("总结", "摘要", "summarize"))


def _has_synthesis_step(plan: list[dict]) -> bool:
    return any(
        step.get("intent") == "synthesize"
        or "综合" in str(step.get("description", ""))
        or "最终回答" in str(step.get("description", ""))
        for step in plan
    )


def _is_expected_failure_task(user_task: str) -> bool:
    lowered = user_task.lower()
    return bool(re.search(r"not_exist|missing|不存在|失败原因", lowered))


def _final_explains_no_evidence(state: AgentState) -> bool:
    texts = [str(state.get("final_answer") or "")]
    texts.extend(str(step.get("result") or "") for step in state.get("plan", []))
    joined = "\n".join(texts).lower()
    return any(
        keyword in joined
        for keyword in (
            "无证据",
            "没有证据",
            "没有找到",
            "未找到",
            "足够依据",
            "no evidence",
            "not configured",
            "service_unavailable",
        )
    )


def route_after_verifier(state: AgentState) -> str:
    if state.get("current_step_index", 0) >= len(state.get("plan", [])):
        return "summarizer"
    if state.get("agent_step_count", 0) >= MAX_AGENT_STEPS:
        return "summarizer"
    if state.get("is_complete"):
        return "summarizer"
    return "executor"
