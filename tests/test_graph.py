from src.graph import build_graph, make_initial_state
from src.nodes.executor import _select_tool
from src.nodes.verifier import verifier_node
from src.state import MAX_AGENT_STEPS, MAX_EXECUTION_ROUNDS


def test_graph_generates_plan_and_summary_without_llm():
    app = build_graph(llm=None)
    result = app.invoke(make_initial_state("帮我制定一个三天学习 LangGraph 的计划"))

    assert result["plan"]
    assert result["plan"][0]["description"] == "生成完整结果"
    assert result["plan"][0]["status"] == "completed"
    assert result["is_complete"] is True
    assert "任务执行完成" in result["final_answer"]
    assert "执行结果" in result["final_answer"]


def test_graph_saves_notes_without_llm():
    app = build_graph(llm=None)
    result = app.invoke(
        make_initial_state(
            "帮我制定一个三天学习 LangGraph 的计划，并保存为笔记",
            auto_approve=True,
            enabled_tools=["notes_tool"],
        )
    )

    assert any(item.get("tool") == "notes_tool" for item in result["tool_results"])
    assert result["approvals"][0]["approved"] is True
    assert result["is_complete"] is True


def test_graph_skips_notes_when_approval_is_missing_in_noninteractive_mode():
    app = build_graph(llm=None)
    result = app.invoke(
        make_initial_state(
            "帮我制定一个三天学习 LangGraph 的计划，并保存为笔记",
            enabled_tools=["notes_tool"],
        )
    )

    assert result["tool_results"][0]["status"] == "denied"
    assert result["approvals"][0]["approved"] is False
    assert any(step["status"] == "skipped" for step in result["plan"])


def test_graph_uses_math_and_datetime_tools_without_repeating_calculator():
    app = build_graph(llm=None)
    result = app.invoke(
        make_initial_state(
            "计算 12 * (8 + 5)，再告诉我今天日期",
            permission_rules=[{"permission": "read", "pattern": "*", "action": "allow"}],
        )
    )

    tool_names = [item["tool"] for item in result["tool_results"]]
    assert tool_names == ["calculator", "datetime_tool"]
    planned_tools = [step["tool_name"] for step in result["plan"]]
    assert "calculator" in planned_tools
    assert "datetime_tool" in planned_tools


def test_graph_skips_disabled_tool():
    app = build_graph(llm=None)
    result = app.invoke(make_initial_state("计算 1 + 2", enabled_tools=["datetime_tool"]))

    assert result["tool_results"] == []
    assert all(step["tool_name"] is None for step in result["plan"])
    assert all(step["status"] == "completed" for step in result["plan"])


def test_executor_rejects_tool_when_user_did_not_intend_it():
    tool_name, tool_input = _select_tool(
        "帮我制定一个三天学习 LangGraph 的计划",
        "查询当前日期时间，确定计划起始日",
        "datetime_tool",
        {"datetime_tool"},
    )

    assert tool_name is None
    assert tool_input is None


def test_verifier_stops_at_max_rounds():
    state = make_initial_state("执行一个很长的任务")
    state["plan"] = [f"步骤 {index}" for index in range(MAX_EXECUTION_ROUNDS + 2)]
    state["current_step_index"] = MAX_EXECUTION_ROUNDS

    result = verifier_node(state, llm=None)

    assert result["is_complete"] is True


class AlwaysIncompleteLLM:
    def invoke(self, messages):
        class Response:
            content = '{"is_complete": false, "reason": "模型认为任务还没完成"}'

        return Response()


def test_verifier_does_not_loop_when_plan_is_finished_even_if_model_disagrees():
    state = make_initial_state("执行一个无法保存的任务")
    state["plan"] = [
        {
            "id": 1,
            "description": "说明当前模式只提供规划，不执行写入",
            "status": "completed",
            "tool_name": None,
            "result": "已完成",
        }
    ]
    state["current_step_index"] = 1

    result = verifier_node(state, llm=AlwaysIncompleteLLM())

    assert result["is_complete"] is True
    assert "计划步骤已执行完毕" in result["execution_log"][-1]


def test_route_forces_summary_when_plan_finished_even_if_state_says_incomplete():
    from src.nodes.verifier import route_after_verifier

    state = make_initial_state("测试路由")
    state["plan"] = [
        {
            "id": 1,
            "description": "完成一步",
            "status": "completed",
            "tool_name": None,
            "result": "done",
        }
    ]
    state["current_step_index"] = 1
    state["is_complete"] = False

    assert route_after_verifier(state) == "summarizer"


def test_verifier_stops_at_agent_step_budget():
    state = make_initial_state("测试预算")
    state["plan"] = [
        {
            "id": 1,
            "description": "还没做",
            "status": "pending",
            "tool_name": None,
            "result": None,
        }
    ]
    state["agent_step_count"] = MAX_AGENT_STEPS

    result = verifier_node(state, llm=None)

    assert result["is_complete"] is True
    assert "最大循环步数" in result["execution_log"][-1]


def test_verifier_acceptance_detects_missing_required_write():
    state = make_initial_state("帮我保存为笔记", enabled_tools=["notes_tool"])
    state["plan"] = [
        {
            "id": 1,
            "description": "生成完整结果",
            "status": "completed",
            "tool_name": None,
            "result": "done",
        }
    ]
    state["current_step_index"] = 1

    result = verifier_node(state, llm=None)

    assert result["is_complete"] is False
    assert result["acceptance"]["passed"] is False
    assert "notes_tool" in result["acceptance"]["issues"][0]
    assert result["plan"][-1]["intent"] == "remediate"


def test_verifier_does_not_append_noop_remediation_for_stable_tool_failure():
    state = make_initial_state("search knowledge base")
    state["plan"] = [
        {
            "id": 1,
            "description": "Search RAG",
            "status": "error",
            "tool_name": "rag_search",
            "result": "service unavailable",
        }
    ]
    state["current_step_index"] = 1
    state["tool_results"] = [
        {
            "step": 1,
            "tool": "rag_search",
            "status": "error",
            "error": "RAG service is unavailable",
            "metadata": {"code": "service_unavailable"},
        }
    ]
    state["feedback_decision"] = {"action": "repair", "reason": "tool_result:rag_search:service_unavailable"}

    result = verifier_node(state, llm=None)

    assert result["is_complete"] is True
    assert result["acceptance"]["passed"] is False
    assert len(result.get("plan", state["plan"])) == 1


def test_verifier_accepts_honest_rag_no_evidence_explanation():
    state = make_initial_state("检索知识库 LangGraph checkpoint")
    state["plan"] = [
        {
            "id": 1,
            "description": "检索 RAG 知识库",
            "status": "completed",
            "tool_name": "rag_search",
            "result": "知识库中没有找到足够依据。",
        }
    ]
    state["current_step_index"] = 1
    state["tool_results"] = [
        {
            "step": 1,
            "tool": "rag_search",
            "status": "success",
            "result": "知识库中没有找到足够依据。",
            "metadata": {"code": "no_evidence", "sources": [], "diagnostics": {"no_evidence": True}},
        }
    ]
    state["feedback_decision"] = {"action": "retrieve", "reason": "tool result has no supporting evidence"}

    result = verifier_node(state, llm=None)

    assert result["is_complete"] is True
    assert result["acceptance"]["passed"] is True
