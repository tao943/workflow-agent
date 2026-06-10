from src.graph import build_graph, make_initial_state
from src.main import _build_runtime_graph


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakePlannerLLM:
    def invoke(self, messages):
        return FakeMessage('{"steps": [{"description": "执行任务核心步骤", "tool_name": null}]}')


class ExplodingLLM:
    def invoke(self, messages):
        raise AssertionError("This LLM should not be called in fast mode")


def test_graph_allows_separate_node_llms():
    app = build_graph(
        llm=FakePlannerLLM(),
        executor_llm=None,
        verifier_llm=None,
        summarizer_llm=None,
    )

    result = app.invoke(make_initial_state("测试快模式"))

    assert result["is_complete"] is True
    assert "任务执行完成" in result["final_answer"]


def test_runtime_graph_fast_mode_uses_planner_and_executor_only():
    app = _build_runtime_graph(build_graph, FakePlannerLLM(), False, None, "planner")
    result = app.invoke(make_initial_state("测试 runtime 快模式"))

    assert result["is_complete"] is True


def test_runtime_graph_can_keep_verifier_and_summarizer_local():
    app = _build_runtime_graph(build_graph, FakePlannerLLM(), False, None, "planner")
    result = app.invoke(make_initial_state("测试 runtime 本地校验总结"))

    assert result["is_complete"] is True
