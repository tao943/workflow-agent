import json

from src.assignment import (
    AnalysisStrategyRouter,
    AssignmentGate,
    AssignmentPlanner,
    GraphReplanner,
    TaskClassifier,
    TaskGraph,
    TaskNode,
    build_member_capabilities,
)
from src.config import AppConfig, TeamModeConfig, get_team_spec


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _GoodRouteLLM:
    def invoke(self, messages):
        return _FakeResponse(
            json.dumps(
                {
                    "intent": "analysis",
                    "confidence": 0.91,
                    "rationale_summary": "project analysis needs evidence then review",
                    "required_capabilities": ["project_evidence", "planning", "review"],
                    "suggested_graph": {
                        "tasks": [
                            {
                                "id": "semantic_evidence",
                                "title": "Collect semantic evidence",
                                "description": "Collect project evidence.",
                                "assigned_to": "researcher",
                                "depends_on": [],
                                "node_type": "task",
                                "condition": "always",
                                "required_tools": ["list_files"],
                                "required_evidence": ["project evidence"],
                                "acceptance_criteria": "tool evidence exists",
                                "risk_level": "low",
                                "why_this_member": "researcher has read tools",
                                "why_now": "evidence comes first",
                                "skip_condition": "",
                            },
                            {
                                "id": "semantic_review",
                                "title": "Review semantic evidence",
                                "description": "Review project findings.",
                                "assigned_to": "reviewer",
                                "depends_on": ["semantic_evidence"],
                                "node_type": "review",
                                "condition": "always",
                                "required_tools": [],
                                "required_evidence": ["project evidence"],
                                "acceptance_criteria": "review cites evidence",
                                "risk_level": "low",
                                "why_this_member": "reviewer validates claims",
                                "why_now": "after evidence",
                                "skip_condition": "",
                            },
                        ]
                    },
                }
            )
        )


class _LowConfidenceRouteLLM:
    def invoke(self, messages):
        return _FakeResponse(json.dumps({"intent": "analysis", "confidence": 0.2, "suggested_graph": {"tasks": []}}))


class _BadJsonRouteLLM:
    def invoke(self, messages):
        return _FakeResponse("not json")


def test_task_classifier_detects_core_intents():
    classifier = TaskClassifier()

    assert classifier.classify("分析当前项目结构") == "analysis"
    assert classifier.classify("帮我实现 memory 优化") == "build"
    assert classifier.classify("检索知识库 LangGraph checkpoint") == "research"
    assert classifier.classify("制定多 agent 优化计划") == "planning"


def test_analysis_baseline_does_not_assign_builder_by_default():
    spec = get_team_spec("default-dev-team")
    graph = AssignmentPlanner(AppConfig()).plan("分析当前项目结构", spec)

    assigned = {task.assigned_to for task in graph.tasks}
    assert "builder" not in assigned
    assert "researcher" in assigned
    assert "planner" in assigned
    assert "reviewer" in assigned


def test_analysis_strategy_router_selects_testing_and_benchmark_focus():
    router = AnalysisStrategyRouter()

    testing = router.route("分析当前测试体系还有哪些缺口")
    benchmark = router.route("根据 Claw-Eval benchmark 结果分析失败原因")

    assert testing.analysis_subtype == "testing"
    assert "pytest.ini" in testing.suggested_files
    assert any("skip" in query for query in testing.search_queries)
    assert benchmark.analysis_subtype == "benchmark"
    assert "src/claw_workflow.py" in benchmark.suggested_files


def test_analysis_graph_records_evidence_plan_metadata():
    spec = get_team_spec("default-dev-team")
    graph = AssignmentPlanner(AppConfig()).plan("分析当前测试体系还有哪些缺口", spec)
    evidence_task = graph.tasks[0]

    assert evidence_task.metadata["analysis_subtype"] == "testing"
    assert evidence_task.metadata["evidence_plan"]["analysis_subtype"] == "testing"
    assert "pytest.ini" in evidence_task.metadata["evidence_plan"]["suggested_files"]


def test_quick_analysis_uses_short_route_without_reviewer_or_builder():
    spec = get_team_spec("default-dev-team")
    graph = AssignmentPlanner(AppConfig()).plan("快速看一下项目问题", spec)

    assert len(graph.tasks) == 1
    assert graph.tasks[0].assigned_to == "researcher"
    assert graph.tasks[0].metadata["evidence_plan"]["depth"] == "quick"


def test_build_baseline_has_research_plan_build_review_chain():
    spec = get_team_spec("default-dev-team")
    graph = AssignmentPlanner(AppConfig()).plan("帮我实现 memory 优化", spec)
    by_id = {task.id: task for task in graph.tasks}

    assert list(by_id) == ["collect_project_evidence", "plan_build", "execute_build", "review_build"]
    assert by_id["plan_build"].depends_on == ["collect_project_evidence"]
    assert set(by_id["execute_build"].depends_on) == {"collect_project_evidence", "plan_build"}
    assert set(by_id["review_build"].depends_on) == {"collect_project_evidence", "plan_build", "execute_build"}


def test_research_baseline_uses_rag_search():
    spec = get_team_spec("research-build")
    graph = AssignmentPlanner(AppConfig()).plan("检索知识库 LangGraph checkpoint", spec)

    assert graph.tasks[0].required_tools == ["rag_search"]
    assert graph.tasks[0].assigned_to == "researcher"


def test_assignment_gate_rejects_unknown_member_and_readonly_write():
    spec = get_team_spec("default-dev-team")
    capabilities = build_member_capabilities(spec)
    gate = AssignmentGate(AppConfig(), capabilities)
    graph = TaskGraph(
        intent="build",
        strategy="llm_refined",
        tasks=[
            TaskNode(id="bad_member", title="Bad", description="Bad", assigned_to="ghost"),
            TaskNode(id="bad_write", title="Write", description="Write", assigned_to="planner", required_tools=["write_file"]),
        ],
    )

    issues = gate.validate(graph)

    assert any("does not exist" in item for item in issues)
    assert any("not allowed" in item or "read-only" in item for item in issues)


def test_assignment_strategy_can_disable_llm_refine():
    spec = get_team_spec("default-dev-team")
    config = AppConfig(team_mode=TeamModeConfig(assignment_strategy="rules"))
    graph = AssignmentPlanner(config, llm=object()).plan("分析当前项目结构", spec)

    assert graph.strategy == "baseline"
    assert not graph.fallback_used


def test_semantic_router_uses_high_confidence_llm_graph():
    spec = get_team_spec("default-dev-team")
    config = AppConfig()
    graph = AssignmentPlanner(config, llm=_GoodRouteLLM()).plan("Analyze this project", spec)

    assert graph.strategy == "llm_semantic"
    assert graph.confidence == 0.91
    assert [task.id for task in graph.tasks] == ["semantic_evidence", "semantic_review"]


def test_semantic_router_falls_back_on_low_confidence_or_bad_json():
    spec = get_team_spec("default-dev-team")

    low = AssignmentPlanner(AppConfig(), llm=_LowConfidenceRouteLLM()).plan("Analyze this project", spec)
    bad = AssignmentPlanner(AppConfig(), llm=_BadJsonRouteLLM()).plan("Analyze this project", spec)

    assert low.strategy == "baseline"
    assert bad.strategy == "baseline"
    assert bad.fallback_used is True


def test_graph_gate_rejects_cycles():
    spec = get_team_spec("default-dev-team")
    gate = AssignmentGate(AppConfig(), build_member_capabilities(spec))
    graph = TaskGraph(
        intent="analysis",
        strategy="llm_semantic",
        tasks=[
            TaskNode(id="a", title="A", description="A", assigned_to="planner", depends_on=["b"]),
            TaskNode(id="b", title="B", description="B", assigned_to="reviewer", depends_on=["a"], node_type="review"),
        ],
    )

    assert any("circular" in item for item in gate.validate(graph))


def test_graph_replanner_adds_repair_and_supplemental_evidence_tasks():
    spec = get_team_spec("default-dev-team")
    graph = TaskGraph(intent="analysis", strategy="baseline", tasks=[TaskNode(id="a", title="A", description="A", assigned_to="planner")])
    replanner = GraphReplanner(AppConfig(), build_member_capabilities(spec))

    additions = replanner.maybe_replan(
        graph,
        completed={"a"},
        failed={"a"},
        results=[{"evidence_check": {"passed": False, "status": "missing"}}],
        replan_round=0,
    )

    assert {item.node_type for item in additions} == {"repair"}
    assert any(item.condition == "if_evidence_missing" for item in additions)


def test_graph_replanner_can_skip_optional_write_task_when_user_did_not_request_write():
    spec = get_team_spec("default-dev-team")
    replanner = GraphReplanner(AppConfig(), build_member_capabilities(spec))
    node = TaskNode(
        id="optional_write",
        title="Optional write",
        description="Write optional note",
        assigned_to="builder",
        optional=True,
        condition="if_user_requested_write",
    )

    assert replanner.should_skip(node, "Analyze this project", completed=set(), failed=set()) == "user did not request write-capable work"
