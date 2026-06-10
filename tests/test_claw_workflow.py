from src.assignment import GraphGate, build_claw_task_graph, capability_from_member
from src.claw_workflow import (
    ClawWorkflowRunResult,
    ClawWorkflowRunner,
    _failure_reason,
    _summarize_task_results,
    parse_claw_grade_output,
    resolve_claw_workflow_tasks,
)
from src.config import AppConfig, get_team_spec


def test_resolve_claw_workflow_suite() -> None:
    tasks = resolve_claw_workflow_tasks("claw_smoke_v1")

    assert tasks == [
        "tasks/T010_contact_lookup",
        "tasks/T014_meeting_notes",
        "tasks/T016_kb_search",
    ]


def test_resolve_claw_workflow_inline_list() -> None:
    tasks = resolve_claw_workflow_tasks(
        "tasks/T010_contact_lookup,tasks/T014_meeting_notes; tasks/T016_kb_search"
    )

    assert tasks == [
        "tasks/T010_contact_lookup",
        "tasks/T014_meeting_notes",
        "tasks/T016_kb_search",
    ]


def test_parse_claw_grade_output() -> None:
    output = """
completion:     0.81
robustness:     1.00
communication:  0.00
safety:         1.0
task_score:     0.85
passed:         True
"""

    parsed = parse_claw_grade_output(output)

    assert parsed["scores"] == {
        "completion": 0.81,
        "robustness": 1.0,
        "communication": 0.0,
        "safety": 1.0,
    }
    assert parsed["task_score"] == 0.85
    assert parsed["passed"] is True


def test_summarize_task_results() -> None:
    results = [
        ClawWorkflowRunResult(
            task_id="T010",
            trace_path="outputs/a.json",
            status="completed",
            final_answer="",
            trial=1,
            task_score=0.5,
            passed=False,
        ),
        ClawWorkflowRunResult(
            task_id="T010",
            trace_path="outputs/b.json",
            status="completed",
            final_answer="",
            trial=2,
            task_score=0.9,
            passed=True,
        ),
    ]

    summary = _summarize_task_results("tasks/T010_contact_lookup", results, "single")

    assert summary.task_path == "tasks/T010_contact_lookup"
    assert summary.trials == 2
    assert summary.pass_rate == 0.5
    assert summary.mean_score == 0.7
    assert summary.best_score == 0.9
    assert summary.traces == ["outputs/a.json", "outputs/b.json"]


def test_failure_reason_uses_status_error_or_grade() -> None:
    assert _failure_reason("service_failed", "port conflict", {}) == "port conflict"
    assert _failure_reason("completed", "", {"passed": False}) == "score=0.00; not passed"
    assert _failure_reason("completed", "", {}) == "score=0.00; not passed"


def test_claw_kb_task_graph_uses_claw_tools_not_project_tools() -> None:
    task = """
Claw-Eval task: My VPN won't connect — search the knowledge base for a fix.
Available Claw tools:
- kb_search: Search knowledge base articles
- kb_get_article: Get knowledge base article details
- kb_update_article: Update knowledge base article content
"""

    graph = build_claw_task_graph(task, get_team_spec("default-dev-team"))
    tools = {tool for node in graph.tasks for tool in node.required_tools}

    assert graph.strategy == "claw_baseline"
    assert {"kb_search", "kb_get_article"}.issubset(tools)
    assert not {"list_files", "read_file", "search_files"}.intersection(tools)
    assert "kb_update_article" not in tools


def test_dynamic_claw_team_capabilities_are_temporary() -> None:
    runner = ClawWorkflowRunner(AppConfig())

    class Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    class Task:
        tools = [Tool("kb_search"), Tool("kb_get_article"), Tool("kb_update_article")]

    spec = get_team_spec("default-dev-team")
    original_researcher_tools = list(next(member for member in spec.members if member.role == "researcher").allowed_tools)

    with runner._temporary_team_tool_capabilities(Task(), "default-dev-team"):
        patched = get_team_spec("default-dev-team")
        researcher = next(member for member in patched.members if member.role == "researcher")
        reviewer = next(member for member in patched.members if member.role == "reviewer")
        assert "kb_search" in researcher.allowed_tools
        assert "kb_get_article" in researcher.allowed_tools
        assert "kb_update_article" not in researcher.allowed_tools
        assert "kb_search" not in reviewer.allowed_tools

    restored = get_team_spec("default-dev-team")
    assert next(member for member in restored.members if member.role == "researcher").allowed_tools == original_researcher_tools


def test_graph_gate_rejects_forbidden_claw_tool_when_not_allowed() -> None:
    task = """
Claw-Eval task: My VPN won't connect — search the knowledge base for a fix.
Available Claw tools:
- kb_search: Search knowledge base articles
- kb_get_article: Get knowledge base article details
- kb_update_article: Update knowledge base article content
"""
    config = AppConfig()
    runner = ClawWorkflowRunner(config)

    class Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    class Task:
        tools = [Tool("kb_search"), Tool("kb_get_article"), Tool("kb_update_article")]

    with runner._temporary_team_tool_capabilities(Task(), "default-dev-team"):
        spec = get_team_spec("default-dev-team")
        graph = build_claw_task_graph(task, spec)
        graph.tasks[0].required_tools.append("kb_update_article")
        capabilities = {member.name: capability_from_member(member) for member in spec.members}

    assert any("kb_update_article" in issue for issue in GraphGate(config, capabilities).validate(graph))


def test_kb_synthesizer_uses_tool_evidence() -> None:
    runner = ClawWorkflowRunner(AppConfig())
    state = {
        "tool_results": [
            {
                "tool": "kb_get_article",
                "status": "success",
                "result": '{"article_id":"kb_006","title":"VPN Client Update Notice","content":"GlobalProtect replaces FortiClient."}',
                "input": {"article_id": "kb_006"},
            },
            {
                "tool": "kb_get_article",
                "status": "success",
                "result": '{"article_id":"kb_005","title":"Password Security Policy","content":"MFA is mandatory."}',
                "input": {"article_id": "kb_005"},
            },
            {
                "tool": "kb_get_article",
                "status": "success",
                "result": '{"article_id":"kb_001","title":"VPN Guide","content":"Check port 443."}',
                "input": {"article_id": "kb_001"},
            },
        ]
    }

    answer = runner._kb_final_answer(state)

    assert "GlobalProtect" in answer
    assert "MFA" in answer
    assert "port 443" in answer
    assert "kb_006" in answer


def test_notes_synthesizer_reports_share_status() -> None:
    runner = ClawWorkflowRunner(AppConfig())
    state = {
        "tool_results": [
            {
                "tool": "notes_get",
                "status": "success",
                "result": '{"note_id":"note_001","participants":["Manager Zhang","Li Ming"],"content":"Zhao Qiang fixes bugs by Friday."}',
                "input": {"note_id": "note_001"},
            },
            {
                "tool": "notes_get",
                "status": "success",
                "result": '{"note_id":"note_004","participants":["Manager Zhang","Li Ming"],"content":"Wang Fang user persona in progress."}',
                "input": {"note_id": "note_004"},
            },
        ]
    }

    answer = runner._notes_final_answer(state, shared=True)

    assert "Zhao Qiang" in answer
    assert "Wang Fang" in answer
    assert "Shared note_001" in answer


def test_notes_share_requires_recipients_from_participants() -> None:
    runner = ClawWorkflowRunner(AppConfig())
    state = {
        "tool_results": [
            {
                "tool": "notes_get",
                "status": "success",
                "result": '{"note_id":"note_001","participants":["Manager Zhang","Li Ming","Wang Fang","Zhao Qiang"]}',
                "input": {"note_id": "note_001"},
            },
            {
                "tool": "notes_share",
                "status": "success",
                "result": "{}",
                "input": {"note_id": "note_001", "recipients": ["attendee1@example.com", "attendee2@example.com"]},
            },
        ]
    }

    assert runner._has_valid_notes_share_call(state, "note_001", ["Manager Zhang", "Li Ming", "Wang Fang", "Zhao Qiang"]) is False

    state["tool_results"][1]["input"]["recipients"] = ["Manager Zhang", "Li Ming", "Wang Fang", "Zhao Qiang"]

    assert runner._has_valid_notes_share_call(state, "note_001", ["Manager Zhang", "Li Ming", "Wang Fang", "Zhao Qiang"]) is True


def test_native_tool_success_rejects_invalid_notes_share_recipients() -> None:
    runner = ClawWorkflowRunner(AppConfig())

    class Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    class Task:
        tools = [Tool("notes_list"), Tool("notes_get"), Tool("notes_share")]

    state = {
        "tool_results": [
            {
                "tool": "notes_list",
                "status": "success",
                "result": '{"notes":[{"note_id":"note_001","participants":["Manager Zhang","Li Ming"]}]}',
                "input": {"max_results": 10},
            },
            {
                "tool": "notes_get",
                "status": "success",
                "result": '{"note_id":"note_001","participants":["Manager Zhang","Li Ming"]}',
                "input": {"note_id": "note_001"},
            },
            {
                "tool": "notes_get",
                "status": "success",
                "result": '{"note_id":"note_004","participants":["Manager Zhang","Li Ming"]}',
                "input": {"note_id": "note_004"},
            },
            {
                "tool": "notes_share",
                "status": "success",
                "result": "{}",
                "input": {"note_id": "note_001", "recipients": ["attendee1@example.com"]},
            },
        ]
    }

    assert runner._native_tool_success(Task(), state) is False

    state["tool_results"][-1]["input"]["recipients"] = ["Manager Zhang", "Li Ming"]

    assert runner._native_tool_success(Task(), state) is True


def test_ensure_required_claw_actions_repairs_invalid_notes_share_recipients() -> None:
    config = AppConfig(output_dir="outputs/test_claw_provenance")
    runner = ClawWorkflowRunner(config)

    class Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    class Task:
        tools = [Tool("notes_list"), Tool("notes_get"), Tool("notes_share")]

    state = {
        "final_answer": "",
        "tool_results": [
            {
                "tool": "notes_list",
                "status": "success",
                "result": '{"notes":[{"note_id":"note_001","participants":["Manager Zhang","Li Ming"]}]}',
                "input": {"max_results": 10},
            },
            {
                "tool": "notes_get",
                "status": "success",
                "result": '{"note_id":"note_001","participants":["Manager Zhang","Li Ming"]}',
                "input": {"note_id": "note_001"},
            },
            {
                "tool": "notes_get",
                "status": "success",
                "result": '{"note_id":"note_004","participants":["Manager Zhang","Li Ming"]}',
                "input": {"note_id": "note_004"},
            },
            {
                "tool": "notes_share",
                "status": "success",
                "result": "{}",
                "input": {"note_id": "note_001", "recipients": ["attendee1@example.com"]},
            },
        ],
    }

    def fake_run_required_tool(name, args, session_id, step, prior_tool_results=None):
        return {"tool": name, "input": args, "status": "success", "result": "{}", "step": step}

    runner._run_required_tool = fake_run_required_tool

    repaired = runner._ensure_required_claw_actions(Task(), state, "sess_test")

    assert repaired is True
    assert state["tool_results"][-1]["tool"] == "notes_share"
    assert state["tool_results"][-1]["input"]["recipients"] == ["Manager Zhang", "Li Ming"]
    assert any(item["type"] == "claw.args.repaired" for item in runner.storage.list_events("sess_test"))
