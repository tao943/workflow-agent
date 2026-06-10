from pathlib import Path
from uuid import uuid4

from src.config import PermissionRule
from src.argument_provenance import ArgumentProvenanceRule
from src.storage import Storage
from src.tool_runtime import ToolRuntime, ToolRunRequest
from src.tools.registry import OUTPUT_SCHEMA, ToolContext, ToolResult, ToolSpec, register_tool


def test_tool_runtime_invalid_args_are_structured():
    root = Path("outputs") / "test_tool_runtime" / uuid4().hex[:8]
    storage = Storage(root / "agent.sqlite")
    result, approvals = ToolRuntime(root / "artifacts").run(
        ToolRunRequest(
            name="read_file",
            args={},
            context=ToolContext("sess", "step", Path("outputs")),
            session_id="sess",
            step_id="step",
            enabled_tools={"read_file"},
            storage=storage,
        )
    )

    assert approvals == []
    assert result.status == "invalid_args"
    assert "invalid arguments" in result.error


def test_tool_runtime_deny_beats_auto_approve():
    root = Path("outputs") / "test_tool_runtime" / uuid4().hex[:8]
    storage = Storage(root / "agent.sqlite")
    result, approvals = ToolRuntime(root / "artifacts").run(
        ToolRunRequest(
            name="write_file",
            args={"path": "outputs/runtime_denied.md", "content": "no"},
            context=ToolContext("sess", "step", Path("outputs")),
            session_id="sess",
            step_id="step",
            enabled_tools={"write_file"},
            permission_rules=[PermissionRule(permission="write", pattern="*", action="deny")],
            auto_approve=True,
            storage=storage,
        )
    )

    assert result.status == "denied"
    assert approvals[0]["approved"] is False


def test_tool_runtime_truncates_long_output():
    root = Path("outputs") / "test_tool_runtime" / uuid4().hex[:8]
    target = Path("outputs/runtime_long.txt")
    target.parent.mkdir(exist_ok=True)
    target.write_text("x" * 5000, encoding="utf-8")
    storage = Storage(root / "agent.sqlite")

    result, _ = ToolRuntime(root / "artifacts").run(
        ToolRunRequest(
            name="read_file",
            args={"path": "outputs/runtime_long.txt"},
            context=ToolContext("sess", "step", Path("outputs")),
            session_id="sess",
            step_id="step",
            enabled_tools={"read_file"},
            permission_rules=[PermissionRule(permission="read", pattern="*", action="allow")],
            storage=storage,
        )
    )

    assert result.truncated
    assert result.raw_output_path


def test_tool_runtime_repairs_arguments_from_prior_tool_results():
    root = Path("outputs") / "test_tool_runtime" / uuid4().hex[:8]
    storage = Storage(root / "agent.sqlite")
    captured = {}

    def execute(args, context):
        captured["args"] = dict(args)
        return ToolResult(title="shared", output="ok", metadata={"args": dict(args)})

    register_tool(
        ToolSpec(
            name="notes_share_runtime_test",
            description="Share a note with participants.",
            parameters={"type": "object", "properties": {"note_id": {"type": "string"}, "recipients": {"type": "array"}}, "required": ["note_id", "recipients"]},
            output_schema=OUTPUT_SCHEMA,
            permissions=[],
            danger_level="confirm",
            examples=[],
            timeout_seconds=5,
            truncate_policy={"max_chars": 4000},
            execute=execute,
            argument_provenance_rules=[
                ArgumentProvenanceRule(
                    target_tool="notes_share_runtime_test",
                    target_arg="recipients",
                    source_tool="notes_get",
                    source_field="participants",
                    source_filter={"note_id": "note_001"},
                    match_mode="set_equals",
                )
            ],
            provenance_policy="repair",
        ),
        replace=True,
    )

    result, _ = ToolRuntime(root / "artifacts").run(
        ToolRunRequest(
            name="notes_share_runtime_test",
            args={"note_id": "note_001", "recipients": ["attendees_from_note_001"]},
            context=ToolContext("sess", "step", Path("outputs")),
            session_id="sess",
            step_id="step",
            enabled_tools={"notes_share_runtime_test"},
            storage=storage,
            prior_tool_results=[
                {
                    "tool": "notes_get",
                    "status": "success",
                    "result": '{"note_id":"note_001","participants":["Manager Zhang","Li Ming"]}',
                }
            ],
        )
    )

    assert result.status == "success"
    assert captured["args"]["recipients"] == ["Manager Zhang", "Li Ming"]
    assert result.metadata["effective_args"]["recipients"] == ["Manager Zhang", "Li Ming"]
    assert any(item["type"] == "tool.argument_provenance.repaired" for item in storage.list_events("sess"))


def test_tool_runtime_requires_argument_provenance():
    root = Path("outputs") / "test_tool_runtime" / uuid4().hex[:8]
    storage = Storage(root / "agent.sqlite")

    def execute(args, context):
        return ToolResult(title="article", output="should not execute")

    register_tool(
        ToolSpec(
            name="kb_get_article_runtime_test",
            description="Get a KB article.",
            parameters={"type": "object", "properties": {"article_id": {"type": "string"}}, "required": ["article_id"]},
            output_schema=OUTPUT_SCHEMA,
            permissions=[],
            danger_level="safe",
            examples=[],
            timeout_seconds=5,
            truncate_policy={"max_chars": 4000},
            execute=execute,
            argument_provenance_rules=[
                ArgumentProvenanceRule(
                    target_tool="kb_get_article_runtime_test",
                    target_arg="article_id",
                    source_tool="kb_search",
                    source_field="articles.article_id",
                    match_mode="in",
                    failure_action="deny",
                )
            ],
            provenance_policy="require",
        ),
        replace=True,
    )

    result, _ = ToolRuntime(root / "artifacts").run(
        ToolRunRequest(
            name="kb_get_article_runtime_test",
            args={"article_id": "kb_999"},
            context=ToolContext("sess", "step", Path("outputs")),
            session_id="sess",
            step_id="step",
            enabled_tools={"kb_get_article_runtime_test"},
            storage=storage,
            prior_tool_results=[
                {
                    "tool": "kb_search",
                    "status": "success",
                    "result": '{"articles":[{"article_id":"kb_001"}]}',
                }
            ],
        )
    )

    assert result.status == "invalid_args"
    assert "unverified arguments" in result.error


def test_write_file_records_provenance_warning_for_non_output_path():
    root = Path("outputs") / "test_tool_runtime" / uuid4().hex[:8]
    storage = Storage(root / "agent.sqlite")

    result, _ = ToolRuntime(root / "artifacts").run(
        ToolRunRequest(
            name="write_file",
            args={"path": "outputs/provenance_warning.md", "content": "ok"},
            context=ToolContext("sess", "step", Path("outputs")),
            session_id="sess",
            step_id="step",
            enabled_tools={"write_file"},
            permission_rules=[PermissionRule(permission="write", pattern="*", action="allow")],
            storage=storage,
        )
    )

    assert result.status == "success"
    assert result.metadata["argument_provenance"]["passed"] is True
