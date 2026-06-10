import os
from pathlib import Path
from uuid import uuid4

import pytest

from src.config import AppConfig, PermissionRule
from src.evals import EvalHarness
from src.rag_runtime import rag_health
from src.runtime import AgentRuntime
from src.storage import Storage
from src.tool_runtime import ToolRuntime, ToolRunRequest
from src.tools.registry import ToolContext


def _require_real(name: str) -> None:
    if os.getenv(name) != "1":
        pytest.skip(f"Set {name}=1 to run this real integration test.")


@pytest.mark.integration
@pytest.mark.rag
def test_real_rag_health_requires_live_service_when_enabled():
    _require_real("RUN_REAL_RAG")

    health = rag_health("LangGraph checkpoint")

    assert health["ok"], health
    assert health["code"] in {"ok", "no_evidence"}, health


@pytest.mark.integration
@pytest.mark.rag
def test_real_rag_eval_suite_fails_or_passes_from_actual_service_state():
    _require_real("RUN_REAL_RAG")
    output_dir = Path("outputs") / "test_real_evals" / f"rag_{uuid4().hex[:8]}"
    report = EvalHarness(AppConfig(), output_dir=output_dir).run_suite("rag_core_real", offline=True)

    assert report.environment["services"]["rag"]["ok"], report.environment
    assert report.passed, report.summary


@pytest.mark.integration
@pytest.mark.mcp
def test_real_mcp_fetch_tool_call_records_evidence():
    _require_real("RUN_REAL_MCP")
    config = AppConfig(output_dir=str(Path("outputs") / "test_real_mcp" / uuid4().hex[:8]))
    storage = Storage(Path(config.output_dir) / "agent.sqlite")
    runtime = AgentRuntime(config, llm=None, storage=storage)
    session_id = "sess_real_mcp"

    result, _ = ToolRuntime(Path(config.output_dir) / "artifacts").run(
        ToolRunRequest(
            name="mcp.fetch.fetch",
            args={"url": "https://example.com"},
            context=ToolContext(session_id, "step", Path(config.output_dir)),
            session_id=session_id,
            step_id="step",
            enabled_tools={"mcp.fetch.fetch"},
            permission_rules=[
                PermissionRule(permission="mcp_server", pattern="*", action="allow"),
                PermissionRule(permission="mcp_tool", pattern="*", action="allow"),
            ],
            auto_approve=True,
            storage=storage,
        )
    )

    assert result.status == "success", result.error
    assert "Example Domain" in (result.output or result.display_output)
    assert storage.list_tool_calls(session_id)
    assert any(item["type"] == "tool.completed" for item in storage.list_events(session_id))
    assert runtime.mcp_manager.health()


@pytest.mark.integration
@pytest.mark.mcp
def test_real_mcp_eval_suite_uses_fetch_server():
    _require_real("RUN_REAL_MCP")
    output_dir = Path("outputs") / "test_real_evals" / f"mcp_{uuid4().hex[:8]}"
    report = EvalHarness(AppConfig(), output_dir=output_dir).run_suite("mcp_core_real", offline=True)

    assert report.environment["services"]["mcp_fetch"]["ok"], report.environment
    assert report.passed, report.summary


@pytest.mark.integration
@pytest.mark.docker
def test_real_docker_skill_eval_suite_runs_container():
    _require_real("RUN_REAL_DOCKER")
    output_dir = Path("outputs") / "test_real_evals" / f"docker_{uuid4().hex[:8]}"
    report = EvalHarness(AppConfig(), output_dir=output_dir).run_suite("docker_core_real", offline=True)

    assert report.environment["services"]["docker"]["ok"], report.environment
    assert report.passed, report.summary
