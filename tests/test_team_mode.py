from pathlib import Path
from uuid import uuid4

from src.config import AppConfig, TeamModeConfig, get_team_spec
from src.runtime import AgentRuntime, RuntimeOptions
from src.storage import Storage
from src.team import TeamRuntime
from src.tool_runtime import ToolRunRequest, ToolRuntime
from src.tools.registry import ToolContext, get_tool_names


def _root() -> Path:
    return Path("outputs") / "test_team_mode" / uuid4().hex[:8]


def test_team_config_defaults_and_specs():
    config = AppConfig()

    assert config.team_mode.enabled is False
    assert get_team_spec("default-dev-team").lead.name == "coordinator"
    assert len(get_team_spec("default-dev-team").members) == 4


def test_team_runtime_rejects_too_many_members():
    root = _root()
    config = AppConfig(
        output_dir=str(root / "outputs"),
        checkpoint_path=str(root / "checkpoints.sqlite"),
        team_mode=TeamModeConfig(max_members=1),
    )
    runtime = TeamRuntime(config, storage=Storage(root / "agent.sqlite"))

    try:
        runtime.run("task", "default-dev-team", auto_approve=True, no_memory=True)
    except ValueError as exc:
        assert "too many members" in str(exc)
    else:
        raise AssertionError("expected max member validation")


def test_team_storage_records_tasks_and_messages():
    root = _root()
    storage = Storage(root / "agent.sqlite")
    storage.create_team("team_demo", "demo", "desc", {"name": "demo"})
    storage.create_team_run("run_demo", "team_demo", "sess_demo", "task")
    storage.add_team_member("member_1", "run_demo", "planner", "planner", "plan", "sess_planner")
    storage.add_team_task("task_1", "run_demo", "Plan", "Make a plan", "planner")
    storage.add_team_message("msg_1", "run_demo", "lead", "planner", {"content": "hello"})

    assert storage.get_team_run("run_demo")["status"] == "running"
    assert storage.list_team_members("run_demo")[0]["name"] == "planner"
    assert storage.list_team_tasks("run_demo")[0]["title"] == "Plan"
    assert storage.list_team_messages("run_demo")[0]["sender"] == "lead"


def test_team_tools_are_registered_and_validate_payload_length():
    root = _root()
    context = ToolContext("sess", "step", root)
    storage = Storage(root / "agent.sqlite")
    storage.create_team_run("run_demo", "team_demo", "sess_demo", "task")

    assert "team_create" in get_tool_names()
    result, _ = ToolRuntime().run(
        ToolRunRequest(
            name="team_send_message",
            args={"team_run_id": "run_demo", "sender": "a", "recipient": "b", "content": "x" * 12001},
            context=context,
            session_id="sess",
            step_id="step",
            enabled_tools={"team_send_message"},
            auto_approve=True,
            storage=storage,
        )
    )

    assert result.status == "invalid_args"
    assert "too long" in (result.error or "")


def test_team_runtime_runs_offline_members():
    root = _root()
    config = AppConfig(output_dir=str(root / "outputs"), checkpoint_path=str(root / "checkpoints.sqlite"))
    storage = Storage(root / "agent.sqlite")
    runtime = AgentRuntime(config, llm=None, storage=storage)

    result = runtime.run(RuntimeOptions(task="Analyze this project", team="research-build", auto_approve=True, no_memory=True))

    assert result.status == "completed"
    assert result.team["team_run_id"]
    assert len(result.team["tasks"]) == 2
    assert all(item["status"] == "completed" for item in result.team["tasks"])
    assert all(item["assigned_to"] != "builder" for item in result.team["tasks"])
    assert "## 当前项目结构分析" in result.final_answer
    assert "## 优化建议" in result.final_answer
    assert "Member results" not in result.final_answer
    assert "证据：ev_" in result.final_answer


def test_team_create_deny_is_not_bypassed_by_yes():
    root = _root()
    config = AppConfig(
        output_dir=str(root / "outputs"),
        checkpoint_path=str(root / "checkpoints.sqlite"),
        permissions=[{"permission": "team:create", "pattern": "*", "action": "deny"}],
    )
    runtime = TeamRuntime(config, storage=Storage(root / "agent.sqlite"))

    try:
        runtime.run("task", "research-build", auto_approve=True, no_memory=True)
    except PermissionError:
        assert True
    else:
        raise AssertionError("deny permission should not be bypassed")


def test_team_llm_synthesis_policy_uses_llm_for_normal_tasks() -> None:
    root = _root()
    runtime = TeamRuntime(AppConfig(output_dir=str(root / "outputs")), llm=object(), storage=Storage(root / "agent.sqlite"))

    assert runtime._should_use_llm_synthesis("分析当前项目结构并给出优化建议", [{"status": "completed"}]) is True


def test_team_llm_synthesis_policy_skips_benchmarks_and_claw() -> None:
    root = _root()
    runtime = TeamRuntime(AppConfig(output_dir=str(root / "outputs")), llm=object(), storage=Storage(root / "agent.sqlite"))

    assert runtime._should_use_llm_synthesis("Claw-Eval task: summarize meeting notes", [{"status": "completed"}]) is False
    assert runtime._should_use_llm_synthesis("根据 DeepEval benchmark 结果分析失败原因", [{"status": "completed"}]) is False
    assert runtime._should_use_llm_synthesis("普通任务", [{"assignment_strategy": "claw_baseline"}]) is False


def test_team_llm_synthesis_policy_respects_config_flag() -> None:
    root = _root()
    config = AppConfig(output_dir=str(root / "outputs"), team_mode=TeamModeConfig(llm_synthesis_enabled=False))
    runtime = TeamRuntime(config, llm=object(), storage=Storage(root / "agent.sqlite"))

    assert runtime._should_use_llm_synthesis("分析当前项目结构并给出优化建议", [{"status": "completed"}]) is False
