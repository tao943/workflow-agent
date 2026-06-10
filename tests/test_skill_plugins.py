import json
import subprocess
from pathlib import Path
from uuid import uuid4

from src.config import AppConfig, PermissionRule, SkillRuntimeConfig
from src.skill_plugins import load_skill_catalog
from src.skills import build_skill_registry, load_and_register_project_skills, tools_for_skills
from src.storage import Storage
from src.tool_runtime import ToolRuntime, ToolRunRequest
from src.tools.registry import ToolContext


def _case_root() -> Path:
    root = Path("outputs") / "test_skill_plugins" / uuid4().hex[:8]
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_skill(
    root: Path,
    name: str = "demo_skill",
    body: str | None = None,
    timeout_seconds: int = 5,
    runtime: str = "python-subprocess",
    image: str | None = None,
    network: bool = False,
) -> None:
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("Use this demo skill for tests.", encoding="utf-8")
    (skill_dir / "skill.json").write_text(
        json.dumps(
            {
                "name": name,
                "version": "0.1.0",
                "description": "test skill",
                "runtime": runtime,
                **({"image": image} if image else {}),
                "entrypoint": "main.py",
                "activation": {"triggers": ["demo"]},
                "tools": [
                    {
                        "name": f"{name}.echo",
                        "description": "echo text",
                        "parameters": {"text": "Text"},
                        "permissions": [f"skill:{name}"],
                        "timeout_seconds": timeout_seconds,
                    }
                ],
                "filesystem": {"write": [f"outputs/skills/{name}"]},
                "network": network,
                "dependencies": [],
            }
        ),
        encoding="utf-8",
    )
    (skill_dir / "main.py").write_text(
        body
        or (
            "import json, sys\n"
            "payload=json.loads(sys.stdin.read() or '{}')\n"
            "text=(payload.get('args') or {}).get('text','')\n"
            "print(json.dumps({'status':'success','title':'ok','output':text,'metadata':{'ok': True}}))\n"
        ),
        encoding="utf-8",
    )


def test_project_skill_loads_and_registers_tool(monkeypatch):
    root = _case_root()
    _write_skill(root)
    monkeypatch.chdir(root)
    config = AppConfig(skill_runtime=SkillRuntimeConfig(paths=["skills"]))

    catalog = load_and_register_project_skills(config)
    registry = build_skill_registry(catalog, config.skill_runtime)

    assert "demo_skill" in catalog.packages
    assert "demo_skill.echo" in tools_for_skills(["demo_skill"], registry)


def test_skill_manifest_unknown_runtime_is_invalid():
    root = _case_root()
    _write_skill(root)
    manifest = root / "skills" / "demo_skill" / "skill.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["runtime"] = "shell"
    manifest.write_text(json.dumps(data), encoding="utf-8")

    catalog = load_skill_catalog(SkillRuntimeConfig(paths=["skills"]), workspace=root)

    assert catalog.errors
    assert "Unsupported runtime" in catalog.errors[0].message


def test_skill_manifest_rejects_path_escape():
    root = _case_root()
    _write_skill(root)
    manifest = root / "skills" / "demo_skill" / "skill.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["entrypoint"] = "../escape.py"
    manifest.write_text(json.dumps(data), encoding="utf-8")

    catalog = load_skill_catalog(SkillRuntimeConfig(paths=["skills"]), workspace=root)

    assert catalog.errors


def test_skill_tool_executes_through_tool_runtime(monkeypatch):
    root = _case_root()
    _write_skill(root)
    monkeypatch.chdir(root)
    config = AppConfig(skill_runtime=SkillRuntimeConfig(paths=["skills"]))
    load_and_register_project_skills(config)
    storage = Storage(root / "outputs" / "agent.sqlite")

    result, approvals = ToolRuntime(root / "outputs" / "artifacts").run(
        ToolRunRequest(
            name="demo_skill.echo",
            args={"text": "hello"},
            context=ToolContext("sess", "step", root / "outputs"),
            session_id="sess",
            step_id="step",
            enabled_tools={"demo_skill.echo"},
            permission_rules=[PermissionRule(permission="skill", pattern="demo_skill", action="allow"), PermissionRule(permission="skill_code", pattern="execute", action="allow")],
            storage=storage,
        )
    )

    assert result.status == "success"
    assert result.output == "hello"
    assert approvals
    assert storage.list_tool_calls("sess")[0]["tool_name"] == "demo_skill.echo"


def test_skill_tool_deny_beats_auto_approve(monkeypatch):
    root = _case_root()
    _write_skill(root)
    monkeypatch.chdir(root)
    config = AppConfig(skill_runtime=SkillRuntimeConfig(paths=["skills"]))
    load_and_register_project_skills(config)

    result, approvals = ToolRuntime(root / "outputs" / "artifacts").run(
        ToolRunRequest(
            name="demo_skill.echo",
            args={"text": "hello"},
            context=ToolContext("sess", "step", root / "outputs"),
            session_id="sess",
            step_id="step",
            enabled_tools={"demo_skill.echo"},
            permission_rules=[PermissionRule(permission="skill", pattern="demo_skill", action="deny")],
            auto_approve=True,
            storage=Storage(root / "outputs" / "agent.sqlite"),
        )
    )

    assert result.status == "denied"
    assert approvals[0]["approved"] is False


def test_skill_tool_timeout_is_structured(monkeypatch):
    root = _case_root()
    _write_skill(root, body="import time\ntime.sleep(2)\n", timeout_seconds=1)
    monkeypatch.chdir(root)
    config = AppConfig(skill_runtime=SkillRuntimeConfig(paths=["skills"]))
    load_and_register_project_skills(config)

    result, _ = ToolRuntime(root / "outputs" / "artifacts").run(
        ToolRunRequest(
            name="demo_skill.echo",
            args={"text": "hello"},
            context=ToolContext("sess", "step", root / "outputs"),
            session_id="sess",
            step_id="step",
            enabled_tools={"demo_skill.echo"},
            permission_rules=[PermissionRule(permission="skill", pattern="demo_skill", action="allow"), PermissionRule(permission="skill_code", pattern="execute", action="allow")],
            storage=Storage(root / "outputs" / f"{uuid4().hex}.sqlite"),
        )
    )

    assert result.status == "timeout"


def test_docker_skill_manifest_loads_when_allowed():
    root = _case_root()
    _write_skill(root, runtime="docker", image="python:3.13-slim")

    catalog = load_skill_catalog(SkillRuntimeConfig(paths=["skills"], docker_allowed_images=["python:3.13-slim"]), workspace=root)

    assert not catalog.errors
    assert catalog.packages["demo_skill"].runtime == "docker"
    assert catalog.packages["demo_skill"].image == "python:3.13-slim"


def test_docker_disabled_rejects_manifest():
    root = _case_root()
    _write_skill(root, runtime="docker", image="python:3.13-slim")

    catalog = load_skill_catalog(SkillRuntimeConfig(paths=["skills"], docker_enabled=False, docker_allowed_images=["python:3.13-slim"]), workspace=root)

    assert catalog.errors
    assert "disabled" in catalog.errors[0].message


def test_docker_image_not_allowlisted_rejects_manifest():
    root = _case_root()
    _write_skill(root, runtime="docker", image="evil:latest")

    catalog = load_skill_catalog(SkillRuntimeConfig(paths=["skills"], docker_allowed_images=["python:3.13-slim"]), workspace=root)

    assert catalog.errors
    assert "allowlisted" in catalog.errors[0].message


def test_docker_network_true_rejects_manifest():
    root = _case_root()
    _write_skill(root, runtime="docker", image="python:3.13-slim", network=True)

    catalog = load_skill_catalog(SkillRuntimeConfig(paths=["skills"], docker_allowed_images=["python:3.13-slim"]), workspace=root)

    assert catalog.errors
    assert "Network access" in catalog.errors[0].message


def test_docker_skill_executes_with_sandbox_flags(monkeypatch):
    root = _case_root()
    _write_skill(root, runtime="docker", image="python:3.13-slim")
    monkeypatch.chdir(root)
    config = AppConfig(skill_runtime=SkillRuntimeConfig(paths=["skills"], docker_allowed_images=["python:3.13-slim"]))
    load_and_register_project_skills(config)
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["payload"] = json.loads(kwargs["input"])
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"status": "success", "title": "ok", "output": "docker ok", "metadata": {"runner": "docker"}}), stderr="")

    monkeypatch.setattr("src.skill_plugins.subprocess.run", fake_run)

    result, _ = ToolRuntime(root / "outputs" / "artifacts").run(
        ToolRunRequest(
            name="demo_skill.echo",
            args={"text": "hello"},
            context=ToolContext("sess", "step", root / "outputs"),
            session_id="sess",
            step_id="step",
            enabled_tools={"demo_skill.echo"},
            permission_rules=[PermissionRule(permission="skill", pattern="demo_skill", action="allow"), PermissionRule(permission="skill_code", pattern="execute", action="allow")],
            storage=Storage(root / "outputs" / "agent.sqlite"),
        )
    )

    command = captured["command"]
    assert result.status == "success"
    assert result.output == "docker ok"
    assert "-i" in command
    assert "--network" in command and "none" in command
    assert "--read-only" in command
    assert "--cap-drop" in command and "ALL" in command
    assert "--security-opt" in command and "no-new-privileges:true" in command
    assert "-v" in command
    assert captured["payload"]["args"] == {"text": "hello"}
    assert captured["payload"]["context"]["output_dir"].endswith("demo_skill")


def test_docker_unavailable_returns_structured_error(monkeypatch):
    root = _case_root()
    _write_skill(root, runtime="docker", image="python:3.13-slim")
    monkeypatch.chdir(root)
    config = AppConfig(skill_runtime=SkillRuntimeConfig(paths=["skills"], docker_allowed_images=["python:3.13-slim"]))
    load_and_register_project_skills(config)

    def fake_run(command, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr("src.skill_plugins.subprocess.run", fake_run)

    result, _ = ToolRuntime(root / "outputs" / "artifacts").run(
        ToolRunRequest(
            name="demo_skill.echo",
            args={"text": "hello"},
            context=ToolContext("sess", "step", root / "outputs"),
            session_id="sess",
            step_id="step",
            enabled_tools={"demo_skill.echo"},
            permission_rules=[PermissionRule(permission="skill", pattern="demo_skill", action="allow"), PermissionRule(permission="skill_code", pattern="execute", action="allow")],
            storage=Storage(root / "outputs" / "agent.sqlite"),
        )
    )

    assert result.status == "error"
    assert result.metadata["code"] == "docker_unavailable"


def test_docker_daemon_unavailable_returns_structured_error(monkeypatch):
    root = _case_root()
    _write_skill(root, runtime="docker", image="python:3.13-slim")
    monkeypatch.chdir(root)
    config = AppConfig(skill_runtime=SkillRuntimeConfig(paths=["skills"], docker_allowed_images=["python:3.13-slim"]))
    load_and_register_project_skills(config)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="failed to connect to the docker API at npipe:////./pipe/docker_engine")

    monkeypatch.setattr("src.skill_plugins.subprocess.run", fake_run)

    result, _ = ToolRuntime(root / "outputs" / "artifacts").run(
        ToolRunRequest(
            name="demo_skill.echo",
            args={"text": "hello"},
            context=ToolContext("sess", "step", root / "outputs"),
            session_id="sess",
            step_id="step",
            enabled_tools={"demo_skill.echo"},
            permission_rules=[PermissionRule(permission="skill", pattern="demo_skill", action="allow"), PermissionRule(permission="skill_code", pattern="execute", action="allow")],
            storage=Storage(root / "outputs" / "agent.sqlite"),
        )
    )

    assert result.status == "error"
    assert result.metadata["code"] == "docker_unavailable"


def test_docker_timeout_is_structured(monkeypatch):
    root = _case_root()
    _write_skill(root, runtime="docker", image="python:3.13-slim", timeout_seconds=1)
    monkeypatch.chdir(root)
    config = AppConfig(skill_runtime=SkillRuntimeConfig(paths=["skills"], docker_allowed_images=["python:3.13-slim"]))
    load_and_register_project_skills(config)

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr("src.skill_plugins.subprocess.run", fake_run)

    result, _ = ToolRuntime(root / "outputs" / "artifacts").run(
        ToolRunRequest(
            name="demo_skill.echo",
            args={"text": "hello"},
            context=ToolContext("sess", "step", root / "outputs"),
            session_id="sess",
            step_id="step",
            enabled_tools={"demo_skill.echo"},
            permission_rules=[PermissionRule(permission="skill", pattern="demo_skill", action="allow"), PermissionRule(permission="skill_code", pattern="execute", action="allow")],
            storage=Storage(root / "outputs" / "agent.sqlite"),
        )
    )

    assert result.status == "timeout"
