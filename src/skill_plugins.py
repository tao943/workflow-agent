import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.config import PermissionRule, SkillRuntimeConfig
from src.tools.registry import OUTPUT_SCHEMA, ToolContext, ToolResult, ToolSpec


SUPPORTED_PERMISSION_PREFIXES = {
    "read",
    "write",
    "rag",
    "web",
    "shell",
    "external_directory",
    "team:create",
    "team:message",
    "team:task",
    "team:delete",
    "team:external_worktree",
    "skill",
    "skill_code",
    "network",
    "dependency_install",
    "mcp_server",
    "mcp_tool",
    "mcp_resource",
}


@dataclass(frozen=True)
class SkillToolManifest:
    name: str
    description: str
    parameters: dict[str, Any]
    permissions: list[str] = field(default_factory=list)
    timeout_seconds: int = 10


@dataclass(frozen=True)
class SkillPackage:
    name: str
    version: str
    description: str
    runtime: str
    image: str | None
    entrypoint: Path
    root: Path
    skill_md: Path | None
    system_prompt: str
    triggers: list[str]
    tools: list[SkillToolManifest]
    permissions: list[str]
    filesystem: dict[str, Any]
    network: bool
    dependencies: list[str]
    manifest_path: Path


@dataclass(frozen=True)
class SkillValidationError:
    path: str
    message: str


@dataclass(frozen=True)
class SkillCatalog:
    packages: dict[str, SkillPackage] = field(default_factory=dict)
    errors: list[SkillValidationError] = field(default_factory=list)

    def dynamic_tool_names(self) -> list[str]:
        names: list[str] = []
        for package in self.packages.values():
            names.extend(tool.name for tool in package.tools)
        return names


class SkillRunner(Protocol):
    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        ...


class SkillPluginRuntime:
    def __init__(self, package: SkillPackage, tool: SkillToolManifest, config: SkillRuntimeConfig | None = None) -> None:
        self.package = package
        self.tool = tool
        self.config = config or SkillRuntimeConfig()

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        runner: SkillRunner
        if self.package.runtime == "docker":
            runner = DockerSkillRunner(self.package, self.tool, self.config)
        else:
            runner = PythonSubprocessSkillRunner(self.package, self.tool)
        return runner.execute(args, context)


class PythonSubprocessSkillRunner:
    def __init__(self, package: SkillPackage, tool: SkillToolManifest) -> None:
        self.package = package
        self.tool = tool

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        output_dir = Path(context.output_dir) / "skills" / self.package.name
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = _skill_payload(self.tool.name, args, context, output_dir)
        try:
            completed = subprocess.run(
                [sys.executable, str(self.package.entrypoint)],
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self.tool.timeout_seconds,
                cwd=str(self.package.root),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                title=f"{self.tool.name} timeout",
                output="",
                error=f"The {self.tool.name} skill tool timed out after {self.tool.timeout_seconds} seconds.",
                status="timeout",
                metadata={"skill": self.package.name, "tool": self.tool.name},
            )

        return _result_from_process(self.package.name, self.tool.name, completed, "Skill process")


class DockerSkillRunner:
    def __init__(self, package: SkillPackage, tool: SkillToolManifest, config: SkillRuntimeConfig) -> None:
        self.package = package
        self.tool = tool
        self.config = config

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        if not self.config.docker_enabled:
            return ToolResult(
                title=f"{self.tool.name} denied",
                output="",
                error="Docker skill runtime is disabled by configuration.",
                status="denied",
                metadata={"skill": self.package.name, "tool": self.tool.name, "code": "docker_disabled"},
            )
        image = self.package.image or self.config.docker_image
        if image not in self.config.docker_allowed_images:
            return ToolResult(
                title=f"{self.tool.name} denied",
                output="",
                error=f"Docker image is not allowlisted: {image}",
                status="denied",
                metadata={"skill": self.package.name, "tool": self.tool.name, "image": image, "code": "image_not_allowed"},
            )

        output_dir = Path(context.output_dir) / "skills" / self.package.name
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = _skill_payload(self.tool.name, args, context, output_dir)
        command = self._docker_command(image, output_dir)
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self.tool.timeout_seconds,
                env=_docker_env(),
                check=False,
            )
        except FileNotFoundError as exc:
            return ToolResult(
                title=f"{self.tool.name} docker unavailable",
                output="",
                error=f"Docker CLI is not available: {exc}",
                status="error",
                metadata={"skill": self.package.name, "tool": self.tool.name, "code": "docker_unavailable"},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                title=f"{self.tool.name} timeout",
                output="",
                error=f"The {self.tool.name} docker skill tool timed out after {self.tool.timeout_seconds} seconds.",
                status="timeout",
                metadata={"skill": self.package.name, "tool": self.tool.name, "image": image},
            )
        stderr = completed.stderr.strip()
        if completed.returncode != 0 and _looks_like_docker_unavailable(stderr):
            return ToolResult(
                title=f"{self.tool.name} docker unavailable",
                output="",
                error=f"Docker is not available for skill execution: {stderr[:1000]}",
                status="error",
                metadata={"skill": self.package.name, "tool": self.tool.name, "image": image, "code": "docker_unavailable"},
            )
        result = _result_from_process(self.package.name, self.tool.name, completed, "Docker skill process")
        result.metadata.setdefault("image", image)
        return result

    def _docker_command(self, image: str, output_dir: Path) -> list[str]:
        command = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            self.config.docker_network,
        ]
        if self.config.docker_read_only:
            command.append("--read-only")
        for capability in self.config.docker_cap_drop:
            command.extend(["--cap-drop", capability])
        for item in self.config.docker_security_opt:
            command.extend(["--security-opt", item])
        command.extend(
            [
                "--memory",
                self.config.docker_memory,
                "--cpus",
                self.config.docker_cpus,
                "--pids-limit",
                str(self.config.docker_pids_limit),
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=64m",
                "-v",
                f"{self.package.root.resolve()}:/skill:ro",
                "-v",
                f"{output_dir.resolve()}:/outputs:rw",
                "-w",
                "/skill",
                image,
                "python",
                f"/skill/{self.package.entrypoint.name}",
            ]
        )
        return command


def _skill_payload(tool_name: str, args: dict[str, Any], context: ToolContext, output_dir: Path) -> dict[str, Any]:
    return {
        "tool": tool_name,
        "args": args,
        "context": {
            "session_id": context.session_id,
            "step_id": context.step_id,
            "workspace": str(Path.cwd().resolve()),
            "output_dir": str(output_dir.resolve()),
        },
    }


def _result_from_process(skill_name: str, tool_name: str, completed: subprocess.CompletedProcess[str], process_name: str) -> ToolResult:
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        return ToolResult(
            title=f"{tool_name} failed",
            output="",
            error=f"{process_name} exited with code {completed.returncode}: {stderr[:1000]}",
            status="error",
            metadata={"skill": skill_name, "tool": tool_name, "stderr": stderr[:1000], "code": "process_failed"},
        )
    try:
        data = json.loads(stdout or "{}")
    except json.JSONDecodeError as exc:
        return ToolResult(
            title=f"{tool_name} invalid output",
            output=stdout[:1000],
            error=f"{process_name} returned invalid JSON: {exc}",
            status="error",
            metadata={"skill": skill_name, "tool": tool_name, "stderr": stderr[:1000], "code": "invalid_json"},
        )
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {"skill": skill_name, "tool": tool_name}
    return ToolResult(
        title=str(data.get("title") or tool_name),
        output=str(data.get("output") or ""),
        metadata=metadata,
        error=str(data["error"]) if data.get("error") else None,
        attachments=[str(item) for item in data.get("attachments", [])] if isinstance(data.get("attachments"), list) else [],
        status=data.get("status") if data.get("status") in {"success", "error", "denied", "invalid_args", "timeout"} else "success",
        display_output=str(data["display_output"]) if data.get("display_output") is not None else None,
        raw_output_path=str(data["raw_output_path"]) if data.get("raw_output_path") else None,
        truncated=bool(data.get("truncated", False)),
    )


def load_skill_catalog(config: SkillRuntimeConfig | None = None, workspace: Path | None = None) -> SkillCatalog:
    config = config or SkillRuntimeConfig()
    workspace = (workspace or Path.cwd()).resolve()
    if not config.loading_enabled:
        return SkillCatalog()
    packages: dict[str, SkillPackage] = {}
    errors: list[SkillValidationError] = []
    for raw_path in config.paths:
        root = (workspace / raw_path).resolve()
        if not _is_inside(root, workspace) and not config.allow_global_skills:
            errors.append(SkillValidationError(str(root), "Global skill paths are disabled."))
            continue
        if not root.exists():
            continue
        for manifest_path in root.glob("*/skill.json"):
            package, error = _load_package(manifest_path.resolve(), config, workspace)
            if error:
                errors.append(error)
                continue
            assert package is not None
            if package.name in packages:
                errors.append(SkillValidationError(str(manifest_path), f"Duplicate skill name: {package.name}"))
                continue
            packages[package.name] = package
    return SkillCatalog(packages=packages, errors=errors)


def doctor_sandbox(config: SkillRuntimeConfig | None = None) -> dict[str, Any]:
    config = config or SkillRuntimeConfig()
    checks: list[dict[str, Any]] = []
    version = _run_probe(["docker", "--version"], timeout=5)
    checks.append({"name": "docker_cli", **version})
    info = _run_probe(["docker", "info", "--format", "{{json .ServerVersion}}"], timeout=10)
    checks.append({"name": "docker_daemon", **info})
    images: list[dict[str, Any]] = []
    for image in config.docker_allowed_images:
        probe = _run_probe(["docker", "image", "inspect", image], timeout=10)
        images.append({"image": image, **probe})
    checks.append({"name": "docker_images", "ok": any(item["ok"] for item in images), "items": images})
    ok = version["ok"] and info["ok"]
    return {
        "ok": ok,
        "code": "ok" if ok else "docker_unavailable",
        "docker_enabled": config.docker_enabled,
        "allowed_images": config.docker_allowed_images,
        "checks": checks,
    }


def package_to_tool_specs(package: SkillPackage, config: SkillRuntimeConfig | None = None) -> list[ToolSpec]:
    config = config or SkillRuntimeConfig()
    if not config.code_enabled:
        return []
    specs: list[ToolSpec] = []
    for tool in package.tools:
        runner = SkillPluginRuntime(package, tool, config)
        permissions = list(dict.fromkeys([*tool.permissions, f"skill:{package.name}", "skill_code:execute"]))
        specs.append(
            ToolSpec(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                output_schema=OUTPUT_SCHEMA,
                permissions=permissions,
                danger_level="confirm",
                examples=[],
                timeout_seconds=tool.timeout_seconds,
                truncate_policy={"max_chars": 4000},
                execute=runner.execute,
            )
        )
    return specs


def package_permission_rules(package: SkillPackage, action: str = "ask") -> list[PermissionRule]:
    rules = [
        PermissionRule(permission="skill", pattern=package.name, action=action),  # type: ignore[arg-type]
        PermissionRule(permission="skill_code", pattern="execute", action=action),  # type: ignore[arg-type]
    ]
    for permission in package.permissions:
        permission_name, pattern = permission.rsplit(":", 1)
        rules.append(PermissionRule(permission=permission_name, pattern=pattern, action="ask"))
    return rules


def _load_package(manifest_path: Path, config: SkillRuntimeConfig, workspace: Path) -> tuple[SkillPackage | None, SkillValidationError | None]:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, SkillValidationError(str(manifest_path), f"Cannot read manifest: {exc}")

    root = manifest_path.parent.resolve()
    name = str(data.get("name") or "").strip()
    if not name:
        return None, SkillValidationError(str(manifest_path), "Missing skill name.")
    if not _valid_identifier(name):
        return None, SkillValidationError(str(manifest_path), f"Invalid skill name: {name}")
    runtime = str(data.get("runtime") or "").strip()
    if runtime not in config.allowed_runtimes:
        return None, SkillValidationError(str(manifest_path), f"Unsupported runtime: {runtime}")
    image = str(data.get("image") or config.docker_image).strip() if runtime == "docker" else None
    if runtime == "docker":
        if not config.docker_enabled:
            return None, SkillValidationError(str(manifest_path), "Docker runtime is disabled.")
        if image not in config.docker_allowed_images:
            return None, SkillValidationError(str(manifest_path), f"Docker image is not allowlisted: {image}")
    if data.get("dependencies"):
        return None, SkillValidationError(str(manifest_path), "Skill dependencies are not supported in v1.")
    if data.get("network"):
        return None, SkillValidationError(str(manifest_path), "Network access is disabled for skill plugins in v1.")

    entrypoint = (root / str(data.get("entrypoint") or "")).resolve()
    if not entrypoint.exists() or not entrypoint.is_file():
        return None, SkillValidationError(str(manifest_path), "Entrypoint does not exist.")
    if not _is_inside(entrypoint, root):
        return None, SkillValidationError(str(manifest_path), "Entrypoint must stay inside the skill directory.")

    filesystem = data.get("filesystem") if isinstance(data.get("filesystem"), dict) else {}
    for value in filesystem.get("write", []) if isinstance(filesystem.get("write"), list) else []:
        if not str(value).replace("\\", "/").startswith(f"outputs/skills/{name}"):
            return None, SkillValidationError(str(manifest_path), "Skill write paths must stay under outputs/skills/<skill_name>.")

    raw_tools = data.get("tools")
    if not isinstance(raw_tools, list) or not raw_tools:
        return None, SkillValidationError(str(manifest_path), "Skill must declare at least one tool.")
    tools: list[SkillToolManifest] = []
    seen_tools: set[str] = set()
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            return None, SkillValidationError(str(manifest_path), "Tool entries must be objects.")
        tool_name = str(raw_tool.get("name") or "").strip()
        if not tool_name.startswith(f"{name}.") or tool_name in seen_tools:
            return None, SkillValidationError(str(manifest_path), f"Tool name must be unique and namespaced as {name}.*")
        seen_tools.add(tool_name)
        permissions = [str(item) for item in raw_tool.get("permissions", []) if isinstance(item, str)]
        for permission in permissions:
            if not _valid_permission(permission):
                return None, SkillValidationError(str(manifest_path), f"Unknown permission: {permission}")
        parameters = raw_tool.get("parameters") if isinstance(raw_tool.get("parameters"), dict) else {}
        tools.append(
            SkillToolManifest(
                name=tool_name,
                description=str(raw_tool.get("description") or tool_name),
                parameters=parameters,
                permissions=permissions,
                timeout_seconds=int(raw_tool.get("timeout_seconds") or 10),
            )
        )

    activation = data.get("activation") if isinstance(data.get("activation"), dict) else {}
    skill_md = root / "SKILL.md"
    system_prompt = skill_md.read_text(encoding="utf-8") if skill_md.exists() else str(data.get("description") or name)
    permissions = [str(item) for item in data.get("permissions", []) if isinstance(item, str)]
    for permission in permissions:
        if not _valid_permission(permission):
            return None, SkillValidationError(str(manifest_path), f"Unknown permission: {permission}")

    return (
        SkillPackage(
            name=name,
            version=str(data.get("version") or "0.1.0"),
            description=str(data.get("description") or name),
            runtime=runtime,
            image=image,
            entrypoint=entrypoint,
            root=root,
            skill_md=skill_md if skill_md.exists() else None,
            system_prompt=system_prompt,
            triggers=[str(item) for item in activation.get("triggers", []) if isinstance(item, str)],
            tools=tools,
            permissions=permissions,
            filesystem=filesystem,
            network=bool(data.get("network", False)),
            dependencies=[str(item) for item in data.get("dependencies", []) if isinstance(item, str)],
            manifest_path=manifest_path,
        ),
        None,
    )


def _valid_identifier(value: str) -> bool:
    return value.replace("_", "").replace("-", "").isalnum()


def _valid_permission(value: str) -> bool:
    if ":" not in value:
        return False
    permission_name, _ = value.rsplit(":", 1)
    return permission_name in SUPPORTED_PERMISSION_PREFIXES


def _run_probe(command: list[str], timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False, env=_docker_env())
    except FileNotFoundError as exc:
        return {"ok": False, "stdout": "", "stderr": str(exc), "returncode": None}
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": f"Command timed out after {timeout} seconds.", "returncode": None}
    return {
        "ok": completed.returncode == 0,
        "stdout": completed.stdout.strip()[:1000],
        "stderr": completed.stderr.strip()[:1000],
        "returncode": completed.returncode,
    }


def _looks_like_docker_unavailable(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(
        marker in lowered
        for marker in (
            "failed to connect to the docker api",
            "cannot connect to the docker daemon",
            "is the docker daemon running",
            "docker_engine",
        )
    )


def _docker_env() -> dict[str, str]:
    env = dict(os.environ)
    docker_config = Path("outputs") / "docker-config"
    docker_config.mkdir(parents=True, exist_ok=True)
    env["DOCKER_CONFIG"] = str(docker_config.resolve())
    return env


def _is_inside(target: Path, root: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False
