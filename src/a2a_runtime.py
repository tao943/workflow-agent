from __future__ import annotations

import os
import hashlib
import mimetypes
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True, repr=False)
class ResolvedA2AAgent:
    role: str
    base_origin: str
    agent_card_url: str
    authorization_header: str
    allow_local_fallback: bool
    workspace_root: str

    def __repr__(self) -> str:
        return f"ResolvedA2AAgent(role={self.role!r}, origin={self.base_origin!r})"


class A2ARuntimeError(RuntimeError):
    def __init__(self, code: str, message: str, retryable: bool = False, details: dict | None = None):
        super().__init__(message)
        self.code, self.retryable, self.details = code, retryable, details or {}


@dataclass(frozen=True)
class RemoteArtifactManifest:
    artifact_id: str
    name: str
    url: str
    mime_type: str
    size_bytes: int
    sha256: str


class A2AArtifactTransport:
    ALLOWED_MIME = {"text/plain", "text/markdown", "application/json", "text/x-diff", "application/zip", "application/octet-stream"}

    def __init__(self, output_dir: str | Path, max_bytes: int = 50 * 1024 * 1024, fetcher=None):
        self.output_dir = Path(output_dir).resolve()
        self.max_bytes = max_bytes
        self.fetcher = fetcher

    def download(self, manifest: RemoteArtifactManifest, authorization_header: str = "") -> dict:
        parsed = urlparse(manifest.url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise A2ARuntimeError("artifact_error", "artifact origin is not allowed")
        if manifest.mime_type not in self.ALLOWED_MIME or manifest.size_bytes > self.max_bytes:
            raise A2ARuntimeError("artifact_error", "artifact size or MIME is not allowed")
        data = self.fetcher(manifest.url, {"Authorization": authorization_header}) if self.fetcher else b""
        if len(data) > self.max_bytes or len(data) != manifest.size_bytes:
            raise A2ARuntimeError("artifact_error", "artifact size mismatch")
        digest = hashlib.sha256(data).hexdigest()
        if digest != manifest.sha256:
            raise A2ARuntimeError("artifact_error", "artifact sha256 mismatch")
        target = resolve_workspace_path(str(self.output_dir), manifest.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return {"artifact_id": manifest.artifact_id, "file_path": str(target), "mime_type": manifest.mime_type, "size_bytes": len(data), "sha256": digest}


class A2AAgentRegistry:
    def __init__(self, config):
        self.config = config

    def resolve(self, role: str) -> ResolvedA2AAgent:
        agent = self.config.agents.get(role)
        if not self.config.enabled or agent is None or not agent.enabled:
            raise A2ARuntimeError("configuration_error", f"A2A role {role!r} is disabled")
        if role == "builder" and agent.allow_local_fallback:
            raise A2ARuntimeError("configuration_error", "Builder local fallback is forbidden")
        parsed = urlparse(agent.agent_card_url)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or not parsed.hostname:
            raise A2ARuntimeError("configuration_error", "invalid A2A endpoint")
        token = os.getenv(agent.token_env) if agent.token_env else None
        if not token:
            raise A2ARuntimeError("configuration_error", "A2A token environment variable is missing")
        origin = f"{parsed.scheme}://{parsed.netloc}"
        return ResolvedA2AAgent(role, origin, agent.agent_card_url, f"Bearer {token}", agent.allow_local_fallback, agent.workspace_root)


class A2AClientRuntime:
    def __init__(self, registry: A2AAgentRegistry, timeout: float = 30.0):
        self.registry, self.timeout = registry, timeout

    def execute(self, request):
        import httpx
        from src.member_execution import MemberExecutionResult
        agent = self.registry.resolve(request.role)
        payload = {"role": request.role, "task": request.description, "team_run_id": request.team_run_id, "task_id": request.logical_task_id, "execution_id": request.execution_id, "idempotency_key": request.idempotency_key, "workspace_root": agent.workspace_root}
        response = httpx.post(agent.base_origin + "/a2a/rest/v1/message:send", headers={"Authorization": agent.authorization_header}, json=payload, timeout=self.timeout)
        response.raise_for_status()
        body = response.json()
        return MemberExecutionResult(request.member_name, request.role, body.get("status", "completed"), str(body.get("result", "")), {"passed": True, "issues": []}, [], body.get("artifacts", []), [], "a2a", remote_context_id=body.get("context_id", ""), remote_task_id=body.get("task_id", ""))


def resolve_workspace_path(workspace_root: str, requested: str) -> Path:
    """Resolve a role file path and reject traversal or symlink escapes."""
    root = Path(workspace_root).expanduser().resolve()
    candidate = (root / requested).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise A2ARuntimeError("workspace_escape", "path escapes workspace root") from exc
    return candidate
