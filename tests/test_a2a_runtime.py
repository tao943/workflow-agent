import pytest
from src.a2a_runtime import A2AAgentRegistry, A2ARuntimeError, resolve_workspace_path, A2AArtifactTransport, RemoteArtifactManifest
import hashlib
from src.config import A2AConfig, A2ARemoteAgentConfig
from src.storage import Storage


def test_storage_round_trips_a2a_task_mapping(tmp_path):
    storage = Storage(tmp_path / "agent.sqlite")
    storage.upsert_a2a_task(team_run_id="run", logical_task_id="review", role="reviewer", business_attempt=0, execution_id="exec-1", idempotency_key="run:review:0", transport_retry_count=1, context_id="ctx", remote_task_id="remote", endpoint="http://127.0.0.1:8103", status="working")
    row = storage.get_a2a_task_by_execution_id("exec-1")
    assert row["remote_task_id"] == "remote"
    assert row["transport_retry_count"] == 1


def test_registry_reads_token_without_exposing_it(monkeypatch):
    monkeypatch.setenv("A2A_RESEARCHER_TOKEN", "secret-token")
    config = A2AConfig(enabled=True, agents={"researcher": A2ARemoteAgentConfig(enabled=True, agent_card_url="http://127.0.0.1:8101/.well-known/agent-card.json", token_env="A2A_RESEARCHER_TOKEN")})
    resolved = A2AAgentRegistry(config).resolve("researcher")
    assert resolved.authorization_header == "Bearer secret-token"
    assert "secret-token" not in repr(resolved)


def test_registry_rejects_builder_fallback():
    config = A2AConfig(enabled=True, agents={"builder": A2ARemoteAgentConfig(enabled=True, agent_card_url="http://127.0.0.1:8102/.well-known/agent-card.json", token_env="T", allow_local_fallback=True)})
    with pytest.raises(A2ARuntimeError):
        A2AAgentRegistry(config).resolve("builder")


def test_workspace_path_rejects_traversal(tmp_path):
    with pytest.raises(A2ARuntimeError, match="workspace"):
        resolve_workspace_path(str(tmp_path), "../outside.txt")


def test_artifact_transport_validates_hash_and_size(tmp_path):
    data = b"artifact"
    manifest = RemoteArtifactManifest("a1", "x.txt", "http://127.0.0.1:8101/artifacts/a1", "text/plain", len(data), hashlib.sha256(data).hexdigest())
    transport = A2AArtifactTransport(tmp_path, max_bytes=100, fetcher=lambda url, headers: data)
    saved = transport.download(manifest, "Bearer token")
    assert saved["sha256"] == manifest.sha256
    assert (tmp_path / "x.txt").read_bytes() == data


def test_a2a_client_preserves_execution_identity(monkeypatch):
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"status": "completed", "result": "ok", "task_id": "remote-1"}
    monkeypatch.setattr("httpx.post", lambda *a, **k: Response())
    from src.a2a_runtime import A2AClientRuntime
    from src.member_execution import MemberTaskRequest
    monkeypatch.setenv("TOK", "secret")
    config = A2AConfig(enabled=True, agents={"researcher": A2ARemoteAgentConfig(enabled=True, agent_card_url="http://127.0.0.1:8101/.well-known/agent-card.json", token_env="TOK")})
    request = MemberTaskRequest("run", "task", "research", "researcher", "researcher", "research", "t", "d", [], [], "", "s", True, "default", True, ".", "exec-1", "run:research:0")
    result = A2AClientRuntime(A2AAgentRegistry(config)).execute(request)
    assert result.execution_mode == "a2a"
    assert result.remote_task_id == "remote-1"


def test_a2a_client_does_not_resubmit_completed_execution(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("httpx.post", lambda *a, **k: calls.append(1))
    from src.a2a_runtime import A2AClientRuntime
    from src.member_execution import MemberTaskRequest
    from src.storage import Storage
    monkeypatch.setenv("TOK", "secret")
    config = A2AConfig(enabled=True, agents={"researcher": A2ARemoteAgentConfig(enabled=True, agent_card_url="http://127.0.0.1:8101/.well-known/agent-card.json", token_env="TOK")})
    storage = Storage(tmp_path / "a.sqlite")
    storage.upsert_a2a_task(team_run_id="run", logical_task_id="research", role="researcher", business_attempt=0, execution_id="exec-1", idempotency_key="run:research:0", transport_retry_count=0, context_id="ctx", remote_task_id="remote", endpoint="http://127.0.0.1:8101", status="completed")
    request = MemberTaskRequest("run", "task", "research", "researcher", "researcher", "research", "t", "d", [], [], "", "s", True, "default", True, ".", "exec-1", "run:research:0")
    result = A2AClientRuntime(A2AAgentRegistry(config), storage=storage).execute(request)
    assert result.remote_task_id == "remote"
    assert calls == []
