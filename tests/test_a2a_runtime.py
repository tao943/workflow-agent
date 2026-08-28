import pytest
from src.a2a_runtime import A2AAgentRegistry, A2ARuntimeError
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
