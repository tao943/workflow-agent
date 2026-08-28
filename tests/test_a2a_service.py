from fastapi.testclient import TestClient
from src.a2a_service import build_role_app
from src.config import AppConfig, A2AConfig, A2ARemoteAgentConfig


def _app():
    config = AppConfig(a2a=A2AConfig(enabled=True, agents={"researcher": A2ARemoteAgentConfig(enabled=True, token_env="TOK", workspace_root="." )}))
    return build_role_app(config, "researcher", "http://127.0.0.1:8101")


def test_agent_card_is_public():
    response = TestClient(_app()).get("/.well-known/agent-card.json")
    assert response.status_code == 200
    assert "token" not in response.text.lower()


def test_a2a_route_requires_bearer_token(monkeypatch):
    monkeypatch.setenv("TOK", "secret")
    response = TestClient(_app()).post("/a2a/rest/v1/message:send", json={})
    assert response.status_code == 401


def test_a2a_route_preserves_trace_context(tmp_path, monkeypatch):
    from src.observability import JsonlTraceProvider
    monkeypatch.setenv("TOK", "secret")
    provider = JsonlTraceProvider(tmp_path / "remote.jsonl")
    lead = JsonlTraceProvider(tmp_path / "lead.jsonl")
    carrier = {}
    with lead.start_span("lead") as span:
        lead.inject(carrier)
        client = TestClient(build_role_app(AppConfig(a2a=A2AConfig(enabled=True, agents={"researcher": A2ARemoteAgentConfig(enabled=True, token_env="TOK")})), "researcher", "http://127.0.0.1:8101", trace_provider=provider))
        response = client.post("/a2a/rest/v1/message:send", headers={"Authorization": "Bearer secret", **carrier}, json={})
    assert response.status_code == 200
    records = [line for line in (tmp_path / "remote.jsonl").read_text().splitlines() if 'a2a.remote' in line]
    assert records and span.identifiers.trace_id in records[0]
