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
