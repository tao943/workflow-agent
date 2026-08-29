from src.main import _resolve_model
from src.config import A2AConfig, A2ARemoteAgentConfig, AppConfig
import pytest


def test_resolve_model_prefers_cli(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-v4-flash")

    assert _resolve_model("deepseek-v4-pro", "gpt-4o-mini") == "deepseek-v4-pro"


def test_resolve_model_prefers_env_over_config(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-v4-flash")

    assert _resolve_model(None, "gpt-4o-mini") == "deepseek-v4-flash"


def test_app_config_disables_external_features_by_default():
    config = AppConfig()
    assert config.a2a.enabled is False
    assert config.a2a.allow_insecure_http is False
    assert config.observability.capture_prompt_content is False
    assert config.langmem.enabled is False


def test_non_loopback_http_requires_explicit_development_override():
    with pytest.raises(ValueError, match="HTTPS"):
        A2AConfig(
            enabled=True,
            agents={
                "builder": A2ARemoteAgentConfig(
                    enabled=True,
                    agent_card_url="http://10.0.0.8:8102/.well-known/agent-card.json",
                    token_env="A2A_BUILDER_TOKEN",
                )
            },
        )
