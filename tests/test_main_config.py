from src.main import _resolve_model


def test_resolve_model_prefers_cli(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-v4-flash")

    assert _resolve_model("deepseek-v4-pro", "gpt-4o-mini") == "deepseek-v4-pro"


def test_resolve_model_prefers_env_over_config(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-v4-flash")

    assert _resolve_model(None, "gpt-4o-mini") == "deepseek-v4-flash"
