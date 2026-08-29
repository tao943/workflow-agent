import json

from src.observability import JsonlTraceProvider, redact_sensitive


def test_redact_sensitive_recursively_removes_credentials():
    value = {"authorization": "Bearer secret-token", "nested": {"api_key": "sk-secret", "safe": "kept"}}
    assert redact_sensitive(value) == {"authorization": "[REDACTED]", "nested": {"api_key": "[REDACTED]", "safe": "kept"}}


def test_jsonl_provider_writes_parent_child_spans(tmp_path):
    path = tmp_path / "trace.jsonl"
    provider = JsonlTraceProvider(path, service_name="workflow-agent-test")
    with provider.start_span("agent.run", {"token": "secret"}) as root:
        with provider.start_span("team.plan", {"team": "default-dev-team"}) as child:
            assert child.identifiers.trace_id == root.identifiers.trace_id
            assert child.identifiers.parent_span_id == root.identifiers.span_id
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [item["name"] for item in records] == ["team.plan", "agent.run"]
    assert records[-1]["attributes"]["token"] == "[REDACTED]"


def test_jsonl_provider_injects_and_extracts_w3c_context(tmp_path):
    provider = JsonlTraceProvider(tmp_path / "trace.jsonl")
    carrier = {}
    with provider.start_span("lead") as lead:
        provider.inject(carrier)
        lead_id = lead.identifiers.trace_id
    with provider.extract(carrier):
        with provider.start_span("remote") as remote:
            assert remote.identifiers.trace_id == lead_id
