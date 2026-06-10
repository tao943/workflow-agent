from pathlib import Path
import json

from src.tools.registry import TOOL_REGISTRY, ToolContext, execute_tool


def test_tool_registry_contains_metadata():
    calculator = TOOL_REGISTRY["calculator"]
    notes = TOOL_REGISTRY["notes_tool"]

    assert calculator.description
    assert calculator.danger_level == "safe"
    assert calculator.parameters
    assert calculator.output_schema
    assert callable(calculator.execute)
    assert notes.danger_level == "confirm"
    assert notes.permissions == ["write:outputs/notes.md"]


def test_rag_search_reports_not_configured_when_missing(monkeypatch):
    monkeypatch.delenv("RAG_API_BASE", raising=False)
    result = execute_tool("rag_search", {"query": "LangGraph"}, ToolContext("sess", None, Path("outputs")))

    assert result.error
    assert result.metadata["code"] == "not_configured"


def test_rag_search_reads_fake_knowledge_base(monkeypatch):
    monkeypatch.delenv("RAG_API_BASE", raising=False)
    root = Path("outputs") / "test_rag"
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "knowledge_base.json").write_text(
        '[{"id":"1","title":"LangGraph","path":"docs/langgraph.md","text":"LangGraph checkpoint helps resume workflows."}]',
        encoding="utf-8",
    )
    monkeypatch.chdir(root)

    result = execute_tool("rag_search", {"query": "LangGraph checkpoint", "top_k": 1}, ToolContext("sess", None, outputs))

    assert result.error is None
    assert result.metadata["sources"][0]["title"] == "LangGraph"


def test_rag_search_calls_http_answer_service(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "answer": "LangGraph checkpoint can resume interrupted workflows.",
                    "sources": [{"source_number": 1, "source": "guide.md", "location": "p1"}],
                    "diagnostics": {"no_evidence": False, "retrieval_query": "LangGraph checkpoint"},
                }
            ).encode("utf-8")

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("RAG_API_BASE", "http://127.0.0.1:8010")
    monkeypatch.setenv("RAG_USER_ID", "user_8be74c46a603")
    monkeypatch.setenv("RAG_KNOWLEDGE_BASE_ID", "kb_001")
    monkeypatch.setenv("RAG_SEARCH_MODE", "answer")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = execute_tool("rag_search", {"query": "LangGraph checkpoint", "top_k": 4}, ToolContext("sess", None, Path("outputs")))

    assert result.error is None
    assert captured["url"] == "http://127.0.0.1:8010/v1/answer"
    assert captured["payload"]["user_id"] == "user_8be74c46a603"
    assert captured["payload"]["knowledge_base_id"] == "kb_001"
    assert captured["payload"]["retrieval_options"]["top_k"] == 4
    assert result.metadata["sources"][0]["source"] == "guide.md"
    assert "LangGraph checkpoint" in result.output
