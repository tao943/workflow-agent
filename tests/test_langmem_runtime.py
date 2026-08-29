from typing import NamedTuple

from src.langmem_runtime import ExtractedAgentMemory, LangMemCandidateExtractor


class FakeManager:
    def invoke(self, _payload):
        return [
            {"memory_type": "semantic", "subject": "supported", "content": "fact", "confidence": 0.9, "evidence_ids": ["ev_1"], "source_refs": ["src:1"]},
            {"memory_type": "semantic", "subject": "unsupported", "content": "guess", "confidence": 0.9, "evidence_ids": ["missing"], "source_refs": ["src:x"]},
        ]


def test_langmem_attributes_evidence_per_candidate():
    candidates = LangMemCandidateExtractor(manager_factory=lambda *a, **k: FakeManager()).extract(role="researcher", team_run_id="run", task_id="task", final_answer="x", accepted_evidence={"ev_1": {}}, accepted_source_refs={"src:1"})
    assert candidates[0].evidence_ids == ["ev_1"]
    assert candidates[0].trust == "evidence"
    assert candidates[1].evidence_ids == []
    assert candidates[1].trust == "model_inference"
    assert candidates[1].confidence == 0.4


class OfficialExtractedMemory(NamedTuple):
    id: str
    content: ExtractedAgentMemory


class OfficialManager:
    def invoke(self, _payload):
        return [
            OfficialExtractedMemory(
                "memory-1",
                ExtractedAgentMemory(
                    memory_type="procedural",
                    subject="run tests",
                    content="Run focused tests before the full suite.",
                    confidence=0.85,
                    evidence_ids=["ev_accepted", "ev_forged"],
                    source_refs=["src:accepted", "src:forged"],
                ),
            )
        ]


def test_langmem_accepts_official_extracted_memory_shape_and_configures_manager():
    captured = {}

    def factory(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return OfficialManager()

    candidates = LangMemCandidateExtractor(
        manager_factory=factory,
        model="openai:gpt-4o-mini",
        max_candidates=3,
    ).extract(
        role="reviewer",
        team_run_id="run-1",
        task_id="task-1",
        final_answer="Tests passed with ev_accepted.",
        accepted_evidence={"ev_accepted": {"status": "success"}},
        accepted_source_refs={"src:accepted"},
    )

    assert captured["model"] == "openai:gpt-4o-mini"
    assert captured["schemas"] == [ExtractedAgentMemory]
    assert captured["enable_inserts"] is True
    assert captured["enable_updates"] is True
    assert captured["enable_deletes"] is False
    assert "ev_accepted" in captured["instructions"]
    assert "src:accepted" in captured["instructions"]
    assert len(candidates) == 1
    assert candidates[0].namespace == "project:team:reviewer"
    assert candidates[0].evidence_ids == ["ev_accepted"]
    assert candidates[0].source_refs == ["src:accepted"]
