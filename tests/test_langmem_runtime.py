from src.langmem_runtime import LangMemCandidateExtractor


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
