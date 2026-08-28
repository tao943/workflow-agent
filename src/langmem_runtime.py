from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field
from src.memory_runtime import MemoryCandidate


class ExtractedAgentMemory(BaseModel):
    memory_type: Literal["semantic", "episodic", "procedural", "preference"]
    subject: str
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)


class LangMemCandidateExtractor:
    def __init__(self, manager_factory=None, model: Any = None, max_candidates: int = 8):
        self.manager_factory = manager_factory
        self.model = model
        self.max_candidates = max_candidates

    def extract(self, *, role: str, team_run_id: str, task_id: str, final_answer: str, accepted_evidence: dict[str, Any] | None = None, accepted_source_refs: set[str] | None = None, evidence_ids: list[str] | None = None, source_refs: list[str] | None = None) -> list[MemoryCandidate]:
        accepted_evidence = accepted_evidence or {item: item for item in (evidence_ids or [])}
        accepted_source_refs = accepted_source_refs or set(source_refs or [])
        if self.manager_factory is None:
            from langmem import create_memory_manager
            self.manager_factory = create_memory_manager
        manager = self.manager_factory(self.model, schemas=[ExtractedAgentMemory], enable_inserts=True, enable_updates=True, enable_deletes=False)
        extracted = manager.invoke({"messages": [{"role": "assistant", "content": final_answer}]})
        if isinstance(extracted, dict):
            extracted = extracted.get("memories", extracted.get("results", []))
        return self._to_candidates(extracted[: self.max_candidates], role, accepted_evidence, accepted_source_refs)

    def _to_candidates(self, items, role: str, accepted_evidence: dict[str, Any], accepted_source_refs: set[str]) -> list[MemoryCandidate]:
        result = []
        for item in items:
            item = item if isinstance(item, ExtractedAgentMemory) else ExtractedAgentMemory.model_validate(item)
            valid_ids = [eid for eid in item.evidence_ids if eid in accepted_evidence]
            valid_refs = [ref for ref in item.source_refs if ref in accepted_source_refs]
            has_evidence = bool(valid_ids or valid_refs)
            result.append(MemoryCandidate(item.memory_type, f"project:team:{role}", item.subject, item.content, "evidence" if has_evidence else "model_inference", min(item.confidence, 1.0 if has_evidence else 0.4), valid_ids, valid_refs))
        return result
