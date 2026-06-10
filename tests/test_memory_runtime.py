from pathlib import Path
from uuid import uuid4

from src.context_harness import ContextHarness, ContextRequest
from src.memory import remember_interaction, save_memory
from src.memory_runtime import (
    MemoryCandidate,
    MemoryConsolidator,
    MemoryGovernanceRuntime,
    MemoryPolicyGate,
    MemoryRetriever,
    MemoryStore,
)
from src.storage import Storage


def _case():
    root = Path("outputs") / "test_memory_runtime" / uuid4().hex[:8]
    return root, MemoryStore(root / "memory.sqlite")


def test_explicit_user_preference_is_hot_written_active():
    _, store = _case()
    runtime = MemoryGovernanceRuntime(store.db_path)

    records = runtime.consolidate_run(
        user_task="以后每次完成阶段进展都不要自动 git commit",
        final_answer="明白。",
        acceptance={"passed": True},
    )

    preferences = [item for item in records if item.memory_type == "preference"]
    assert preferences
    assert preferences[0].status == "active"
    assert preferences[0].trust == "user"


def test_model_only_memory_remains_candidate():
    _, store = _case()
    gate = MemoryPolicyGate(store)

    record = gate.evaluate(MemoryCandidate("semantic", "project", "architecture", "The project uses Redis."))

    assert record.status == "candidate"


def test_evidence_backed_fact_is_promoted():
    _, store = _case()
    gate = MemoryPolicyGate(store)

    record = gate.evaluate(
        MemoryCandidate("semantic", "project", "entrypoint", "CLI entrypoint is src/main.py", "evidence", evidence_ids=["ev_1"])
    )

    assert record.status == "active"


def test_consolidator_promotes_executor_tool_result_with_artifact_source():
    _, store = _case()
    consolidator = MemoryConsolidator(store, MemoryPolicyGate(store))

    records = consolidator.consolidate_run(
        "Inspect entrypoint",
        "Inspection completed.",
        tool_results=[
            {
                "tool": "read_file",
                "input": {"path": "src/main.py"},
                "status": "success",
                "result": "CLI entrypoint is src/main.py",
                "context_artifact_id": "ctx_1",
            }
        ],
        acceptance={"passed": True},
        session_id="sess",
    )

    semantic = next(item for item in records if item.memory_type == "semantic")
    assert semantic.status == "active"
    assert "src/main.py" in semantic.subject
    assert "ctx_1" in semantic.source_refs


def test_conflicting_active_memory_supersedes_old_record():
    _, store = _case()
    gate = MemoryPolicyGate(store)
    old = gate.evaluate(MemoryCandidate("semantic", "project", "model", "Model is v1", "evidence", 0.8, ["ev_1"]))
    new = gate.evaluate(MemoryCandidate("semantic", "project", "model", "Model is v2", "evidence", 0.9, ["ev_2"]))

    assert store.get(old.id).status == "superseded"
    assert new.status == "active"
    assert new.supersedes == old.id


def test_retriever_filters_status_namespace_and_node_type():
    _, store = _case()
    gate = MemoryPolicyGate(store)
    preference = gate.evaluate(MemoryCandidate("preference", "project", "user.preference", "Prefer concise plans", "user"))
    gate.evaluate(MemoryCandidate("episodic", "other", "task", "Concise plans worked", "reviewer", 0.8))
    gate.evaluate(MemoryCandidate("semantic", "project", "guess", "Concise plans use Redis"))

    results = MemoryRetriever(store).retrieve("concise plans", node="planner", namespace="project")

    assert [item.record.id for item in results] == [preference.id]


def test_legacy_migration_is_idempotent():
    root, store = _case()
    legacy = remember_interaction([], "优化 memory", "已新增结构化记忆。")
    save_memory(legacy, store.db_path)
    consolidator = MemoryConsolidator(store, MemoryPolicyGate(store))

    first = consolidator.migrate_legacy()
    second = consolidator.migrate_legacy()

    assert first > 0
    assert second == 0


def test_context_harness_includes_only_active_governed_memory():
    root = Path("outputs") / "test_memory_runtime" / uuid4().hex[:8]
    storage = Storage(root / "agent.sqlite")
    memory = MemoryGovernanceRuntime(root / "memory.sqlite")
    active = memory.policy.evaluate(MemoryCandidate("preference", "project", "user.preference", "Prefer memory governance", "user"))
    memory.policy.evaluate(MemoryCandidate("semantic", "project", "guess", "Memory governance uses Redis"))
    harness = ContextHarness(storage, root=root / "context", memory_db_path=root / "memory.sqlite")

    bundle = harness.prepare(ContextRequest("", "memory governance", "planner"))

    assert active.id in bundle.manifest["selected"]
    assert bundle.manifest["selected_memory"][0]["memory_id"] == active.id
    assert bundle.manifest["selected_memory"][0]["trust"] == "user"
    assert all(item.content != "Memory governance uses Redis" for item in bundle.model_context)


def test_context_harness_can_disable_memory_retrieval():
    root = Path("outputs") / "test_memory_runtime" / uuid4().hex[:8]
    storage = Storage(root / "agent.sqlite")
    memory = MemoryGovernanceRuntime(root / "memory.sqlite")
    active = memory.policy.evaluate(MemoryCandidate("preference", "project", "user.preference", "Prefer isolated memory", "user"))
    harness = ContextHarness(storage, root=root / "context", memory_db_path=root / "memory.sqlite")

    bundle = harness.prepare(ContextRequest("", "isolated memory", "planner", include_memory=False))

    assert active.id not in bundle.manifest["selected"]


def test_memory_usage_tracker_records_selection_and_feedback():
    root = Path("outputs") / "test_memory_runtime" / uuid4().hex[:8]
    storage = Storage(root / "agent.sqlite")
    runtime = MemoryGovernanceRuntime(root / "memory.sqlite", storage=storage)
    record = runtime.policy.evaluate(
        MemoryCandidate("preference", "project", "user.preference", "Prefer explainable context", "user")
    )

    results = runtime.retrieve("explainable context", node="planner", session_id="sess")
    runtime.usage.mark_answer_feedback([record.id], session_id="sess", cited_in_answer=True, helped_acceptance=True, feedback="useful")
    usage = storage.list_memory_usage(memory_id=record.id, session_id="sess")

    assert results[0].record.id == record.id
    assert usage[0]["selected"] is True
    assert usage[0]["cited_in_answer"] is True
    assert usage[0]["helped_acceptance"] is True
    assert usage[0]["feedback"] == "useful"


def test_memory_retrieval_explanation_includes_temporal_and_provenance_fields():
    _, store = _case()
    gate = MemoryPolicyGate(store)
    record = gate.evaluate(
        MemoryCandidate(
            "semantic",
            "project",
            "runtime",
            "Runtime stores context versions",
            "evidence",
            evidence_ids=["ev_1"],
            source_refs=["ctx_1"],
        )
    )
    record.valid_from = "2026-01-01"
    store.upsert(record)

    result = MemoryRetriever(store).retrieve("runtime context versions", node="verifier", namespace="project")[0]

    assert any(part.startswith("valid_from:") for part in result.reasons)
    assert any(part.startswith("sources:") for part in result.reasons)


def test_repeated_successful_episode_creates_procedural_memory():
    _, store = _case()
    consolidator = MemoryConsolidator(store, MemoryPolicyGate(store))
    kwargs = {
        "user_task": "Run memory checks",
        "final_answer": "Memory checks passed.",
        "acceptance": {"passed": True},
    }
    consolidator.consolidate_run(**kwargs, session_id="sess_1")
    records = consolidator.consolidate_run(**kwargs, session_id="sess_2")

    assert any(item.memory_type == "procedural" and item.status == "active" for item in records)
