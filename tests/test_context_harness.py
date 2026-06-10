from pathlib import Path
from uuid import uuid4

from src.checkpoint import fork_checkpoint, list_checkpoints, save_checkpoint
from src.config import ContextHarnessConfig
from src.context_harness import ContextHarness, ContextRequest, ContextStore, ContextSupervisor, MemberReport, ModelUsageRecorder
from src.graph import make_initial_state
from src.memory_runtime import MemoryCandidate, MemoryGovernanceRuntime
from src.storage import Storage


def _case():
    root = Path("outputs") / "test_context_harness" / uuid4().hex[:8]
    return root, Storage(root / "agent.sqlite")


def test_context_store_offloads_deduplicates_and_recalls():
    root, storage = _case()
    store = ContextStore(storage, root / "context")
    first = store.offload("large context result", {"kind": "tool_result", "scope": "sess"})
    second = store.offload("large context result", {"kind": "tool_result", "scope": "sess"})
    assert first.id == second.id
    assert Path(first.file_path).exists()
    assert store.recall(first.id) == "large context result"


def test_context_harness_budget_and_cache_events():
    root, storage = _case()
    storage.create_session("sess", "context", "build")
    storage.add_message("msg", "sess", "user", "hello context")
    harness = ContextHarness(storage, ContextHarnessConfig(max_input_tokens=100, reserved_output_tokens=20), root / "context")
    first = harness.prepare(ContextRequest("sess", "hello", "planner", namespace="sess"))
    second = harness.prepare(ContextRequest("sess", "hello", "planner", namespace="sess"))
    assert first.estimated_tokens <= 80
    assert second.snapshot_id == first.snapshot_id
    assert any(item["type"] == "context.cache.hit" for item in storage.list_events("sess"))


def test_context_harness_manifest_explains_memory_artifacts_and_usage():
    root, storage = _case()
    storage.create_session("sess", "context", "build")
    memory = MemoryGovernanceRuntime(root / "memory.sqlite")
    active = memory.policy.evaluate(
        MemoryCandidate("preference", "project", "context governance", "Prefer governed context selection", "user")
    )
    harness = ContextHarness(storage, root=root / "context", memory_db_path=root / "memory.sqlite")
    artifact = harness.store.offload(
        "governed context artifact body",
        {
            "kind": "tool_result",
            "scope": "sess",
            "summary": "governed context artifact summary",
            "source_ref": "tool:list_files",
            "trust": "evidence",
            "session_id": "sess",
        },
    )

    bundle = harness.prepare(ContextRequest("sess", "governed context", "planner", namespace="sess"))

    assert active.id in bundle.manifest["selected"]
    assert artifact.id in bundle.manifest["selected"]
    assert bundle.manifest["selected_memory"][0]["memory_id"] == active.id
    assert bundle.manifest["selected_artifacts"][0]["artifact_id"] == artifact.id
    assert bundle.manifest["selected_artifacts"][0]["recallable"] is True
    assert bundle.manifest["budget"]["used_tokens"] == bundle.estimated_tokens
    assert "versions" in bundle.manifest
    assert storage.list_memory_usage(memory_id=active.id, session_id="sess")


def test_context_supervisor_detects_failure_and_stagnation():
    _, storage = _case()
    supervisor = ContextSupervisor(ContextHarnessConfig(stagnation_window=2), storage)
    failure = supervisor.process_result({"session_id": "sess", "type": "tool_result", "tool": "x", "args": {}, "status": "error"})
    supervisor.process_result({"session_id": "sess2", "type": "tool_result", "tool": "x", "args": {}, "status": "success"})
    repeated = supervisor.process_result({"session_id": "sess2", "type": "tool_result", "tool": "x", "args": {}, "status": "success"})
    assert failure.action == "retry"
    assert repeated.action == "stop"


def test_context_supervisor_requests_retrieval_or_repair_for_context_gaps():
    _, storage = _case()
    supervisor = ContextSupervisor(ContextHarnessConfig(), storage)

    missing_source = supervisor.process_result({"session_id": "sess", "type": "tool_result", "tool": "x", "status": "success"})
    missing_evidence = supervisor.process_result(
        {
            "session_id": "sess",
            "type": "final_answer",
            "required_evidence": ["ev_1"],
            "evidence_ids": ["ev_1"],
            "final_answer": "done without citation",
        }
    )

    assert missing_source.action == "retrieve"
    assert missing_evidence.action == "repair"


def test_context_join_deduplicates_evidence_and_reports_conflicts():
    root, storage = _case()
    harness = ContextHarness(storage, root=root / "context")
    reports = [
        MemberReport("a", "completed", claims=[{"claim": "same", "evidence_ids": ["ev_1"]}], evidence_ids=["ev_1"]),
        MemberReport("b", "completed", claims=[{"claim": "same", "evidence_ids": ["ev_2"]}], evidence_ids=["ev_1", "ev_2"]),
    ]
    bundle = harness.join(reports, "sess")
    assert bundle.local_context["conflicts"] == ["same"]
    assert '"ev_1", "ev_2"' in bundle.system_context


def test_checkpoint_lineage_creates_branch():
    root, _ = _case()
    db_path = root / "checkpoints.sqlite"
    checkpoint_id = save_checkpoint("lineage-run", make_initial_state("original"), "running", "planner", db_path=db_path)
    branch_id, forked = fork_checkpoint("lineage-run", checkpoint_id, {"user_task": "forked"}, db_path=db_path)
    lineage = list_checkpoints("lineage-run", db_path=db_path)
    assert branch_id.startswith("branch_")
    assert forked["user_task"] == "forked"
    assert lineage[-1]["parent_checkpoint_id"] == checkpoint_id


def test_model_usage_recorder_records_provider_cache_metrics():
    _, storage = _case()

    class Response:
        usage_metadata = {}
        response_metadata = {"token_usage": {"prompt_cache_hit_tokens": 80, "prompt_cache_miss_tokens": 20}}

    class LLM:
        def invoke(self, messages):
            return Response()

    ModelUsageRecorder(LLM(), storage, "sess").invoke([])
    event = storage.list_events("sess")[-1]
    assert event["type"] == "model.cache.usage"
    assert event["data"]["cache_hit_ratio"] == 0.8
