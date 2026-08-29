from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from src.config import A2AConfig, A2ARemoteAgentConfig, AgentSpec, AppConfig, LangMemConfig
from src.memory_runtime import MemoryCandidate, MemoryGovernanceRuntime
from src.storage import Storage
from src.team import TeamRuntime


def _root() -> Path:
    return Path("outputs") / "test_team_langmem" / uuid4().hex[:8]


class RecordingExtractor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def extract(self, **kwargs):
        self.calls.append(kwargs)
        evidence_ids = list(kwargs["accepted_evidence"])
        source_refs = sorted(kwargs["accepted_source_refs"])
        return [
            MemoryCandidate(
                memory_type="semantic",
                namespace=f"project:team:{kwargs['role']}",
                subject=f"accepted:{kwargs['task_id']}",
                content="A reusable evidence-backed finding.",
                trust="evidence" if evidence_ids else "model_inference",
                confidence=0.9 if evidence_ids else 0.4,
                evidence_ids=evidence_ids[:1],
                source_refs=source_refs[:1],
            )
        ]


def _runtime(root: Path, extractor, *, fallback: bool = True) -> TeamRuntime:
    config = AppConfig(
        output_dir=str(root / "outputs"),
        checkpoint_path=str(root / "checkpoints.sqlite"),
        langmem=LangMemConfig(enabled=True, fallback_to_rule_consolidator=fallback),
    )
    return TeamRuntime(
        config,
        storage=Storage(root / "agent.sqlite"),
        langmem_extractor_factory=lambda **_kwargs: extractor,
    )


def test_evidence_identity_is_order_independent_and_task_scoped():
    root = _root()
    runtime = _runtime(root, RecordingExtractor())
    first = {"tool": "read_file", "args": {"path": "README.md"}, "raw_output_path": "artifact.txt"}
    second = {"raw_output_path": "artifact.txt", "args": {"path": "README.md"}, "tool": "read_file"}

    first_id, first_ref = runtime._evidence_identity("run-1", "task-1", first)
    reordered_id, reordered_ref = runtime._evidence_identity("run-1", "task-1", second)
    other_task_id, _ = runtime._evidence_identity("run-1", "task-2", first)

    assert first_id == reordered_id
    assert first_ref == reordered_ref
    assert first_id != other_task_id
    assert first_id.startswith("ev_")
    assert first_ref.endswith(first_id)


def test_evidence_cards_and_member_reports_use_stable_lead_ids():
    root = _root()
    runtime = _runtime(root, RecordingExtractor())
    evidence = [{"evidence_id": "ev_0123456789abcdef", "tool": "read_file", "status": "success", "args": {}}]
    item = {"logical_id": "logical-1"}
    payload = {
        "status": "completed",
        "final_answer": "- finding",
        "evidence": evidence,
        "artifacts": [],
        "acceptance": {"issues": []},
        "evidence_check": {"issues": []},
    }

    assert runtime._evidence_cards(evidence)[0] == "- ev_0123456789abcdef"
    assert runtime._member_report(item, payload)["evidence_ids"] == ["ev_0123456789abcdef"]


def test_candidate_evidence_id_is_paired_with_its_exact_source_ref():
    root = _root()
    runtime = _runtime(root, RecordingExtractor())
    accepted = {
        "ev_a": {"evidence_id": "ev_a", "source_ref": "src:a"},
        "ev_b": {"evidence_id": "ev_b", "source_ref": "src:b"},
    }
    candidate = MemoryCandidate(
        "semantic",
        "untrusted:namespace",
        "fact",
        "supported",
        "evidence",
        0.9,
        evidence_ids=["ev_a"],
        source_refs=[],
    )

    governed = runtime._govern_extracted_candidates(
        [candidate],
        role="researcher",
        accepted_evidence=accepted,
        accepted_source_refs={"src:a", "src:b"},
    )[0]

    assert governed.evidence_ids == ["ev_a"]
    assert governed.source_refs == ["src:a"]
    assert "src:b" not in governed.source_refs


def test_team_langmem_persists_only_lead_accepted_candidates():
    root = _root()
    extractor = RecordingExtractor()
    runtime = _runtime(root, extractor)

    runtime.run("Analyze this project", "research-build", auto_approve=True)

    assert extractor.calls
    evidenced_calls = [call for call in extractor.calls if call["accepted_evidence"]]
    assert evidenced_calls
    assert all(call["accepted_source_refs"] for call in evidenced_calls)
    assert all(eid.startswith("ev_") and len(eid) == 19 for call in evidenced_calls for eid in call["accepted_evidence"])
    records = MemoryGovernanceRuntime(runtime.memory_path).store.list()
    assert records
    assert all(record.namespace.startswith("project:team:") for record in records)
    assert any(record.status == "active" and record.evidence_ids for record in records)
    assert all(record.status == "candidate" for record in records if not record.evidence_ids)


class FailingExtractor:
    def extract(self, **_kwargs):
        raise RuntimeError("secret result content must not enter event")


def test_team_langmem_failure_emits_sanitized_event_and_falls_back_per_role():
    root = _root()
    runtime = _runtime(root, FailingExtractor(), fallback=True)

    result = runtime.run("Analyze this project", "research-build", auto_approve=True)

    events = [event for event in runtime.storage.list_events(result.session_id) if event["type"] == "memory_extraction_error"]
    assert len(events) == 1
    assert events[0]["data"] == {"role": "researcher", "error_type": "RuntimeError"}
    records = MemoryGovernanceRuntime(runtime.memory_path).store.list()
    assert records
    assert all(record.namespace.startswith("project:team:") for record in records)
    assert all("secret result content" not in record.content for record in records)


def test_team_langmem_failure_can_continue_without_rule_fallback():
    root = _root()
    runtime = _runtime(root, FailingExtractor(), fallback=False)

    runtime.run("Analyze this project", "research-build", auto_approve=True)

    assert MemoryGovernanceRuntime(runtime.memory_path).store.list() == []


def test_a2a_results_must_pass_lead_evidence_gate_and_cannot_forge_ids(monkeypatch):
    root = _root()
    extractor = RecordingExtractor()
    agents = {
        role: A2ARemoteAgentConfig(
            enabled=True,
            agent_card_url=f"http://127.0.0.1:81{index}/.well-known/agent-card.json",
            allow_local_fallback=False,
        )
        for index, role in enumerate(("researcher", "reviewer"), start=1)
    }
    config = AppConfig(
        output_dir=str(root / "outputs"),
        checkpoint_path=str(root / "checkpoints.sqlite"),
        langmem=LangMemConfig(enabled=True),
        a2a=A2AConfig(enabled=True, agents=agents),
    )
    runtime = TeamRuntime(
        config,
        storage=Storage(root / "agent.sqlite"),
        langmem_extractor_factory=lambda **_kwargs: extractor,
    )

    monkeypatch.setattr(
        "src.a2a_runtime.A2AAgentRegistry.resolve",
        lambda _self, role: SimpleNamespace(base_origin=f"http://127.0.0.1/{role}", authorization_header="Bearer test"),
    )

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "status": "completed",
                "result": "Remote result claims forged evidence ev_forged.",
                "evidence": [{"evidence_id": "ev_forged", "status": "success"}],
            }

    monkeypatch.setattr("httpx.post", lambda *_args, **_kwargs: Response())

    runtime.run("Analyze this project", "research-build", auto_approve=True)

    assert extractor.calls
    assert all("ev_forged" not in call["accepted_evidence"] for call in extractor.calls)
    member_runs = runtime.storage.list_team_member_runs(runtime.storage.list_team_runs()[0]["id"])
    assert all((item["result"].get("evidence_check") or {}).get("passed") is True for item in member_runs)


def _seed_role_memory(runtime: TeamRuntime, role: str, content: str, memory_type: str = "semantic") -> None:
    MemoryGovernanceRuntime(runtime.memory_path).evaluate_candidates(
        [
            MemoryCandidate(
                memory_type=memory_type,
                namespace=f"project:team:{role}",
                subject=f"{role} convention",
                content=content,
                trust="evidence",
                confidence=0.9,
                evidence_ids=[f"ev_seed_{role}"],
                source_refs=[f"seed:{role}"],
            )
        ]
    )


def _prepare_direct_member(runtime: TeamRuntime, member: AgentSpec, task_id: str) -> dict:
    team_run_id = "run-direct"
    if not runtime.storage.get_team_run(team_run_id):
        runtime.storage.create_team_run(team_run_id, "team-direct", "lead-session", "task")
        runtime.storage.add_team_member("member-direct", team_run_id, member.name, member.role, member.profile, f"session-{member.name}")
    runtime.storage.add_team_task(task_id, team_run_id, "Direct task", "Inspect the project", member.name)
    return {
        "id": task_id,
        "logical_id": f"logical-{task_id}",
        "title": "Direct task",
        "description": "Inspect the project",
        "member": member,
        "required_tools": [],
        "required_evidence": [],
        "metadata": {},
    }


def test_local_member_receives_only_bounded_role_memory_and_no_memory_disables_it():
    root = _root()
    captured_tasks: list[str] = []

    class FakeAgentRuntime:
        def run(self, options):
            captured_tasks.append(options.task)
            return SimpleNamespace(
                status="completed",
                session_id=options.session_id,
                final_answer="Completed.",
                acceptance={"passed": True, "issues": []},
                artifacts=[],
            )

    config = AppConfig(
        output_dir=str(root / "outputs"),
        langmem=LangMemConfig(enabled=True, recall_char_limit=2000),
    )
    runtime = TeamRuntime(
        config,
        storage=Storage(root / "agent.sqlite"),
        agent_runtime_factory=lambda *_args: FakeAgentRuntime(),
        langmem_extractor_factory=lambda **_kwargs: RecordingExtractor(),
    )
    _seed_role_memory(runtime, "builder", "Use pathlib for portable paths.\x00", "semantic")
    _seed_role_memory(runtime, "builder", "Remember the successful build episode.", "episodic")
    _seed_role_memory(runtime, "builder", "Run focused tests before the full suite.", "procedural")
    _seed_role_memory(runtime, "builder", "Prefer deterministic commands.", "preference")
    _seed_role_memory(runtime, "researcher", "Researcher-only source policy.")
    member = AgentSpec(name="builder", role="builder", profile="build")

    runtime._run_member_task("run-direct", "lead-session", _prepare_direct_member(runtime, member, "task-with-memory"), True, "default", False)
    runtime._run_member_task("run-direct", "lead-session", _prepare_direct_member(runtime, member, "task-no-memory"), True, "default", True)

    assert "BEGIN_UNTRUSTED_GOVERNED_MEMORY" in captured_tasks[0]
    assert "Use pathlib for portable paths." in captured_tasks[0]
    assert "Remember the successful build episode." in captured_tasks[0]
    assert "Run focused tests before the full suite." in captured_tasks[0]
    assert "Prefer deterministic commands." in captured_tasks[0]
    assert "Researcher-only source policy." not in captured_tasks[0]
    assert "\x00" not in captured_tasks[0]
    memory_block = captured_tasks[0].split("BEGIN_UNTRUSTED_GOVERNED_MEMORY", 1)[1]
    assert len("BEGIN_UNTRUSTED_GOVERNED_MEMORY" + memory_block) <= 2000
    assert "BEGIN_UNTRUSTED_GOVERNED_MEMORY" not in captured_tasks[1]


def test_a2a_member_receives_the_same_role_scoped_memory_context(monkeypatch):
    root = _root()
    sent_tasks: list[str] = []
    config = AppConfig(
        output_dir=str(root / "outputs"),
        langmem=LangMemConfig(enabled=True, recall_char_limit=400),
        a2a=A2AConfig(
            enabled=True,
            agents={
                "researcher": A2ARemoteAgentConfig(
                    enabled=True,
                    agent_card_url="http://127.0.0.1:8111/.well-known/agent-card.json",
                )
            },
        ),
    )
    runtime = TeamRuntime(config, storage=Storage(root / "agent.sqlite"))
    _seed_role_memory(runtime, "researcher", "Prefer primary sources.")
    _seed_role_memory(runtime, "builder", "Builder-only convention.")
    member = AgentSpec(name="researcher", role="researcher", profile="research")
    item = _prepare_direct_member(runtime, member, "task-a2a-memory")

    monkeypatch.setattr(
        "src.a2a_runtime.A2AAgentRegistry.resolve",
        lambda _self, _role: SimpleNamespace(base_origin="http://127.0.0.1:8111", authorization_header="Bearer test"),
    )

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "completed", "result": "Remote research completed."}

    def post(*_args, **kwargs):
        sent_tasks.append(kwargs["json"]["task"])
        return Response()

    monkeypatch.setattr("httpx.post", post)

    runtime._run_member_task("run-direct", "lead-session", item, True, "default", False)

    assert "BEGIN_UNTRUSTED_GOVERNED_MEMORY" in sent_tasks[0]
    assert "Prefer primary sources." in sent_tasks[0]
    assert "Builder-only convention." not in sent_tasks[0]


def test_langmem_disabled_skips_extraction_and_role_recall():
    root = _root()
    captured_tasks: list[str] = []

    class ExtractorMustNotRun:
        def extract(self, **_kwargs):
            raise AssertionError("LangMem extractor must remain disabled")

    class FakeAgentRuntime:
        def run(self, options):
            captured_tasks.append(options.task)
            return SimpleNamespace(
                status="completed",
                session_id=options.session_id,
                final_answer="Completed.",
                acceptance={"passed": True, "issues": []},
                artifacts=[],
            )

    config = AppConfig(output_dir=str(root / "outputs"), langmem=LangMemConfig(enabled=False))
    runtime = TeamRuntime(
        config,
        storage=Storage(root / "agent.sqlite"),
        agent_runtime_factory=lambda *_args: FakeAgentRuntime(),
        langmem_extractor_factory=lambda **_kwargs: ExtractorMustNotRun(),
    )
    _seed_role_memory(runtime, "builder", "Must not be recalled while disabled.")
    member = AgentSpec(name="builder", role="builder", profile="build")

    runtime._run_member_task("run-direct", "lead-session", _prepare_direct_member(runtime, member, "task-disabled"), True, "default", False)

    assert "BEGIN_UNTRUSTED_GOVERNED_MEMORY" not in captured_tasks[0]
