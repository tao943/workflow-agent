from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from src.config import ContextHarnessConfig
from src.memory_runtime import MemoryGovernanceRuntime
from src.storage import Storage
from src.observability import NoOpTraceProvider


ContextKind = Literal["tool_result", "message", "evidence", "summary", "member_report"]
FeedbackAction = Literal["continue", "retrieve", "retry", "repair", "rollback", "stop"]


def estimate_tokens(text: str) -> int:
    # Conservative provider-independent estimate for mixed Chinese/English text.
    return max(1, (len(str(text)) + 2) // 3)


@dataclass
class ContextArtifact:
    id: str
    content_hash: str
    kind: str
    file_path: str
    summary: str
    source_ref: str
    scope: str
    trust: str
    token_count: int
    created_at: str = ""
    invalidated_at: str | None = None


@dataclass
class StructuredSummary:
    goal: str = ""
    constraints: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    completed_work: list[str] = field(default_factory=list)
    failed_attempts: list[str] = field(default_factory=list)
    unresolved_issues: list[str] = field(default_factory=list)
    active_plan: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)


@dataclass
class ContextRequest:
    session_id: str
    task: str
    node: str
    agent: str = "build"
    current_step: dict[str, Any] | None = None
    dependency_task_ids: list[str] = field(default_factory=list)
    namespace: str = ""
    prompt_version: str = ""
    include_memory: bool = True


@dataclass
class ContextItem:
    id: str
    kind: str
    content: str
    source_ref: str
    priority: int
    relevance_score: float
    token_count: int
    trust: str = "unverified"
    pinned: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextBundle:
    model_context: list[ContextItem]
    local_context: dict[str, Any]
    manifest: dict[str, Any]
    estimated_tokens: int
    snapshot_id: str = ""

    @property
    def system_context(self) -> str:
        return "\n\n".join(item.content for item in self.model_context)

    @property
    def estimated_chars(self) -> int:
        return len(self.system_context)


@dataclass
class FeedbackDecision:
    action: FeedbackAction
    reason: str
    required_context: list[str] = field(default_factory=list)
    invalidated_context: list[str] = field(default_factory=list)
    target_checkpoint: str | None = None


@dataclass
class MemberReport:
    task_id: str
    status: str
    claims: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    failed_attempts: list[str] = field(default_factory=list)
    open_issues: list[str] = field(default_factory=list)


class ContextStore:
    def __init__(self, storage: Storage, root: str | Path = "outputs/context") -> None:
        self.storage = storage
        self.root = Path(root)
        self.artifact_dir = self.root / "artifacts"
        self.summary_dir = self.root / "summaries"
        self.snapshot_dir = self.root / "snapshots"

    def offload(self, content: str, metadata: dict[str, Any]) -> ContextArtifact:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        existing = self.storage.get_context_artifact_by_hash(content_hash)
        if existing:
            return ContextArtifact(**existing)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifact_dir / f"ctx_{content_hash[:16]}.txt"
        path.write_text(content, encoding="utf-8")
        artifact = ContextArtifact(
            id=f"ctx_{uuid4().hex[:12]}",
            content_hash=content_hash,
            kind=str(metadata.get("kind", "message")),
            file_path=str(path),
            summary=str(metadata.get("summary") or self._summarize(content)),
            source_ref=str(metadata.get("source_ref", "")),
            scope=str(metadata.get("scope", "project")),
            trust=str(metadata.get("trust", "unverified")),
            token_count=estimate_tokens(content),
        )
        self.storage.add_context_artifact(asdict(artifact))
        self._event(metadata.get("session_id"), "context.offloaded", asdict(artifact))
        return artifact

    def recall(self, artifact_id: str, session_id: str = "") -> str:
        artifact = self.storage.get_context_artifact(artifact_id)
        if not artifact:
            raise ValueError(f"Unknown context artifact: {artifact_id}")
        path = Path(artifact["file_path"]).resolve()
        allowed_root = self.root.resolve()
        if allowed_root not in path.parents:
            raise PermissionError("Context artifact path is outside the context root.")
        content = path.read_text(encoding="utf-8")
        self._event(session_id, "context.recalled", {"artifact_id": artifact_id})
        return content

    def retrieve(self, query: str, scope: str = "", limit: int = 8) -> list[ContextArtifact]:
        rows = self.storage.search_context_artifacts(query, scope=scope, limit=limit)
        return [ContextArtifact(**row) for row in rows]

    def invalidate(self, context_id: str, reason: str, session_id: str = "") -> None:
        self.storage.invalidate_context_artifact(context_id, reason)
        self._event(session_id, "context.invalidated", {"context_id": context_id, "reason": reason})

    def save_snapshot(self, payload: dict[str, Any]) -> str:
        snapshot_id = str(payload.get("id") or f"ctxsnap_{uuid4().hex[:12]}")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_dir / f"snapshot_{snapshot_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.storage.add_context_snapshot(snapshot_id, payload, str(path))
        return snapshot_id

    def _summarize(self, content: str, limit: int = 500) -> str:
        compact = " ".join(str(content).split())
        return compact if len(compact) <= limit else f"{compact[:limit]}..."

    def _event(self, session_id: Any, event_type: str, data: dict[str, Any]) -> None:
        if session_id:
            self.storage.add_event(str(session_id), event_type, data)


class ContextSupervisor:
    def __init__(self, config: ContextHarnessConfig, storage: Storage) -> None:
        self.config = config
        self.storage = storage

    def process_result(self, event: dict[str, Any]) -> FeedbackDecision:
        session_id = str(event.get("session_id") or "")
        status = str(event.get("status") or "success")
        fingerprint = self._fingerprint(event)
        repeats = self.storage.count_feedback_fingerprint(session_id, fingerprint)
        current_count = repeats + 1
        if status in {"timeout", "error", "invalid_args", "denied"}:
            metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
            stable_error_codes = {"not_configured", "service_unavailable", "permission_denied", "invalid_args"}
            if str(metadata.get("code") or "") in stable_error_codes or status in {"invalid_args", "denied"}:
                action = "repair"
            else:
                action = "retry" if repeats < self.config.max_retries_per_step else "repair"
            decision = FeedbackDecision(action, f"tool result status is {status}")
        elif current_count >= self.config.stagnation_window:
            decision = FeedbackDecision("stop", "repeated identical result indicates stagnation")
            self._event(session_id, "context.stagnation.detected", {"fingerprint": fingerprint, "repeats": current_count})
        elif event.get("type") in {"agent_output", "final_answer"} and event.get("required_evidence") and not _contains_known_evidence_ref(event):
            decision = FeedbackDecision("repair", "agent output is missing required evidence references", list(event.get("required_evidence") or []))
        elif event.get("memory_conflict") or event.get("conflicting_memory_ids"):
            conflicts = [str(item) for item in event.get("conflicting_memory_ids") or []]
            decision = FeedbackDecision("repair", "new evidence conflicts with active memory", invalidated_context=conflicts)
            for memory_id in conflicts:
                try:
                    MemoryGovernanceRuntime(Path("outputs") / "memory.sqlite", storage=self.storage).invalidate(memory_id, session_id=session_id)
                except Exception:
                    pass
        elif status == "success" and event.get("type") == "tool_result" and not _has_source_or_evidence(event):
            decision = FeedbackDecision("retrieve", "successful tool result lacks source, artifact, or evidence id")
        elif _is_no_evidence_result(event):
            decision = FeedbackDecision("retrieve", "tool result has no supporting evidence")
        elif event.get("required_evidence") and not event.get("evidence_ids"):
            decision = FeedbackDecision("retrieve", "required evidence is missing", list(event.get("required_evidence") or []))
        else:
            decision = FeedbackDecision("continue", "result advanced the task")
        self.storage.add_context_feedback(session_id, fingerprint, asdict(decision))
        self._event(session_id, "context.feedback.decided", asdict(decision))
        return decision

    def _fingerprint(self, event: dict[str, Any]) -> str:
        stable = {
            "type": event.get("type"),
            "tool": event.get("tool"),
            "args": event.get("args"),
            "status": event.get("status"),
            "error": event.get("error"),
        }
        return hashlib.sha256(json.dumps(stable, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _event(self, session_id: str, event_type: str, data: dict[str, Any]) -> None:
        if session_id:
            self.storage.add_event(session_id, event_type, data)


def _is_no_evidence_result(event: dict[str, Any]) -> bool:
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    diagnostics = metadata.get("diagnostics") if isinstance(metadata.get("diagnostics"), dict) else {}
    return str(metadata.get("code") or "") == "no_evidence" or bool(diagnostics.get("no_evidence"))


def _has_source_or_evidence(event: dict[str, Any]) -> bool:
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    return bool(
        event.get("evidence_id")
        or event.get("raw_output_path")
        or event.get("context_artifact_id")
        or metadata.get("sources")
        or metadata.get("source")
        or metadata.get("path")
        or metadata.get("feedback_decision")
    )


def _contains_known_evidence_ref(event: dict[str, Any]) -> bool:
    content = str(event.get("content") or event.get("final_answer") or event.get("output") or "")
    known = {str(item) for item in event.get("evidence_ids") or []}
    if not known:
        return "证据：ev_" in content or "evidence: ev_" in content.lower()
    return any(item in content for item in known)


class ContextHarness:
    def __init__(
        self,
        storage: Storage,
        config: ContextHarnessConfig | None = None,
        root: str | Path = "outputs/context",
        memory_db_path: str | Path | None = None,
        trace_provider=None,
    ) -> None:
        self.storage = storage
        self.config = config or ContextHarnessConfig()
        self.store = ContextStore(storage, root)
        self.supervisor = ContextSupervisor(self.config, storage)
        self.memory = MemoryGovernanceRuntime(
            memory_db_path or Path(root).parent / "memory.sqlite",
            storage=storage,
        )
        self._cache: dict[str, tuple[float, ContextBundle]] = {}
        self.trace_provider = trace_provider or NoOpTraceProvider()

    def prepare(self, request: ContextRequest) -> ContextBundle:
        with self.trace_provider.start_span("context.prepare", {"context.session_id": request.session_id, "context.node": request.node}):
            return self._prepare_impl(request)

    def _prepare_impl(self, request: ContextRequest) -> ContextBundle:
        key = self._cache_key(request)
        cached = self._cache.get(key)
        if cached and time.time() - cached[0] <= self.config.retrieval_cache_ttl_seconds:
            self._event(request.session_id, "context.cache.hit", {"key": key, "node": request.node})
            return cached[1]
        self._event(request.session_id, "context.cache.miss", {"key": key, "node": request.node})
        items = self._select_items(request)
        budget = max(1, self.config.max_input_tokens - self.config.reserved_output_tokens)
        selected: list[ContextItem] = []
        used = 0
        dropped: list[dict[str, Any]] = []
        for item in sorted(items, key=lambda row: (-row.priority, -row.relevance_score)):
            if used + item.token_count > budget and not item.pinned:
                dropped.append({"id": item.id, "reason": "budget_exceeded", "tokens": item.token_count})
                continue
            selected.append(item)
            used += item.token_count
        manifest = {
            "request": asdict(request),
            "selected": [item.id for item in selected],
            "dropped": dropped,
            "dropped_by_budget": dropped,
            "memory": [item.metadata for item in selected if item.kind == "memory"],
            "selected_memory": [item.metadata for item in selected if item.kind == "memory"],
            "artifacts": [item.metadata for item in selected if item.source_ref.startswith("ctx_") or item.metadata.get("artifact_id")],
            "selected_artifacts": [item.metadata for item in selected if item.source_ref.startswith("ctx_") or item.metadata.get("artifact_id")],
            "retrieval_reason": self._retrieval_reasons(selected),
            "trust_summary": self._trust_summary(selected),
            "budget": {
                "max_input_tokens": self.config.max_input_tokens,
                "reserved_output_tokens": self.config.reserved_output_tokens,
                "budget_tokens": budget,
                "used_tokens": used,
            },
            "budget_tokens": budget,
            "used_tokens": used,
            "versions": {
                "prompt_version": request.prompt_version,
                "memory_version": self._memory_version(),
                "artifact_version": self._artifact_version(request.namespace or request.session_id),
                "permission_scope": request.namespace or request.session_id or "project",
            },
        }
        bundle = ContextBundle(selected, {"namespace": request.namespace}, manifest, used)
        bundle.snapshot_id = self.store.save_snapshot({"id": f"ctxsnap_{uuid4().hex[:12]}", "manifest": manifest})
        self._cache[key] = (time.time(), bundle)
        self._event(request.session_id, "context.built", {"snapshot_id": bundle.snapshot_id, **manifest})
        return bundle

    def process_result(self, event: dict[str, Any]) -> FeedbackDecision:
        return self.supervisor.process_result(event)

    def compact(self, scope: dict[str, Any]) -> StructuredSummary:
        summary = StructuredSummary(
            goal=str(scope.get("goal") or scope.get("user_task") or ""),
            constraints=list(scope.get("constraints") or []),
            decisions=list(scope.get("decisions") or []),
            completed_work=[
                str(item.get("description"))
                for item in scope.get("plan", [])
                if item.get("status") == "completed"
            ],
            failed_attempts=[
                str(item.get("description"))
                for item in scope.get("plan", [])
                if item.get("status") in {"error", "skipped"}
            ],
            unresolved_issues=list((scope.get("acceptance") or {}).get("issues") or []),
            active_plan=[
                str(item.get("description"))
                for item in scope.get("plan", [])
                if item.get("status") == "pending"
            ],
            evidence_refs=[
                str(item.get("evidence_id"))
                for item in scope.get("tool_results", [])
                if item.get("evidence_id")
            ],
            artifact_refs=[
                str(item.get("context_artifact_id") or item.get("raw_output_path"))
                for item in scope.get("tool_results", [])
                if item.get("context_artifact_id") or item.get("raw_output_path")
            ],
        )
        return summary

    def join(self, reports: list[MemberReport], session_id: str = "") -> ContextBundle:
        evidence: list[str] = []
        artifacts: list[str] = []
        conflicts: list[str] = []
        claims: dict[str, dict[str, Any]] = {}
        for report in reports:
            evidence.extend(report.evidence_ids)
            artifacts.extend(report.artifacts)
            for claim in report.claims:
                text = str(claim.get("claim") or "")
                if text and text in claims and claims[text] != claim:
                    conflicts.append(text)
                elif text:
                    claims[text] = claim
        content = json.dumps(
            {
                "member_reports": [asdict(report) for report in reports],
                "evidence_ids": list(dict.fromkeys(evidence)),
                "artifacts": list(dict.fromkeys(artifacts)),
                "conflicts": conflicts,
            },
            ensure_ascii=False,
        )
        item = ContextItem("team_join", "member_report", content, "team_join", 100, 1.0, estimate_tokens(content), "verified", True)
        bundle = ContextBundle([item], {"conflicts": conflicts}, {"report_count": len(reports)}, item.token_count)
        self._event(session_id, "context.branch.joined", {"reports": len(reports), "conflicts": conflicts})
        return bundle

    def _select_items(self, request: ContextRequest) -> list[ContextItem]:
        items: list[ContextItem] = [
            ContextItem("current_task", "instruction", f"Current task: {request.task}", "request", 100, 1.0, estimate_tokens(request.task), "verified", True)
        ]
        summary = self.storage.get_session_summary(request.session_id) if request.session_id else ""
        if summary:
            items.append(ContextItem("session_summary", "summary", summary, request.session_id, 80, 0.8, estimate_tokens(summary), "inferred", metadata={"reason": "session_summary"}))
        for message in self.storage.list_messages(request.session_id)[-6:] if request.session_id else []:
            content = f"最近消息：\n- {message['role']}: {message['content']}"
            items.append(ContextItem(message["id"], "message", content, message["id"], 70, self._relevance(request.task, content), estimate_tokens(content), metadata={"reason": "recent_message"}))
        for artifact in self.store.retrieve(request.task, scope=request.namespace or request.session_id, limit=8):
            items.append(
                ContextItem(
                    artifact.id,
                    artifact.kind,
                    f"[artifact:{artifact.id}] {artifact.summary}",
                    artifact.source_ref or artifact.id,
                    60,
                    self._relevance(request.task, artifact.summary),
                    estimate_tokens(artifact.summary),
                    artifact.trust,
                    metadata={
                        "artifact_id": artifact.id,
                        "kind": artifact.kind,
                        "summary": artifact.summary,
                        "source_ref": artifact.source_ref,
                        "trust": artifact.trust,
                        "recallable": True,
                    },
                )
            )
        if request.include_memory:
            for result in self.memory.retrieve(
                request.task,
                node=request.node,
                namespace="project",
                limit=8,
                session_id=request.session_id,
            ):
                record = result.record
                content = f"[{record.memory_type}] {record.subject}: {record.content}"
                items.append(
                    ContextItem(
                        record.id,
                        "memory",
                        content,
                        f"memory:{record.id}",
                        65 if record.memory_type == "preference" else 55,
                        result.score,
                        estimate_tokens(content),
                        "verified",
                        metadata={
                            "memory_id": record.id,
                            "type": record.memory_type,
                            "trust": record.trust,
                            "confidence": record.confidence,
                            "score": result.score,
                            "reason": result.reasons,
                            "valid_from": record.valid_from,
                            "valid_to": record.valid_to,
                            "supersedes": record.supersedes,
                            "source_refs": record.source_refs,
                        },
                    )
                )
        return items

    def _cache_key(self, request: ContextRequest) -> str:
        payload = {
            "task": request.task,
            "node": request.node,
            "agent": request.agent,
            "namespace": request.namespace,
            "prompt_version": request.prompt_version,
            "include_memory": request.include_memory,
            "message_count": len(self.storage.list_messages(request.session_id)) if request.session_id else 0,
            "memory_version": self._memory_version(),
            "artifact_version": self._artifact_version(request.namespace or request.session_id),
            "permission_scope": request.namespace or request.session_id or "project",
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _memory_version(self) -> str:
        try:
            stats = self.memory.store.stats()
        except Exception:
            return "memory:unknown"
        return hashlib.sha256(json.dumps(stats, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    def _artifact_version(self, scope: str = "") -> str:
        try:
            artifacts = self.storage.search_context_artifacts("", scope=scope, limit=1000)
        except Exception:
            return "artifact:unknown"
        payload = [(item.get("id"), item.get("created_at"), item.get("invalidated_at")) for item in artifacts]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    def _retrieval_reasons(self, items: list[ContextItem]) -> list[dict[str, Any]]:
        return [
            {
                "id": item.id,
                "kind": item.kind,
                "priority": item.priority,
                "relevance_score": item.relevance_score,
                "pinned": item.pinned,
                "reason": item.metadata.get("reason") or ("pinned" if item.pinned else "ranked_context"),
            }
            for item in items
        ]

    def _trust_summary(self, items: list[ContextItem]) -> dict[str, int]:
        summary: dict[str, int] = {}
        for item in items:
            summary[item.trust] = summary.get(item.trust, 0) + 1
        return summary

    def _relevance(self, query: str, text: str) -> float:
        query_terms = set(re.findall(r"[\w-]{2,}", query.lower()))
        text_terms = set(re.findall(r"[\w-]{2,}", text.lower()))
        return len(query_terms & text_terms) / max(1, len(query_terms))

    def _event(self, session_id: str, event_type: str, data: dict[str, Any]) -> None:
        if session_id:
            self.storage.add_event(session_id, event_type, data)


class ModelUsageRecorder:
    def __init__(self, llm: Any, storage: Storage, session_id: str) -> None:
        self.llm = llm
        self.storage = storage
        self.session_id = session_id

    def invoke(self, messages, *args, **kwargs):
        started = time.perf_counter()
        response = self.llm.invoke(messages, *args, **kwargs)
        duration_ms = int((time.perf_counter() - started) * 1000)
        usage = dict(getattr(response, "usage_metadata", None) or {})
        metadata = dict(getattr(response, "response_metadata", None) or {})
        token_usage = dict(metadata.get("token_usage") or metadata.get("usage") or {})
        hit = int(token_usage.get("prompt_cache_hit_tokens") or usage.get("prompt_cache_hit_tokens") or 0)
        miss = int(token_usage.get("prompt_cache_miss_tokens") or usage.get("prompt_cache_miss_tokens") or 0)
        total = hit + miss
        self.storage.add_event(
            self.session_id,
            "model.cache.usage",
            {
                "prompt_cache_hit_tokens": hit,
                "prompt_cache_miss_tokens": miss,
                "cache_hit_ratio": hit / total if total else 0.0,
                "duration_ms": duration_ms,
                "usage": usage or token_usage,
            },
        )
        return response

    def __getattr__(self, name: str):
        return getattr(self.llm, name)


def context_for_state(state: dict[str, Any], node: str, prompt_version: str = "") -> ContextBundle:
    storage = Storage(state.get("storage_path") or "outputs/agent.sqlite")
    config = ContextHarnessConfig.model_validate(state.get("context_harness_config") or {})
    output_dir = Path(state.get("output_dir") or "outputs")
    harness = ContextHarness(storage, config, output_dir / "context")
    current_index = int(state.get("current_step_index", 0) or 0)
    plan = state.get("plan") or []
    current_step = plan[current_index] if current_index < len(plan) else None
    return harness.prepare(
        ContextRequest(
            session_id=state.get("session_id", ""),
            task=state.get("user_task", ""),
            node=node,
            agent=state.get("agent", "build"),
            current_step=current_step,
            namespace=state.get("context_namespace", state.get("session_id", "")),
            prompt_version=prompt_version,
            include_memory=bool(state.get("memory_enabled", True)),
        )
    )
