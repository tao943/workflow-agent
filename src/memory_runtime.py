from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4


MEMORY_TYPES = {"semantic", "episodic", "procedural", "preference"}
MEMORY_STATUSES = {"candidate", "active", "superseded", "invalidated", "archived"}
TRUST_SCORES = {
    "user": 1.0,
    "evidence": 0.9,
    "reviewer": 0.8,
    "repeated_experience": 0.75,
    "model_inference": 0.35,
    "legacy": 0.3,
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in payload] if isinstance(payload, list) else []


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[\w-]{2,}", str(text).lower()))


def _clip(text: str, limit: int = 900) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else f"{compact[:limit]}..."


@dataclass
class MemoryRecord:
    id: str
    memory_type: str
    namespace: str
    subject: str
    content: str
    status: str = "candidate"
    confidence: float = 0.35
    trust: str = "model_inference"
    evidence_ids: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    valid_from: str | None = None
    valid_to: str | None = None
    supersedes: str | None = None
    created_at: str = ""
    updated_at: str = ""
    last_accessed_at: str | None = None
    access_count: int = 0


@dataclass
class MemoryCandidate:
    memory_type: str
    namespace: str
    subject: str
    content: str
    trust: str = "model_inference"
    confidence: float | None = None
    evidence_ids: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    valid_from: str | None = None


@dataclass
class MemorySearchResult:
    record: MemoryRecord
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class MemoryUsageRecord:
    memory_id: str
    session_id: str
    run_id: str = ""
    node: str = ""
    selected: bool = True
    cited_in_answer: bool = False
    helped_acceptance: bool = False
    feedback: str | None = None


class EmbeddingReranker(Protocol):
    def rerank(self, query: str, records: list[MemoryRecord]) -> list[MemoryRecord]: ...


class MemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_records (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    trust TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    source_refs_json TEXT NOT NULL,
                    valid_from TEXT,
                    valid_to TEXT,
                    supersedes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(namespace, memory_type, content_hash)
                );
                CREATE TABLE IF NOT EXISTS memory_migrations (
                    migration_key TEXT PRIMARY KEY,
                    completed_at TEXT NOT NULL
                );
                """
            )
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_records_fts "
                    "USING fts5(id UNINDEXED, subject, content, namespace, memory_type)"
                )
            except sqlite3.OperationalError:
                pass

    def upsert(self, record: MemoryRecord) -> MemoryRecord:
        now = _now()
        record.created_at = record.created_at or now
        record.updated_at = now
        content_hash = hashlib.sha256(record.content.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM memory_records WHERE namespace = ? AND memory_type = ? AND content_hash = ?",
                (record.namespace, record.memory_type, content_hash),
            ).fetchone()
            if existing:
                merged = self._row(existing)
                merged.evidence_ids = list(dict.fromkeys(merged.evidence_ids + record.evidence_ids))
                merged.source_refs = list(dict.fromkeys(merged.source_refs + record.source_refs))
                merged.confidence = max(merged.confidence, record.confidence)
                if record.status == "active":
                    merged.status = "active"
                merged.updated_at = now
                conn.execute(
                    """
                    UPDATE memory_records
                    SET status = ?, confidence = ?, trust = ?, evidence_ids_json = ?,
                        source_refs_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        merged.status,
                        merged.confidence,
                        record.trust if TRUST_SCORES.get(record.trust, 0) >= TRUST_SCORES.get(merged.trust, 0) else merged.trust,
                        json.dumps(merged.evidence_ids, ensure_ascii=False),
                        json.dumps(merged.source_refs, ensure_ascii=False),
                        now,
                        merged.id,
                    ),
                )
                return merged
            conn.execute(
                """
                INSERT INTO memory_records (
                    id, memory_type, namespace, subject, content, content_hash, status,
                    confidence, trust, evidence_ids_json, source_refs_json, valid_from,
                    valid_to, supersedes, created_at, updated_at, last_accessed_at, access_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.memory_type,
                    record.namespace,
                    record.subject,
                    record.content,
                    content_hash,
                    record.status,
                    record.confidence,
                    record.trust,
                    json.dumps(record.evidence_ids, ensure_ascii=False),
                    json.dumps(record.source_refs, ensure_ascii=False),
                    record.valid_from,
                    record.valid_to,
                    record.supersedes,
                    record.created_at,
                    record.updated_at,
                    record.last_accessed_at,
                    record.access_count,
                ),
            )
            try:
                conn.execute(
                    "INSERT INTO memory_records_fts (id, subject, content, namespace, memory_type) VALUES (?, ?, ?, ?, ?)",
                    (record.id, record.subject, record.content, record.namespace, record.memory_type),
                )
            except sqlite3.OperationalError:
                pass
        return record

    def get(self, memory_id: str) -> MemoryRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM memory_records WHERE id = ?", (memory_id,)).fetchone()
        return self._row(row) if row else None

    def list(self, status: str | None = None, limit: int = 100) -> list[MemoryRecord]:
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM memory_records WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM memory_records ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(row) for row in rows]

    def find_active_subject(self, namespace: str, memory_type: str, subject: str) -> list[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memory_records
                WHERE namespace = ? AND memory_type = ? AND subject = ? AND status = 'active'
                ORDER BY confidence DESC, updated_at DESC
                """,
                (namespace, memory_type, subject),
            ).fetchall()
        return [self._row(row) for row in rows]

    def update_status(self, memory_id: str, status: str, supersedes: str | None = None) -> bool:
        if status not in MEMORY_STATUSES:
            raise ValueError(f"Unknown memory status: {status}")
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE memory_records SET status = ?, supersedes = COALESCE(?, supersedes), updated_at = ? WHERE id = ?",
                (status, supersedes, _now(), memory_id),
            )
        return cursor.rowcount > 0

    def mark_accessed(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        with self._connect() as conn:
            conn.executemany(
                "UPDATE memory_records SET last_accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
                [(_now(), memory_id) for memory_id in memory_ids],
            )

    def search(self, query: str, namespace: str = "", memory_types: list[str] | None = None, limit: int = 8) -> list[MemoryRecord]:
        ids: list[str] = []
        terms = [term for term in _terms(query) if len(term) > 1]
        if terms:
            try:
                fts_query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in list(terms)[:12])
                with self._connect() as conn:
                    ids = [
                        row[0]
                        for row in conn.execute(
                            "SELECT id FROM memory_records_fts WHERE memory_records_fts MATCH ? ORDER BY bm25(memory_records_fts) LIMIT ?",
                            (fts_query, limit * 4),
                        ).fetchall()
                    ]
            except sqlite3.OperationalError:
                ids = []
        records = [self.get(memory_id) for memory_id in ids]
        filtered = [
            item for item in records
            if item and item.status == "active"
            and (not namespace or item.namespace == namespace or item.namespace.startswith(f"{namespace}:"))
            and (not memory_types or item.memory_type in memory_types)
            and not item.valid_to
        ]
        if len(filtered) < limit:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM memory_records WHERE status = 'active' AND valid_to IS NULL ORDER BY updated_at DESC LIMIT 200"
                ).fetchall()
            query_terms = _terms(query)
            fallback = sorted(
                (
                    (len(query_terms & _terms(f"{row['subject']} {row['content']}")), self._row(row))
                    for row in rows
                    if (not namespace or row["namespace"] == namespace or row["namespace"].startswith(f"{namespace}:"))
                    and (not memory_types or row["memory_type"] in memory_types)
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            known = {item.id for item in filtered}
            filtered.extend(item for score, item in fallback if score > 0 and item.id not in known)
        return filtered[:limit]

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            by_status = dict(conn.execute("SELECT status, COUNT(*) FROM memory_records GROUP BY status").fetchall())
            by_type = dict(conn.execute("SELECT memory_type, COUNT(*) FROM memory_records GROUP BY memory_type").fetchall())
            total = conn.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
        return {"total": total, "by_status": by_status, "by_type": by_type}

    def migration_done(self, key: str) -> bool:
        with self._connect() as conn:
            return conn.execute("SELECT 1 FROM memory_migrations WHERE migration_key = ?", (key,)).fetchone() is not None

    def mark_migration_done(self, key: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO memory_migrations (migration_key, completed_at) VALUES (?, ?)",
                (key, _now()),
            )

    def _row(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            memory_type=row["memory_type"],
            namespace=row["namespace"],
            subject=row["subject"],
            content=row["content"],
            status=row["status"],
            confidence=float(row["confidence"]),
            trust=row["trust"],
            evidence_ids=_json_list(row["evidence_ids_json"]),
            source_refs=_json_list(row["source_refs_json"]),
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            supersedes=row["supersedes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed_at=row["last_accessed_at"],
            access_count=int(row["access_count"]),
        )


class MemoryPolicyGate:
    def __init__(self, store: MemoryStore, min_active_confidence: float = 0.7) -> None:
        self.store = store
        self.min_active_confidence = min_active_confidence

    def evaluate(self, candidate: MemoryCandidate) -> MemoryRecord:
        trust_score = TRUST_SCORES.get(candidate.trust, 0.2)
        confidence = candidate.confidence if candidate.confidence is not None else trust_score
        active = (
            candidate.trust == "user"
            or bool(candidate.evidence_ids)
            or candidate.trust in {"reviewer", "repeated_experience"} and confidence >= self.min_active_confidence
        )
        status = "active" if active else "candidate"
        record = MemoryRecord(
            id=f"memr_{uuid4().hex[:12]}",
            memory_type=candidate.memory_type,
            namespace=candidate.namespace,
            subject=candidate.subject,
            content=_clip(candidate.content),
            status=status,
            confidence=confidence,
            trust=candidate.trust,
            evidence_ids=candidate.evidence_ids,
            source_refs=candidate.source_refs,
            valid_from=candidate.valid_from or _now(),
        )
        if status == "active":
            for old in self.store.find_active_subject(candidate.namespace, candidate.memory_type, candidate.subject):
                if old.content != record.content and record.confidence >= old.confidence:
                    self.store.update_status(old.id, "superseded")
                    record.supersedes = old.id
        return self.store.upsert(record)

    def invalidate(self, memory_id: str) -> bool:
        return self.store.update_status(memory_id, "invalidated")


class MemoryRetriever:
    NODE_TYPES = {
        "planner": ["preference", "procedural", "semantic"],
        "executor": ["episodic", "procedural", "preference"],
        "verifier": ["semantic", "episodic"],
        "summarizer": ["semantic", "preference", "procedural"],
        "team_lead": ["procedural", "episodic", "preference", "semantic"],
        "team_researcher": ["semantic", "procedural", "episodic", "preference"],
        "team_builder": ["procedural", "episodic", "semantic", "preference"],
        "team_reviewer": ["semantic", "episodic", "procedural", "preference"],
    }

    def __init__(self, store: MemoryStore, reranker: EmbeddingReranker | None = None) -> None:
        self.store = store
        self.reranker = reranker

    def retrieve(self, query: str, node: str = "planner", namespace: str = "project", limit: int = 8) -> list[MemorySearchResult]:
        allowed = self.NODE_TYPES.get(node, list(MEMORY_TYPES))
        records = self.store.search(query, namespace=namespace, memory_types=allowed, limit=limit * 2)
        if self.reranker:
            records = self.reranker.rerank(query, records)
        query_terms = _terms(query)
        results: list[MemorySearchResult] = []
        for record in records:
            overlap = len(query_terms & _terms(f"{record.subject} {record.content}"))
            trust = TRUST_SCORES.get(record.trust, 0.2)
            confidence = max(0.0, min(1.0, record.confidence))
            usefulness = min(record.access_count, 10) / 100
            node_weight = 0.2 if record.memory_type in allowed[:2] else 0.0
            temporal_penalty = -1.0 if record.valid_to else 0.0
            score = overlap * 0.2 + trust * 0.45 + confidence * 0.2 + usefulness + node_weight + temporal_penalty
            results.append(
                MemorySearchResult(
                    record,
                    score,
                    [
                        "recall:bm25_fts_or_keyword",
                        f"node:{node}",
                        f"type:{record.memory_type}",
                        f"trust:{record.trust}:{trust:.2f}",
                        f"confidence:{confidence:.2f}",
                        f"overlap:{overlap}",
                        f"usefulness:{record.access_count}",
                        f"valid_from:{record.valid_from or ''}",
                        f"valid_to:{record.valid_to or 'current'}",
                        f"supersedes:{record.supersedes or ''}",
                        f"sources:{','.join(record.source_refs[:3])}",
                    ],
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        selected = results[:limit]
        self.store.mark_accessed([item.record.id for item in selected])
        return selected


class MemoryConsolidator:
    def __init__(self, store: MemoryStore, policy: MemoryPolicyGate) -> None:
        self.store = store
        self.policy = policy

    def consolidate_run(
        self,
        user_task: str,
        final_answer: str,
        tool_results: list[dict[str, Any]] | None = None,
        acceptance: dict[str, Any] | None = None,
        namespace: str = "project",
        session_id: str = "",
        learnings: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        source_refs = [f"session:{session_id}"] if session_id else []
        preference = self._explicit_preference(user_task)
        if preference:
            records.append(self.policy.evaluate(MemoryCandidate("preference", namespace, "user.preference", preference, "user", 1.0, source_refs=source_refs)))

        evidence_ids = [
            str(item.get("evidence_id") or item.get("tool") or "")
            for item in tool_results or []
            if item.get("status") == "success" and (item.get("evidence_id") or item.get("tool"))
        ]
        source_refs.extend(
            str(ref)
            for item in tool_results or []
            for ref in (item.get("context_artifact_id"), item.get("raw_output_path"))
            if ref
        )
        source_refs = list(dict.fromkeys(source_refs))
        for item in tool_results or []:
            if item.get("status") != "success":
                continue
            content = str(item.get("display_output") or item.get("output") or item.get("result") or "")
            if not content:
                continue
            evidence_id = str(item.get("evidence_id") or item.get("tool") or "tool")
            tool_input = item.get("input") or item.get("args") or {}
            subject_suffix = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)[:180] if tool_input else ""
            records.append(
                self.policy.evaluate(
                    MemoryCandidate(
                        "semantic",
                        namespace,
                        f"tool:{item.get('tool', 'unknown')}:{subject_suffix}",
                        _clip(content, 600),
                        "evidence",
                        0.9,
                        [evidence_id],
                        source_refs,
                    )
                )
            )

        passed = bool((acceptance or {}).get("passed"))
        trust = "reviewer" if passed else "model_inference"
        episode = self.policy.evaluate(
            MemoryCandidate(
                "episodic",
                namespace,
                _clip(user_task, 180),
                _clip(final_answer, 800),
                trust,
                0.8 if passed else 0.35,
                evidence_ids if passed else [],
                source_refs,
            )
        )
        records.append(episode)
        if episode.status == "active" and len(episode.source_refs) >= 2:
            records.append(
                self.policy.evaluate(
                    MemoryCandidate(
                        "procedural",
                        namespace,
                        f"repeated:{episode.subject}",
                        f"Reusable approach from repeated successful task: {episode.content}",
                        "repeated_experience",
                        0.75,
                        episode.evidence_ids,
                        episode.source_refs,
                    )
                )
            )
        for category in ("conventions", "successes", "gotchas", "commands"):
            for item in (learnings or {}).get(category, [])[:8]:
                records.append(
                    self.policy.evaluate(
                        MemoryCandidate(
                            "procedural",
                            f"{namespace}:team",
                            f"team.{category}",
                            str(item),
                            "reviewer" if passed else "model_inference",
                            0.8 if passed else 0.35,
                            evidence_ids if passed else [],
                            source_refs,
                        )
                    )
                )
        return records

    def migrate_legacy(self) -> int:
        if self.store.migration_done("legacy_memory_entries_v1"):
            return 0
        with self.store._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_entries'"
            ).fetchone()
            rows = conn.execute(
                "SELECT id, user_task, summary, facts_json, preferences_json, decisions_json FROM memory_entries"
            ).fetchall() if exists else []
        count = 0
        for row in rows:
            source = [f"legacy:{row['id']}"]
            self.policy.evaluate(MemoryCandidate("episodic", "project", row["user_task"], row["summary"] or "", "legacy", 0.3, source_refs=source))
            count += 1
            for value in _json_list(row["preferences_json"]):
                self.policy.evaluate(MemoryCandidate("preference", "project", "legacy.preference", value, "legacy", 0.3, source_refs=source))
            for value in _json_list(row["facts_json"]):
                self.policy.evaluate(MemoryCandidate("semantic", "project", "legacy.fact", value, "legacy", 0.3, source_refs=source))
            for value in _json_list(row["decisions_json"]):
                self.policy.evaluate(MemoryCandidate("procedural", "project", "legacy.decision", value, "legacy", 0.3, source_refs=source))
        self.store.mark_migration_done("legacy_memory_entries_v1")
        return count

    def _explicit_preference(self, task: str) -> str:
        markers = ("以后", "每次", "默认", "我希望", "我喜欢", "不要", "优先", "请记住")
        return _clip(task, 500) if any(marker in task for marker in markers) else ""


def _explicit_preference_v2(self, task: str) -> str:
    markers = ("以后", "每次", "默认", "我希望", "我喜欢", "不要", "优先", "请记住", "always", "prefer", "never")
    return _clip(task, 500) if any(marker in task for marker in markers) else ""


MemoryConsolidator._explicit_preference = _explicit_preference_v2


class MemoryUsefulnessTracker:
    def __init__(self, storage: Any | None = None) -> None:
        self.storage = storage

    def record_selected(
        self,
        results: list[MemorySearchResult],
        session_id: str = "",
        run_id: str = "",
        node: str = "",
    ) -> None:
        if not self.storage:
            return
        for item in results:
            self.storage.add_memory_usage(
                asdict(
                    MemoryUsageRecord(
                        memory_id=item.record.id,
                        session_id=session_id,
                        run_id=run_id,
                        node=node,
                        selected=True,
                    )
                )
            )

    def mark_answer_feedback(
        self,
        memory_ids: list[str],
        session_id: str = "",
        cited_in_answer: bool = False,
        helped_acceptance: bool = False,
        feedback: str | None = None,
    ) -> None:
        if not self.storage:
            return
        for memory_id in memory_ids:
            self.storage.update_memory_usage_feedback(
                memory_id,
                session_id=session_id,
                cited_in_answer=cited_in_answer,
                helped_acceptance=helped_acceptance,
                feedback=feedback,
            )


class MemoryGovernanceRuntime:
    def __init__(self, db_path: str | Path, storage: Any | None = None, min_active_confidence: float = 0.7) -> None:
        self.store = MemoryStore(db_path)
        self.policy = MemoryPolicyGate(self.store, min_active_confidence)
        self.retriever = MemoryRetriever(self.store)
        self.consolidator = MemoryConsolidator(self.store, self.policy)
        self.usage = MemoryUsefulnessTracker(storage)
        self.storage = storage

    def consolidate_run(self, **kwargs) -> list[MemoryRecord]:
        records = self.consolidator.consolidate_run(**kwargs)
        session_id = str(kwargs.get("session_id") or "")
        if self.storage and session_id:
            for record in records:
                event = "memory.promoted" if record.status == "active" else "memory.candidate.created"
                self.storage.add_event(session_id, event, asdict(record))
                if record.supersedes:
                    self.storage.add_event(
                        session_id,
                        "memory.superseded",
                        {"memory_id": record.supersedes, "superseded_by": record.id},
                    )
            self.storage.add_event(session_id, "memory.consolidated", {"records": len(records)})
        return records

    def evaluate_candidates(self, candidates: list[MemoryCandidate], session_id: str = "") -> list[MemoryRecord]:
        records = [self.policy.evaluate(candidate) for candidate in candidates]
        if self.storage and session_id:
            for record in records:
                event = "memory.promoted" if record.status == "active" else "memory.candidate.created"
                self.storage.add_event(session_id, event, asdict(record))
        return records

    def hot_write_user_task(self, user_task: str, namespace: str = "project", session_id: str = "") -> MemoryRecord | None:
        preference = self.consolidator._explicit_preference(user_task)
        if not preference:
            return None
        record = self.policy.evaluate(
            MemoryCandidate(
                "preference",
                namespace,
                "user.preference",
                preference,
                "user",
                1.0,
                source_refs=[f"session:{session_id}"] if session_id else [],
            )
        )
        if self.storage and session_id:
            self.storage.add_event(session_id, "memory.promoted", asdict(record))
            if record.supersedes:
                self.storage.add_event(
                    session_id,
                    "memory.superseded",
                    {"memory_id": record.supersedes, "superseded_by": record.id},
                )
        return record

    def retrieve(self, query: str, node: str = "planner", namespace: str = "project", limit: int = 8, session_id: str = "") -> list[MemorySearchResult]:
        results = self.retriever.retrieve(query, node, namespace, limit)
        self.usage.record_selected(results, session_id=session_id, node=node)
        if self.storage and session_id:
            self.storage.add_event(
                session_id,
                "memory.retrieved",
                {
                    "query": query,
                    "node": node,
                    "memory_ids": [item.record.id for item in results],
                    "items": [
                        {
                            "memory_id": item.record.id,
                            "score": item.score,
                            "type": item.record.memory_type,
                            "trust": item.record.trust,
                            "status": item.record.status,
                            "reasons": item.reasons,
                        }
                        for item in results
                    ],
                },
            )
        return results

    def invalidate(self, memory_id: str, session_id: str = "") -> bool:
        updated = self.policy.invalidate(memory_id)
        if updated and self.storage and session_id:
            self.storage.add_event(session_id, "memory.invalidated", {"memory_id": memory_id})
        return updated

    def explain(self, query: str, node: str = "planner", namespace: str = "project", limit: int = 8) -> list[dict[str, Any]]:
        return [
            {"record": asdict(item.record), "score": item.score, "reasons": item.reasons}
            for item in self.retrieve(query, node, namespace, limit)
        ]
