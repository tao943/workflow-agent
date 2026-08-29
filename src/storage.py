import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.storage_repositories import ContextRepository, EventRepository, PermissionRepository, SessionRepository, TeamRepository


DEFAULT_DB = Path("outputs/agent.sqlite")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Storage:
    def __init__(self, db_path: str | Path = DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()
        self.sessions = SessionRepository(self)
        self.events = EventRepository(self)
        self.teams = TeamRepository(self)
        self.context = ContextRepository(self)
        self.permissions = PermissionRepository(self)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=OFF")
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    status TEXT NOT NULL,
                    parent_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS steps (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    tool_name TEXT,
                    result TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    step_id TEXT,
                    tool_name TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    output TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS permissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    permission TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    action TEXT NOT NULL,
                    scope TEXT NOT NULL DEFAULT 'once',
                    source TEXT NOT NULL DEFAULT 'runtime',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_compactions (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    message_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS message_parts (
                    id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS context_artifacts (
                    id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    trust TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    invalidated_at TEXT,
                    invalidation_reason TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS context_snapshots (
                    id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS context_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    run_id TEXT,
                    node TEXT NOT NULL,
                    selected INTEGER NOT NULL,
                    cited_in_answer INTEGER NOT NULL DEFAULT 0,
                    helped_acceptance INTEGER NOT NULL DEFAULT 0,
                    feedback TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS teams (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS team_runs (
                    id TEXT PRIMARY KEY,
                    team_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    final_answer TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS team_members (
                    id TEXT PRIMARY KEY,
                    team_run_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS team_messages (
                    id TEXT PRIMARY KEY,
                    team_run_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS team_tasks (
                    id TEXT PRIMARY KEY,
                    team_run_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    assigned_to TEXT,
                    status TEXT NOT NULL,
                    result TEXT,
                    required_tools_json TEXT NOT NULL DEFAULT '[]',
                    required_evidence_json TEXT NOT NULL DEFAULT '[]',
                    evidence_status TEXT NOT NULL DEFAULT 'missing',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS team_member_runs (
                    id TEXT PRIMARY KEY,
                    team_run_id TEXT NOT NULL,
                    member_name TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS a2a_tasks (
                    team_run_id TEXT NOT NULL,
                    logical_task_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    business_attempt INTEGER NOT NULL,
                    execution_id TEXT NOT NULL UNIQUE,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    transport_retry_count INTEGER NOT NULL DEFAULT 0,
                    context_id TEXT NOT NULL,
                    remote_task_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (team_run_id, logical_task_id, business_attempt)
                );
                CREATE TABLE IF NOT EXISTS a2a_artifacts (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._ensure_column(conn, "permissions", "scope", "TEXT NOT NULL DEFAULT 'once'")
            self._ensure_column(conn, "permissions", "source", "TEXT NOT NULL DEFAULT 'runtime'")
            self._ensure_column(conn, "team_tasks", "required_tools_json", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(conn, "team_tasks", "required_evidence_json", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(conn, "team_tasks", "evidence_status", "TEXT NOT NULL DEFAULT 'missing'")
            self._ensure_column(conn, "team_tasks", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS context_artifacts_fts USING fts5(id UNINDEXED, summary, source_ref, kind, scope)"
                )
            except sqlite3.OperationalError:
                pass

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def upsert_a2a_task(self, *, team_run_id: str, logical_task_id: str, role: str, business_attempt: int, execution_id: str, idempotency_key: str, transport_retry_count: int, context_id: str, remote_task_id: str, endpoint: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute("""INSERT INTO a2a_tasks (team_run_id, logical_task_id, role, business_attempt, execution_id, idempotency_key, transport_retry_count, context_id, remote_task_id, endpoint, status, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(team_run_id, logical_task_id, business_attempt) DO UPDATE SET transport_retry_count=excluded.transport_retry_count, context_id=excluded.context_id, remote_task_id=excluded.remote_task_id, status=excluded.status, updated_at=excluded.updated_at""", (team_run_id, logical_task_id, role, business_attempt, execution_id, idempotency_key, transport_retry_count, context_id, remote_task_id, endpoint, status, _now()))

    def get_a2a_task_by_execution_id(self, execution_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM a2a_tasks WHERE execution_id = ?", (execution_id,)).fetchone()
            return dict(row) if row else None

    def list_a2a_tasks(self, execution_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM a2a_tasks WHERE (? IS NULL OR execution_id = ?)", (execution_id, execution_id)).fetchall()
            return [dict(row) for row in rows]

    def create_session(self, session_id: str, title: str, agent: str, parent_id: str | None = None) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, title, agent, status, parent_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                (session_id, title, agent, "running", parent_id, now, now),
            )

    def update_session_status(self, session_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
                (status, _now(), session_id),
            )

    def add_message(self, message_id: str, session_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (message_id, session_id, role, content, _now()),
            )

    def upsert_step(
        self,
        step_id: str,
        session_id: str,
        step_index: int,
        description: str,
        status: str,
        tool_name: str | None,
        result: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO steps (id, session_id, step_index, description, status, tool_name, result, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    tool_name = excluded.tool_name,
                    result = excluded.result,
                    updated_at = excluded.updated_at
                """,
                (step_id, session_id, step_index, description, status, tool_name, result, _now()),
            )

    def add_tool_call(
        self,
        call_id: str,
        session_id: str,
        step_id: str | None,
        tool_name: str,
        input_data: dict[str, Any],
        output: str | None,
        error: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tool_calls (id, session_id, step_id, tool_name, input_json, output, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (call_id, session_id, step_id, tool_name, json.dumps(input_data, ensure_ascii=False), output, error, _now()),
            )

    def add_permission(
        self,
        session_id: str,
        permission: str,
        pattern: str,
        action: str,
        scope: str = "once",
        source: str = "runtime",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO permissions (session_id, permission, pattern, action, scope, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, permission, pattern, action, scope, source, _now()),
            )

    def add_event(self, session_id: str, event_type: str, data: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (session_id, type, data_json, created_at) VALUES (?, ?, ?, ?)",
                (session_id, event_type, json.dumps(data, ensure_ascii=False), _now()),
            )

    def list_sessions(self, limit: int = 10) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, agent, status, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {"id": row[0], "title": row[1], "agent": row[2], "status": row[3], "updated_at": row[4]}
            for row in rows
        ]

    def latest_session_id(self) -> str | None:
        sessions = self.list_sessions(limit=1)
        return sessions[0]["id"] if sessions else None

    def list_events(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT type, data_json, created_at
                FROM events
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            {"type": row[0], "data": json.loads(row[1]), "created_at": row[2]}
            for row in rows
        ]

    def add_message_part(self, part_id: str, message_id: str, session_id: str, part_type: str, content: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO message_parts (id, message_id, session_id, type, content_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content_json = excluded.content_json,
                    created_at = excluded.created_at
                """,
                (part_id, message_id, session_id, part_type, json.dumps(content, ensure_ascii=False), _now()),
            )

    def list_message_parts(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, message_id, type, content_json, created_at
                FROM message_parts
                WHERE session_id = ?
                ORDER BY rowid ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            {"id": row[0], "message_id": row[1], "type": row[2], "content": json.loads(row[3]), "created_at": row[4]}
            for row in rows
        ]

    def add_context_artifact(self, artifact: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO context_artifacts (
                    id, content_hash, kind, file_path, summary, source_ref, scope,
                    trust, token_count, invalidated_at, invalidation_reason, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_hash) DO NOTHING
                """,
                (
                    artifact["id"],
                    artifact["content_hash"],
                    artifact["kind"],
                    artifact["file_path"],
                    artifact["summary"],
                    artifact.get("source_ref", ""),
                    artifact.get("scope", "project"),
                    artifact.get("trust", "unverified"),
                    int(artifact.get("token_count", 0)),
                    artifact.get("invalidated_at"),
                    None,
                    artifact.get("created_at") or _now(),
                ),
            )
            try:
                conn.execute("DELETE FROM context_artifacts_fts WHERE id = ?", (artifact["id"],))
                conn.execute(
                    "INSERT INTO context_artifacts_fts (id, summary, source_ref, kind, scope) VALUES (?, ?, ?, ?, ?)",
                    (
                        artifact["id"],
                        artifact["summary"],
                        artifact.get("source_ref", ""),
                        artifact["kind"],
                        artifact.get("scope", "project"),
                    ),
                )
            except sqlite3.OperationalError:
                pass

    def get_context_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, content_hash, kind, file_path, summary, source_ref, scope,
                       trust, token_count, created_at, invalidated_at
                FROM context_artifacts WHERE id = ?
                """,
                (artifact_id,),
            ).fetchone()
        return self._context_artifact_row(row)

    def get_context_artifact_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, content_hash, kind, file_path, summary, source_ref, scope,
                       trust, token_count, created_at, invalidated_at
                FROM context_artifacts WHERE content_hash = ?
                """,
                (content_hash,),
            ).fetchone()
        return self._context_artifact_row(row)

    def search_context_artifacts(self, query: str, scope: str = "", limit: int = 8) -> list[dict[str, Any]]:
        terms = [item.lower() for item in str(query).replace("_", " ").split() if len(item) > 1]
        if terms:
            try:
                fts_query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms[:12])
                with self._connect() as conn:
                    ids = [
                        row[0]
                        for row in conn.execute(
                            "SELECT id FROM context_artifacts_fts WHERE context_artifacts_fts MATCH ? ORDER BY bm25(context_artifacts_fts) LIMIT ?",
                            (fts_query, limit * 3),
                        ).fetchall()
                    ]
                ranked = []
                for artifact_id in ids:
                    item = self.get_context_artifact(artifact_id)
                    if item and item.get("invalidated_at") is None and (not scope or item["scope"] == scope or item["scope"].startswith(scope)):
                        ranked.append(item)
                if ranked:
                    return ranked[:limit]
            except sqlite3.OperationalError:
                pass
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, content_hash, kind, file_path, summary, source_ref, scope,
                       trust, token_count, created_at, invalidated_at
                FROM context_artifacts
                WHERE invalidated_at IS NULL
                  AND (? = '' OR scope = ? OR scope LIKE ?)
                ORDER BY created_at DESC
                LIMIT 100
                """,
                (scope, scope, f"{scope}%"),
            ).fetchall()
        ranked = []
        for row in rows:
            item = self._context_artifact_row(row)
            if not item:
                continue
            haystack = f"{item['summary']} {item['source_ref']} {item['kind']}".lower()
            score = sum(1 for term in terms if term in haystack)
            if not terms or score:
                ranked.append((score, item))
        return [item for _, item in sorted(ranked, key=lambda value: value[0], reverse=True)[:limit]]

    def invalidate_context_artifact(self, artifact_id: str, reason: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE context_artifacts SET invalidated_at = ?, invalidation_reason = ? WHERE id = ?",
                (_now(), reason, artifact_id),
            )

    def add_context_snapshot(self, snapshot_id: str, payload: dict[str, Any], file_path: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO context_snapshots (id, payload_json, file_path, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET payload_json = excluded.payload_json, file_path = excluded.file_path
                """,
                (snapshot_id, json.dumps(payload, ensure_ascii=False), file_path, _now()),
            )

    def get_context_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json, file_path, created_at FROM context_snapshots WHERE id = ?",
                (snapshot_id,),
            ).fetchone()
        if not row:
            return None
        return {"id": snapshot_id, "payload": json.loads(row[0]), "file_path": row[1], "created_at": row[2]}

    def add_context_feedback(self, session_id: str, fingerprint: str, decision: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO context_feedback (session_id, fingerprint, decision_json, created_at) VALUES (?, ?, ?, ?)",
                (session_id, fingerprint, json.dumps(decision, ensure_ascii=False), _now()),
            )

    def count_feedback_fingerprint(self, session_id: str, fingerprint: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM context_feedback WHERE session_id = ? AND fingerprint = ?",
                (session_id, fingerprint),
            ).fetchone()
        return int(row[0]) if row else 0

    def list_context_feedback(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT fingerprint, decision_json, created_at FROM context_feedback WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [{"fingerprint": row[0], "decision": json.loads(row[1]), "created_at": row[2]} for row in rows]

    def add_memory_usage(self, usage: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_usage (
                    memory_id, session_id, run_id, node, selected,
                    cited_in_answer, helped_acceptance, feedback, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    usage["memory_id"],
                    usage.get("session_id", ""),
                    usage.get("run_id", ""),
                    usage.get("node", ""),
                    1 if usage.get("selected", True) else 0,
                    1 if usage.get("cited_in_answer", False) else 0,
                    1 if usage.get("helped_acceptance", False) else 0,
                    usage.get("feedback"),
                    usage.get("created_at") or _now(),
                ),
            )

    def update_memory_usage_feedback(
        self,
        memory_id: str,
        session_id: str = "",
        cited_in_answer: bool | None = None,
        helped_acceptance: bool | None = None,
        feedback: str | None = None,
    ) -> None:
        assignments = []
        params: list[Any] = []
        if cited_in_answer is not None:
            assignments.append("cited_in_answer = ?")
            params.append(1 if cited_in_answer else 0)
        if helped_acceptance is not None:
            assignments.append("helped_acceptance = ?")
            params.append(1 if helped_acceptance else 0)
        if feedback is not None:
            assignments.append("feedback = ?")
            params.append(feedback)
        if not assignments:
            return
        with self._connect() as conn:
            conn.execute(
                f"UPDATE memory_usage SET {', '.join(assignments)} WHERE memory_id = ? AND (? = '' OR session_id = ?)",
                params + [memory_id, session_id, session_id],
            )

    def list_memory_usage(self, memory_id: str | None = None, session_id: str | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT memory_id, session_id, run_id, node, selected, cited_in_answer,
                   helped_acceptance, feedback, created_at
            FROM memory_usage
            WHERE (? IS NULL OR memory_id = ?) AND (? IS NULL OR session_id = ?)
            ORDER BY id DESC
        """
        with self._connect() as conn:
            rows = conn.execute(query, (memory_id, memory_id, session_id, session_id)).fetchall()
        return [
            {
                "memory_id": row[0],
                "session_id": row[1],
                "run_id": row[2],
                "node": row[3],
                "selected": bool(row[4]),
                "cited_in_answer": bool(row[5]),
                "helped_acceptance": bool(row[6]),
                "feedback": row[7],
                "created_at": row[8],
            }
            for row in rows
        ]

    def _context_artifact_row(self, row) -> dict[str, Any] | None:
        if not row:
            return None
        return {
            "id": row[0],
            "content_hash": row[1],
            "kind": row[2],
            "file_path": row[3],
            "summary": row[4],
            "source_ref": row[5],
            "scope": row[6],
            "trust": row[7],
            "token_count": row[8],
            "created_at": row[9],
            "invalidated_at": row[10],
        }

    def list_tool_calls(self, session_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if session_id:
                rows = conn.execute(
                    """
                    SELECT id, session_id, step_id, tool_name, input_json, output, error, created_at
                    FROM tool_calls WHERE session_id = ?
                    ORDER BY created_at ASC
                    """,
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, session_id, step_id, tool_name, input_json, output, error, created_at
                    FROM tool_calls
                    ORDER BY created_at DESC
                    LIMIT 100
                    """
                ).fetchall()
        return [
            {
                "id": row[0],
                "session_id": row[1],
                "step_id": row[2],
                "tool_name": row[3],
                "input": json.loads(row[4]),
                "output": row[5],
                "error": row[6],
                "created_at": row[7],
            }
            for row in rows
        ]

    def list_permissions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, permission, pattern, action, scope, source, created_at
                FROM permissions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "id": row[0],
                "session_id": row[1],
                "permission": row[2],
                "pattern": row[3],
                "action": row[4],
                "scope": row[5],
                "source": row[6],
                "created_at": row[7],
            }
            for row in rows
        ]

    def revoke_permission(self, permission_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM permissions WHERE id = ?", (permission_id,))
            return cursor.rowcount > 0

    def list_messages(self, session_id: str) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY rowid ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            {"id": row[0], "role": row[1], "content": row[2], "created_at": row[3]}
            for row in rows
        ]

    def get_session_summary(self, session_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM session_compactions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row[0] if row else ""

    def compact_session(self, session_id: str, keep_tail_messages: int = 6, limit: int = 4000) -> str:
        messages = self.list_messages(session_id)
        if len(messages) <= keep_tail_messages:
            return self.get_session_summary(session_id)

        compacted_messages = messages[:-keep_tail_messages]
        lines = [
            "以下是该 session 较早消息的本地压缩摘要，最近消息仍保留在 session 表中："
        ]
        for item in compacted_messages:
            content = " ".join(item["content"].split())
            if len(content) > 500:
                content = f"{content[:500]}..."
            lines.append(f"- {item['role']} @ {item['created_at']}: {content}")
        summary = "\n".join(lines)
        if len(summary) > limit:
            summary = f"{summary[:limit]}..."

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_compactions (session_id, summary, message_count, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary = excluded.summary,
                    message_count = excluded.message_count,
                    updated_at = excluded.updated_at
                """,
                (session_id, summary, len(compacted_messages), _now()),
            )
        self.add_message_part(
            f"part_compaction_{session_id}",
            f"msg_compaction_{session_id}",
            session_id,
            "compaction",
            {"summary": summary, "message_count": len(compacted_messages)},
        )
        return summary

    def create_team(self, team_id: str, name: str, description: str, spec: dict[str, Any], status: str = "active") -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO teams (id, name, description, spec_json, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    spec_json = excluded.spec_json,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (team_id, name, description, json.dumps(spec, ensure_ascii=False), status, now, now),
            )

    def list_teams(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, name, description, spec_json, status, created_at, updated_at
                FROM teams
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "spec": json.loads(row[3]),
                "status": row[4],
                "created_at": row[5],
                "updated_at": row[6],
            }
            for row in rows
        ]

    def delete_team(self, team_id: str) -> bool:
        with self._connect() as conn:
            active = conn.execute(
                "SELECT COUNT(*) FROM team_runs WHERE team_id = ? AND status = 'running'",
                (team_id,),
            ).fetchone()[0]
            if active:
                return False
            cursor = conn.execute("DELETE FROM teams WHERE id = ?", (team_id,))
            return cursor.rowcount > 0

    def create_team_run(self, run_id: str, team_id: str, session_id: str, task: str, status: str = "running") -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO team_runs (id, team_id, session_id, task, status, final_answer, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, team_id, session_id, task, status, "", now, now),
            )

    def update_team_run(self, run_id: str, status: str, final_answer: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE team_runs
                SET status = ?, final_answer = COALESCE(?, final_answer), updated_at = ?
                WHERE id = ?
                """,
                (status, final_answer, _now(), run_id),
            )

    def get_team_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, team_id, session_id, task, status, final_answer, created_at, updated_at
                FROM team_runs WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "team_id": row[1],
            "session_id": row[2],
            "task": row[3],
            "status": row[4],
            "final_answer": row[5],
            "created_at": row[6],
            "updated_at": row[7],
        }

    def list_team_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, team_id, session_id, task, status, final_answer, created_at, updated_at
                FROM team_runs ORDER BY updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "id": row[0],
                "team_id": row[1],
                "session_id": row[2],
                "task": row[3],
                "status": row[4],
                "final_answer": row[5],
                "created_at": row[6],
                "updated_at": row[7],
            }
            for row in rows
        ]

    def add_team_member(self, member_id: str, team_run_id: str, name: str, role: str, profile: str, session_id: str, status: str = "idle") -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO team_members (id, team_run_id, name, role, profile, session_id, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (member_id, team_run_id, name, role, profile, session_id, status, now, now),
            )

    def update_team_member(self, team_run_id: str, name: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE team_members SET status = ?, updated_at = ? WHERE team_run_id = ? AND name = ?",
                (status, _now(), team_run_id, name),
            )

    def list_team_members(self, team_run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, name, role, profile, session_id, status, created_at, updated_at
                FROM team_members WHERE team_run_id = ? ORDER BY rowid ASC
                """,
                (team_run_id,),
            ).fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "role": row[2],
                "profile": row[3],
                "session_id": row[4],
                "status": row[5],
                "created_at": row[6],
                "updated_at": row[7],
            }
            for row in rows
        ]

    def add_team_message(self, message_id: str, team_run_id: str, sender: str, recipient: str, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO team_messages (id, team_run_id, sender, recipient, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, team_run_id, sender, recipient, json.dumps(payload, ensure_ascii=False), _now()),
            )

    def list_team_messages(self, team_run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, sender, recipient, payload_json, created_at
                FROM team_messages WHERE team_run_id = ? ORDER BY rowid ASC
                """,
                (team_run_id,),
            ).fetchall()
        return [
            {"id": row[0], "sender": row[1], "recipient": row[2], "payload": json.loads(row[3]), "created_at": row[4]}
            for row in rows
        ]

    def add_team_task(
        self,
        task_id: str,
        team_run_id: str,
        title: str,
        description: str,
        assigned_to: str | None,
        status: str = "pending",
        required_tools: list[str] | None = None,
        required_evidence: list[str] | None = None,
        evidence_status: str = "missing",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO team_tasks (
                    id, team_run_id, title, description, assigned_to, status, result,
                    required_tools_json, required_evidence_json, evidence_status, metadata_json,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    description = excluded.description,
                    assigned_to = excluded.assigned_to,
                    status = excluded.status,
                    required_tools_json = excluded.required_tools_json,
                    required_evidence_json = excluded.required_evidence_json,
                    evidence_status = excluded.evidence_status,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    task_id,
                    team_run_id,
                    title,
                    description,
                    assigned_to,
                    status,
                    "",
                    json.dumps(required_tools or [], ensure_ascii=False),
                    json.dumps(required_evidence or [], ensure_ascii=False),
                    evidence_status,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )

    def update_team_task(
        self,
        task_id: str,
        status: str,
        result: str | None = None,
        assigned_to: str | None = None,
        evidence_status: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE team_tasks
                SET status = ?,
                    result = COALESCE(?, result),
                    assigned_to = COALESCE(?, assigned_to),
                    evidence_status = COALESCE(?, evidence_status),
                    updated_at = ?
                WHERE id = ?
                """,
                (status, result, assigned_to, evidence_status, _now(), task_id),
            )

    def update_team_task_metadata(self, task_id: str, updates: dict[str, Any]) -> None:
        current = self.get_team_task(task_id)
        if not current:
            return
        metadata = dict(current.get("metadata") or {})
        metadata.update(updates)
        with self._connect() as conn:
            conn.execute(
                "UPDATE team_tasks SET metadata_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(metadata, ensure_ascii=False), _now(), task_id),
            )

    def list_team_tasks(self, team_run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, description, assigned_to, status, result, required_tools_json, required_evidence_json, evidence_status, metadata_json, created_at, updated_at
                FROM team_tasks WHERE team_run_id = ? ORDER BY rowid ASC
                """,
                (team_run_id,),
            ).fetchall()
        return [
            {
                "id": row[0],
                "title": row[1],
                "description": row[2],
                "assigned_to": row[3],
                "status": row[4],
                "result": row[5],
                "required_tools": json.loads(row[6]),
                "required_evidence": json.loads(row[7]),
                "evidence_status": row[8],
                "metadata": json.loads(row[9]),
                "created_at": row[10],
                "updated_at": row[11],
            }
            for row in rows
        ]

    def get_team_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, team_run_id, title, description, assigned_to, status, result, required_tools_json, required_evidence_json, evidence_status, metadata_json, created_at, updated_at
                FROM team_tasks WHERE id = ?
                """,
                (task_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "team_run_id": row[1],
            "title": row[2],
            "description": row[3],
            "assigned_to": row[4],
            "status": row[5],
            "result": row[6],
            "required_tools": json.loads(row[7]),
            "required_evidence": json.loads(row[8]),
            "evidence_status": row[9],
            "metadata": json.loads(row[10]),
            "created_at": row[11],
            "updated_at": row[12],
        }

    def add_team_member_run(self, member_run_id: str, team_run_id: str, member_name: str, session_id: str, status: str, result: dict[str, Any]) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO team_member_runs (id, team_run_id, member_name, session_id, status, result_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (member_run_id, team_run_id, member_name, session_id, status, json.dumps(result, ensure_ascii=False), now, now),
            )

    def list_team_member_runs(self, team_run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, member_name, session_id, status, result_json, created_at, updated_at
                FROM team_member_runs WHERE team_run_id = ? ORDER BY rowid ASC
                """,
                (team_run_id,),
            ).fetchall()
        return [
            {
                "id": row[0],
                "member_name": row[1],
                "session_id": row[2],
                "status": row[3],
                "result": json.loads(row[4]),
                "created_at": row[5],
                "updated_at": row[6],
            }
            for row in rows
        ]
