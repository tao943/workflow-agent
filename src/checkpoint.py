import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from src.state import AgentState


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
CHECKPOINT_DB = OUTPUT_DIR / "checkpoints" / "checkpoints.sqlite"


def create_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    checkpoint_db = Path(db_path) if db_path else CHECKPOINT_DB
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(checkpoint_db)
    try:
        _initialize(conn)
    except sqlite3.OperationalError:
        conn.close()
        _reset_broken_database(checkpoint_db)
        conn = sqlite3.connect(checkpoint_db)
        _initialize(conn)
    return conn


def _initialize(conn: sqlite3.Connection) -> None:
    # Some Windows-synced folders reject SQLite rollback journal writes.
    # This project stores small learning checkpoints, so journal_mode=OFF is
    # acceptable and keeps the demo portable in constrained directories.
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            user_task TEXT NOT NULL,
            status TEXT NOT NULL,
            state_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            node_name TEXT NOT NULL,
            state_json TEXT NOT NULL,
            checkpoint_id TEXT,
            parent_checkpoint_id TEXT,
            branch_id TEXT,
            context_snapshot_id TEXT,
            tool_effects_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        )
        """
    )
    for column, definition in (
        ("checkpoint_id", "TEXT"),
        ("parent_checkpoint_id", "TEXT"),
        ("branch_id", "TEXT"),
        ("context_snapshot_id", "TEXT"),
        ("tool_effects_json", "TEXT NOT NULL DEFAULT '[]'"),
    ):
        columns = [row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()]
        if column not in columns:
            conn.execute(f"ALTER TABLE events ADD COLUMN {column} {definition}")


def _reset_broken_database(checkpoint_db: Path = CHECKPOINT_DB) -> None:
    for path in (checkpoint_db, Path(str(checkpoint_db) + "-journal")):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def save_checkpoint(
    run_id: str,
    state: AgentState,
    status: str,
    node_name: str,
    db_path: str | Path | None = None,
    parent_checkpoint_id: str | None = None,
    branch_id: str = "main",
) -> str:
    state_json = json.dumps(state, ensure_ascii=False)
    now = datetime.now().isoformat(timespec="seconds")
    checkpoint_id = f"cp_{uuid.uuid4().hex[:12]}"
    tool_effects = [
        {
            "tool": item.get("tool"),
            "status": item.get("status"),
            "effect_id": item.get("effect_id"),
            "artifact": item.get("context_artifact_id") or item.get("raw_output_path"),
        }
        for item in state.get("tool_results", [])
        if item.get("effect_id") or item.get("context_artifact_id") or item.get("raw_output_path")
    ]
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, user_task, status, state_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                user_task = excluded.user_task,
                status = excluded.status,
                state_json = excluded.state_json,
                updated_at = excluded.updated_at
            """,
            (run_id, state["user_task"], status, state_json, now),
        )
        conn.execute(
            """
            INSERT INTO events (
                run_id, node_name, state_json, checkpoint_id, parent_checkpoint_id,
                branch_id, context_snapshot_id, tool_effects_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                node_name,
                state_json,
                checkpoint_id,
                parent_checkpoint_id,
                branch_id,
                state.get("context_snapshot_id", ""),
                json.dumps(tool_effects, ensure_ascii=False),
                now,
            ),
        )
    return checkpoint_id


def load_checkpoint(run_id: str, db_path: str | Path | None = None) -> AgentState:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT state_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    if row is None:
        raise ValueError(f"未找到 run_id：{run_id}")

    return json.loads(row[0])


def list_runs(limit: int = 10, db_path: str | Path | None = None) -> list[dict[str, str]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT run_id, user_task, status, updated_at
            FROM runs
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        {"run_id": row[0], "user_task": row[1], "status": row[2], "updated_at": row[3]}
        for row in rows
    ]


def list_checkpoints(run_id: str, db_path: str | Path | None = None) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT checkpoint_id, parent_checkpoint_id, branch_id, node_name,
                   context_snapshot_id, tool_effects_json, created_at
            FROM events WHERE run_id = ? ORDER BY id ASC
            """,
            (run_id,),
        ).fetchall()
    return [
        {
            "checkpoint_id": row[0],
            "parent_checkpoint_id": row[1],
            "branch_id": row[2] or "main",
            "node": row[3],
            "context_snapshot_id": row[4] or "",
            "tool_effects": json.loads(row[5] or "[]"),
            "created_at": row[6],
        }
        for row in rows
    ]


def fork_checkpoint(
    run_id: str,
    checkpoint_id: str,
    patch: dict | None = None,
    db_path: str | Path | None = None,
) -> tuple[str, AgentState]:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT state_json FROM events WHERE run_id = ? AND checkpoint_id = ?",
            (run_id, checkpoint_id),
        ).fetchone()
    if not row:
        raise ValueError(f"Unknown checkpoint: {checkpoint_id}")
    state = json.loads(row[0])
    state.update(patch or {})
    branch_id = f"branch_{uuid.uuid4().hex[:10]}"
    save_checkpoint(
        run_id,
        state,
        "running",
        "fork",
        db_path=db_path,
        parent_checkpoint_id=checkpoint_id,
        branch_id=branch_id,
    )
    return branch_id, state
