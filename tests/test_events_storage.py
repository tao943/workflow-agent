from pathlib import Path

from src.graph import build_graph, make_initial_state
from src.storage import Storage


def test_permission_events_are_persisted():
    db_path = Path("outputs") / "test_events" / "permission_events.sqlite"
    session_id = "sess_permission_events"
    app = build_graph(llm=None)

    app.invoke(
        make_initial_state(
            "帮我制定一个三天学习 LangGraph 的计划，并保存为笔记",
            session_id=session_id,
            auto_approve=True,
            storage_path=str(db_path),
            permission_rules=[{"permission": "write", "pattern": "*", "action": "ask"}],
        )
    )

    events = Storage(db_path).list_events(session_id)
    event_types = [item["type"] for item in events]
    assert "permission.requested" in event_types
    assert "permission.replied" in event_types
