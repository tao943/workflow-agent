from pathlib import Path
from uuid import uuid4

from src.session import create_or_continue_session
from src.storage import Storage


def test_create_session_and_continue():
    storage = Storage(Path("outputs") / "test_sessions" / "agent_session_continue.sqlite")

    first = create_or_continue_session(storage, "hello", "build", "default", session_id="sess_test")
    second = create_or_continue_session(storage, "again", "build", "default", session_id="sess_test")

    assert first.session_id == "sess_test"
    assert second.session_id == "sess_test"
    assert storage.latest_session_id() == "sess_test"


def test_fork_records_parent():
    storage = Storage(Path("outputs") / "test_sessions" / "agent_session_fork.sqlite")

    create_or_continue_session(storage, "base", "build", "default", session_id="sess_base")
    forked = create_or_continue_session(storage, "child", "build", "default", session_id="sess_child", parent_id="sess_base")

    sessions = storage.list_sessions()
    assert forked.session_id == "sess_child"
    assert any(item["id"] == "sess_child" for item in sessions)


def test_storage_compacts_session_messages():
    storage = Storage(Path("outputs") / "test_sessions" / "agent_session_compact.sqlite")
    session_id = f"sess_{uuid4().hex[:8]}"
    storage.create_session(session_id, "title", "build")
    for index in range(8):
        role = "user" if index % 2 == 0 else "assistant"
        storage.add_message(f"msg_{uuid4().hex[:8]}_{index}", session_id, role, f"message {index}")

    summary = storage.compact_session(session_id, keep_tail_messages=2)

    assert "message 0" in summary
    assert "message 5" in summary
    assert "message 6" not in summary
    assert storage.get_session_summary(session_id) == summary


def test_storage_records_message_parts():
    storage = Storage(Path("outputs") / "test_sessions" / f"agent_parts_{uuid4().hex[:8]}.sqlite")
    storage.create_session("sess_parts", "parts", "build")
    storage.add_message("msg_parts", "sess_parts", "assistant", "hello")
    storage.add_message_part("part_text", "msg_parts", "sess_parts", "text", {"content": "hello"})

    parts = storage.list_message_parts("sess_parts")

    assert parts[0]["type"] == "text"
    assert parts[0]["content"]["content"] == "hello"
