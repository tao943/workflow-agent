import uuid
from dataclasses import dataclass

from src.events import EventSink
from src.storage import Storage


def new_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:12]}"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class SessionContext:
    session_id: str
    message_id: str
    agent: str
    storage: Storage
    events: EventSink


def create_or_continue_session(
    storage: Storage,
    task: str,
    agent: str,
    output_format: str,
    session_id: str | None = None,
    parent_id: str | None = None,
) -> SessionContext:
    resolved_session_id = session_id or new_session_id()
    title = task[:50] + ("..." if len(task) > 50 else "")
    storage.create_session(resolved_session_id, title or "Untitled session", agent, parent_id=parent_id)
    message_id = new_id("msg")
    storage.add_message(message_id, resolved_session_id, "user", task)
    storage.add_message_part(new_id("part"), message_id, resolved_session_id, "text", {"role": "user", "content": task})
    events = EventSink(resolved_session_id, output_format=output_format)
    events.emit("session.created", title=title, agent=agent, parent_id=parent_id)
    events.emit("message.created", message_id=message_id, role="user", content=task)
    storage.add_event(resolved_session_id, "session.created", {"title": title, "agent": agent, "parent_id": parent_id})
    storage.add_event(resolved_session_id, "message.created", {"message_id": message_id, "role": "user", "content": task})
    return SessionContext(resolved_session_id, message_id, agent, storage, events)
